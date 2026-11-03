"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import sys

def _compute_covering_radius(points):
    """Compute the covering radius in degrees using convex hull facets."""
    try:
        hull = ConvexHull(points)
        max_radius = 0.0
        for simplex in hull.simplices:
            # Get the facet points
            facet_points = points[simplex]
            # Find the circumcenter of the facet
            # For a triangle on the unit sphere, the circumcenter is the normalized cross product
            # of two edges from the centroid, or we can use the formula for circumcenter of a triangle
            # Actually, for points on a unit sphere, the covering radius for a facet is the angle
            # from the facet's circumcenter to the farthest vertex.
            # The circumcenter of a spherical triangle can be found by normalizing the vector
            # that is equidistant from all three vertices.
            
            # For a triangle with vertices a, b, c on the unit sphere:
            # The circumcenter is proportional to a*(|b|^2*|c|^2 - ...) but simpler:
            # It's the normalized vector (a x b) x (b x c) or we can solve the linear system.
            
            # Simple approach: the circumcenter direction is given by the normalized
            # vector that satisfies dot(circ, a) = dot(circ, b) = dot(circ, c)
            # This means circ is in the span of (a-b) and (a-c) perpendicular... 
            # Actually, circ = normalize(a * (b·c - 0.5) + ... ) 
            # Let's use the formula: for unit vectors a,b,c, the circumcenter direction is
            # a*(sin(2*angle_bc)) + b*(sin(2*angle_ca)) + c*(sin(2*angle_ab)) normalized? No.
            
            # Better: solve for the point on the sphere equidistant from a, b, c.
            # Let d be the circumcenter. d·a = d·b = d·c = cos(r) where r is the circumradius.
            # d = α*a + β*b + γ*c, with d·a = d·b = d·c.
            # This gives: α + β*(b·a) + γ*(c·a) = α*(a·b) + β + γ*(c·b) = ...
            
            # Simpler numerical approach: use the fact that the circumcenter is the 
            # normalized vector (a × b) × (b × c) ... let me just use a direct formula.
            
            a, b, c = facet_points[0], facet_points[1], facet_points[2]
            # Circumcenter of triangle on unit sphere:
            # d = (a * (b·c) + b * (c·a) + c * (a·b) - (a·b)*(b·c)*c - ...) 
            # Let's just use: the circumcenter is the normalized vector
            # that is the cross product of (b-a) and (c-a) rotated... 
            # Actually the simplest: d ∝ a*(|b-c|²) + b*(|c-a|²) + c*(|a-b|²)
            # No, that's the incenter.
            
            # For circumcenter: d·(a-b) = 0 and d·(b-c) = 0 (d is equidistant from a,b,c)
            # So d is perpendicular to (a-b) and (b-c), hence d ∝ (a-b) × (b-c)
            v1 = a - b
            v2 = b - c
            d = np.cross(v1, v2)
            norm_d = np.linalg.norm(d)
            if norm_d < 1e-12:
                # Degenerate, use centroid
                d = (a + b + c) / 3.0
                norm_d = np.linalg.norm(d)
            d = d / norm_d
            
            # The circumradius is the angle from d to any vertex
            cos_r = np.clip(np.dot(d, a), -1.0, 1.0)
            r = np.arccos(cos_r)
            max_radius = max(max_radius, r)
        
        return np.degrees(max_radius)
    except Exception:
        return 180.0  # Fallback for invalid configurations


def _compute_covering_radius_fast(points):
    """Faster covering radius computation using vectorized operations."""
    try:
        hull = ConvexHull(points)
        max_cos = 1.0  # We want to minimize the angle, so maximize the min cos
        min_cos = 1.0
        for simplex in hull.simplices:
            a, b, c = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            v1 = a - b
            v2 = b - c
            d = np.cross(v1, v2)
            norm_d = np.linalg.norm(d)
            if norm_d < 1e-12:
                d = (a + b + c)
                norm_d = np.linalg.norm(d)
            d = d / norm_d
            cos_r = np.dot(d, a)
            min_cos = min(min_cos, cos_r)
        
        return np.degrees(np.arccos(np.clip(min_cos, -1.0, 1.0)))
    except Exception:
        return 180.0


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = _compute_covering_radius_fast(points)
    
    # Gradient-free optimization using Nelder-Mead or similar
    # Since we have 390 parameters, we need an efficient approach.
    # Let's try a local search approach: move points to reduce the covering radius.
    
    # Strategy: iteratively find the "worst" point (the one whose Voronoi cell has the largest circumradius)
    # and move it toward the center of that cell.
    
    def _local_optimize(pts, max_iters=500):
        """Local optimization by moving points to reduce covering radius."""
        pts = pts.copy()
        for iteration in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            # Compute the covering radius for each point (approximate)
            # For each point, find its "Voronoi cell" on the sphere and compute the circumradius
            # This is expensive, so let's use a simpler heuristic.
            
            # Compute pairwise distances
            dots = np.clip(pts @ pts.T, -1.0, 1.0)
            dists = np.arccos(dots)
            
            # For each point, find its nearest neighbors and estimate local covering radius
            # Move each point slightly toward the centroid of its farthest neighbors
            improvements = 0.0
            for k in range(N):
                if time.time() > deadline - 5.0:
                    break
                
                # Find the point's "worst direction" - the direction where the gap to nearest neighbors is largest
                # Use the antipodal point as a proxy for the worst case
                antipode = -pts[k]
                
                # Find neighbors of k
                neighbor_dists = dists[k].copy()
                neighbor_dists[k] = np.inf
                # Get the 10 nearest neighbors
                nn_idx = np.argsort(neighbor_dists)[:10]
                nn_pts = pts[nn_idx]
                
                # The worst direction is roughly the antipode of the centroid of nearest neighbors
                centroid = nn_pts.mean(axis=0)
                centroid_norm = np.linalg.norm(centroid)
                if centroid_norm > 1e-12:
                    centroid /= centroid_norm
                
                # Move point k slightly toward the direction that maximizes coverage
                # This is heuristic: move toward the antipode of the nearest neighbor centroid
                move_dir = antipode - centroid
                move_norm = np.linalg.norm(move_dir)
                if move_norm > 1e-12:
                    move_dir /= move_norm
                    # Small step
                    step = 0.01
                    new_pt = pts[k] + step * move_dir
                    new_pt = new_pt / np.linalg.norm(new_pt)
                    
                    # Check if this improves the local covering radius
                    old_local_radius = np.max(neighbor_dists[:10])
                    pts[k] = new_pt
                    dots_new = np.clip(pts @ pts.T, -1.0, 1.0)
                    dists_new = np.arccos(dots_new)
                    new_neighbor_dists = dists_new[k].copy()
                    new_neighbor_dists[k] = np.inf
                    new_local_radius = np.max(new_neighbor_dists[:10])
                    
                    if new_local_radius < old_local_radius:
                        improvements += 1.0
                    else:
                        pts[k] = pts[pts[k] == pts[k]]  # revert
                        # Actually, let's just revert properly
                        pts[k] = pts[k]  # This doesn't work, let me fix
            # This approach is getting complex. Let me use a simpler method.
        return pts
    
    # Let's use a cleaner approach: iterative closest point + local moves
    # Use scipy's minimize with a reduced parameterization
    
    # Actually, let's try a different strategy: use the fact that the optimal configuration
    # is known to be close to the Hardin-Sloane-Smith result. Let's start from a better initial
    # configuration and do local refinement.
    
    # Better initial configuration: use a more uniform distribution
    # Try multiple random restarts with local optimization
    
    def _local_refine(pts):
        """Local refinement using steepest descent on the covering radius."""
        pts = pts.copy()
        lr = 0.005
        for it in range(200):
            if time.time() > deadline - 5.0:
                break
            
            # Compute gradient of covering radius w.r.t. each point
            # The covering radius is determined by the worst facet
            try:
                hull = ConvexHull(pts)
            except Exception:
                break
            
            # Find the facet with the largest circumradius
            max_r = 0.0
            worst_simplex = None
            worst_circ = None
            for simplex in hull.simplices:
                a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
                v1 = a - b
                v2 = b - c
                d = np.cross(v1, v2)
                norm_d = np.linalg.norm(d)
                if norm_d < 1e-12:
                    continue
                d = d / norm_d
                cos_r = np.dot(d, a)
                r = np.arccos(np.clip(cos_r, -1.0, 1.0))
                if r > max_r:
                    max_r = r
                    worst_simplex = simplex
                    worst_circ = d
            
            if worst_simplex is None:
                break
            
            # Move the three vertices of the worst facet toward the circumcenter
            # to reduce the circumradius
            for idx in worst_simplex:
                move_dir = worst_circ - pts[idx]
                move_norm = np.linalg.norm(move_dir)
                if move_norm > 1e-12:
                    move_dir /= move_norm
                    new_pt = pts[idx] + lr * move_dir
                    new_pt = new_pt / np.linalg.norm(new_pt)
                    pts[idx] = new_pt
        
        return pts
    
    # Run local refinement
    points = _local_refine(points)
    radius = _compute_covering_radius_fast(points)
    if radius < best_radius:
        best_radius = radius
        best_points = points.copy()
    
    # Try a few more iterations of refinement
    for _ in range(3):
        if time.time() > deadline - 10.0:
            break
        points = _local_refine(points)
        radius = _compute_covering_radius_fast(points)
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()
    
    # Ensure all points are on the unit sphere
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points

# EVOLVE-BLOCK-END
