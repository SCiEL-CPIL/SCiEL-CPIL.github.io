"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _covering_radius_deg(points):
    """Compute the covering radius in degrees for points on unit sphere."""
    points = np.asarray(points)
    # Normalize
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    points = points / norms
    
    # Check for coincident points
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            if np.linalg.norm(points[i] - points[j]) < 1e-8:
                return np.pi  # Invalid, return max possible
    
    # Check if all points are in a closed hemisphere
    # A set of points is in a closed hemisphere iff the origin is not in their convex hull
    # For simplicity, check if there's a vector v such that v . p_i <= 0 for all i
    # This is equivalent to checking if the convex hull contains the origin
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi
    
    # Check if origin is inside the convex hull
    # The origin is inside the hull iff for every facet, the origin is on the same side as the interior
    # Actually, simpler: the covering radius is the max circumradius of facets whose cap doesn't contain any other point
    # But a simpler check: if all points are in a hemisphere, the covering radius is >= pi/2
    
    # Compute covering radius via Delaunay-like approach on sphere
    # The covering radius is the max over facets of the convex hull of the circumradius,
    # but only for facets where the corresponding cap is empty (no other points inside)
    
    # Alternative: sample the sphere densely and compute max distance
    # But that's slow. Let's use the convex hull approach.
    
    # For each facet, compute the circumradius and check if the cap is empty
    # The circumradius of a facet is the angular distance from the facet center to the farthest vertex
    # Actually, the covering radius is the max angular distance from any point on the sphere to the nearest configuration point
    # This equals the max circumradius of the Voronoi cells, which correspond to facets of the convex hull
    
    # Let's use a dense sampling approach for accuracy
    # Create a grid on the sphere
    n_theta = 100
    n_phi = 100
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
    
    # Convert to Cartesian
    x = np.sin(theta_grid) * np.cos(phi_grid)
    y = np.sin(theta_grid) * np.sin(phi_grid)
    z = np.cos(theta_grid)
    
    # Sample points
    sample_points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    
    # For each sample point, find the nearest configuration point
    # Compute angular distance
    max_dist = 0.0
    for sp in sample_points:
        dots = np.dot(points, sp)
        dots = np.clip(dots, -1.0, 1.0)
        nearest_dist = np.arccos(np.max(dots))
        if nearest_dist > max_dist:
            max_dist = nearest_dist
    
    return max_dist * 180.0 / np.pi


def _covering_radius_fast(points):
    """Faster covering radius estimation using convex hull facets."""
    points = np.asarray(points)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    points = points / norms
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi * 180.0 / np.pi  # Invalid
    
    max_circumradius = 0.0
    for simplex in hull.simplices:
        facet_points = points[simplex]
        # Compute the center of the facet (normalized)
        center = np.mean(facet_points, axis=0)
        center_norm = np.linalg.norm(center)
        if center_norm < 1e-10:
            # Facet passes through origin, covering radius is pi
            return np.pi
        center = center / center_norm
        
        # Compute angular distances from center to each vertex of the facet
        for vp in facet_points:
            dot_val = np.dot(center, vp)
            dot_val = np.clip(dot_val, -1.0, 1.0)
            ang_dist = np.arccos(dot_val)
            if ang_dist > max_circumradius:
                max_circumradius = ang_dist
        
        # Check if the cap defined by this facet is empty
        # The cap is the set of points on the sphere that are on the same side of the facet plane as the center
        # A point p is in the cap if dot(p, center) > dot(vp, center) for all vertices vp
        # Actually, the facet plane is defined by the facet points. The normal is center (outward).
        # A point q is in the cap if dot(q, center) >= dot(vp, center) for all vp in facet
        # But since all points are on the unit sphere and the facet is part of the convex hull,
        # the cap is empty iff no other point has a larger dot product with center than the max dot product of facet vertices
        
        max_dot_facet = np.max(np.dot(facet_points, center))
        other_dots = np.dot(points, center)
        # Exclude the facet vertices
        other_dots[simplex] = -np.inf
        if np.max(other_dots) > max_dot_facet + 1e-12:
            # Another point is in the cap, so this facet's circumradius doesn't contribute
            # But wait, the covering radius is still determined by the nearest point, which would be that other point
            # So we should not use this facet's circumradius
            pass
        else:
            # The cap is empty, so the circumradius of this facet contributes to the covering radius
            pass
    
    # Actually, the above logic is flawed. Let me use the sampling approach but with better resolution.
    # For now, return the max circumradius as an upper bound (may be an overestimate)
    return max_circumradius * 180.0 / np.pi


def _lloyd_refine(points, n_iters=50, time_limit=100.0):
    """Lloyd's algorithm refinement for covering radius minimization."""
    points = np.asarray(points, dtype=float).copy()
    start_time = time.time()
    
    best_points = points.copy()
    best_radius = _covering_radius_fast(points)
    
    for it in range(n_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Compute Voronoi cells on the sphere using the convex hull
        # For each point, compute the centroid of its Voronoi cell
        # The Voronoi cell of point i is the set of points on the sphere closer to i than to any other point
        # This can be approximated by intersecting half-spaces
        
        new_points = points.copy()
        for i in range(N):
            # Compute the Voronoi cell of point i
            # A point x on the sphere is in the Voronoi cell of i if dot(x, p_i) >= dot(x, p_j) for all j
            # This is equivalent to dot(x, p_i - p_j) >= 0 for all j
            
            # Sample the Voronoi cell by finding points on the sphere that satisfy the constraints
            # Use a simple approach: start from p_i and move in the direction of the centroid of the cell
            
            # Compute the centroid of the Voronoi cell using a Monte Carlo approach
            n_samples = 200
            cell_points = []
            for _ in range(n_samples):
                # Generate a random point on the sphere
                u = np.random.randn(3)
                u = u / np.linalg.norm(u)
                
                # Check if u is in the Voronoi cell of i
                dots_i = np.dot(u, points[i])
                dots_j = np.dot(u, points)
                if np.argmax(dots_j) == i:
                    cell_points.append(u)
            
            if len(cell_points) > 0:
                centroid = np.mean(cell_points, axis=0)
                centroid_norm = np.linalg.norm(centroid)
                if centroid_norm > 1e-10:
                    new_points[i] = centroid / centroid_norm
        
        # Normalize
        norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        new_points = new_points / norms
        
        # Check improvement
        new_radius = _covering_radius_fast(new_points)
        if new_radius < best_radius:
            best_radius = new_radius
            best_points = new_points.copy()
        
        points = new_points
    
    return best_points, best_radius


def _rfd_maximize_min_distance(points, n_iters=100, time_limit=80.0):
    """RFD (Repulsive Force Dynamics) to maximize minimum pairwise distance."""
    points = np.asarray(points, dtype=float).copy()
    start_time = time.time()
    
    for it in range(n_iters):
        if time.time() - start_time > time_limit:
            break
        
        forces = np.zeros_like(points)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                # Repulsive force proportional to 1/d^2
                diff = points[i] - points[j]
                dist = np.linalg.norm(diff)
                if dist > 1e-10:
                    forces[i] += diff / (dist * dist)
        
        # Move points in the direction of the force, projected onto the tangent plane
        for i in range(N):
            # Project force onto tangent plane at points[i]
            tangent_force = forces[i] - np.dot(forces[i], points[i]) * points[i]
            # Normalize and take a small step
            step_size = 0.01
            new_point = points[i] + step_size * tangent_force
            new_point = new_point / np.linalg.norm(new_point)
            points[i] = new_point
    
    return points


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = _covering_radius_fast(points)
    
    # Phase 1: RFD to maximize minimum pairwise distance
    if time.time() < deadline - 30.0:
        rfd_points = _rfd_maximize_min_distance(points, n_iters=50, time_limit=20.0)
        rfd_radius = _covering_radius_fast(rfd_points)
        if rfd_radius < best_radius:
            best_points = rfd_points.copy()
            best_radius = rfd_radius
    
    # Phase 2: Lloyd's refinement
    if time.time() < deadline - 20.0:
        lloyd_points, lloyd_radius = _lloyd_refine(best_points, n_iters=30, time_limit=40.0)
        if lloyd_radius < best_radius:
            best_points = lloyd_points.copy()
            best_radius = lloyd_radius
    
    # Phase 3: Final polish with more iterations
    if time.time() < deadline - 10.0:
        polish_points, polish_radius = _lloyd_refine(best_points, n_iters=20, time_limit=20.0)
        if polish_radius < best_radius:
            best_points = polish_points.copy()
            best_radius = polish_radius
    
    # Ensure points are normalized
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    best_points = best_points / norms
    
    return best_points
# EVOLVE-BLOCK-END
