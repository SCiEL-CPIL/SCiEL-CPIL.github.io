"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull, Delaunay
import scipy.optimize as optimize

def _normalize(v):
    v = np.asarray(v, dtype=float)
    if v.ndim == 1:
        norm = np.linalg.norm(v)
        if norm < 1e-12:
            return v
        return v / norm
    else:
        norms = np.linalg.norm(v, axis=1)
        norms[norms < 1e-12] = 1.0
        return v / norms[:, np.newaxis]

def _covering_radius_exact(pts):
    """
    Compute the exact covering radius.
    The covering radius is the maximum angular distance from any point on the sphere 
    to the nearest point in pts. This occurs at the circumcenters of the Delaunay triangles 
    (facets of the convex hull) that lie within the triangle, or at vertices/edges.
    """
    try:
        hull = ConvexHull(pts)
        tri_idx = hull.simplices[:, :3]
        
        max_angle = 0.0
        # Vectorized circumcenter calculation is tricky due to edge cases, 
        # but N=50 is small, so a loop is fast enough.
        for tri in tri_idx:
            p1, p2, p3 = pts[tri[0]], pts[tri[1]], pts[tri[2]]
            
            # Calculate circumcenter of the spherical triangle p1, p2, p3.
            # The circumcenter C is the point on the unit sphere equidistant from p1, p2, p3.
            # It lies in the direction of the cross product of (p2-p1) and (p3-p1)? No.
            
            # Let C = a*p1 + b*p2 + c*p3.
            # C dot p1 = C dot p2 = C dot p3 = cos(R).
            # |C|^2 = 1.
            
            # A simpler geometric approach:
            # The circumcenter is the intersection of the perpendicular bisectors.
            # In 3D, the circumcenter of a triangle on the unit sphere is the normalized vector:
            # C = normalize( cross(p2-p1, p3-p1) + ... ) -- this is the normal to the plane, not the circumcenter.
            
            # Correct formula for circumcenter of a spherical triangle:
            # Let u = p1, v = p2, w = p3.
            # The circumcenter is proportional to:
            # C_dir = cross(v-u, w-u) is the normal to the plane.
            # Actually, the circumcenter is the solution to:
            # C . (v-u) = 0.5 * (|v|^2 - |u|^2) = 0 (since |v|=|u|=1) -> C . (v-u) = 0
            # C . (w-u) = 0
            # C . C = 1
            
            # This means C is orthogonal to (v-u) and (w-u).
            # So C is parallel to cross(v-u, w-u).
            # Let n = cross(v-u, w-u).
            # C = +/- n / ||n||.
            # We need to choose the sign such that C is "inside" the triangle, i.e., C . p1 > 0, etc.
            # Actually, the circumcenter of the triangle formed by points on the unit sphere 
            # is NOT simply the normal to the plane. The normal to the plane is the pole.
            # The circumcenter is the point on the sphere equidistant from the vertices.
            
            # Let's solve the linear system:
            # C . p1 = k
            # C . p2 = k
            # C . p3 = k
            # C . C = 1
            
            # From the first three, C is in the null space of:
            # [p1; p2; p3] * [1, -1, 0] ... no.
            # C . (p1 - p2) = 0
            # C . (p1 - p3) = 0
            
            # So C is parallel to n = cross(p1 - p2, p1 - p3).
            n = np.cross(p1 - p2, p1 - p3)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-10:
                continue
            
            C1 = n / n_norm
            C2 = -C1
            
            # Check which one is the circumcenter of the "small" triangle (inside the convex hull facet)
            # The circumcenter must be in the same hemisphere as the triangle vertices relative to the plane?
            # Actually, for a convex hull facet, the circumcenter is the one that is "outside" the polytope 
            # if the triangle is acute? No, the Voronoi vertex is the circumcenter.
            
            # Let's check the distance to p1.
            # If C is the circumcenter, C . p1 = cos(R).
            # We want the R that corresponds to the Voronoi cell.
            # The Voronoi cell is the set of points closer to p1 than any other.
            # The vertex of the Voronoi cell is the circumcenter.
            
            # There are two candidates: C1 and C2.
            # One is the circumcenter of the triangle on the "short" arc, the other on the "long" arc.
            # We want the one that is the center of the cap covering the triangle.
            # This is the one where the angle is < 90 degrees? Not necessarily.
            
            # Heuristic: The correct circumcenter is the one that has a positive dot product 
            # with the normal of the facet pointing OUTWARDS?
            # The normal of the facet (outward) is n_out = cross(p2-p1, p3-p1).
            # Note: My n above is cross(p1-p2, p1-p3) = cross(-(p2-p1), -(p3-p1)) = cross(p2-p1, p3-p1).
            # So n is the outward normal (assuming right-hand rule order p1,p2,p3 matches hull orientation).
            # scipy's ConvexHull gives simplices in counter-clockwise order when viewed from outside.
            # So n = cross(p2-p1, p3-p1) is the outward normal.
            
            # The circumcenter C must satisfy C . n > 0?
            # Let's test. If the triangle is small, the circumcenter is close to the centroid.
            # The centroid g = (p1+p2+p3)/3. g . n > 0.
            # So C should have the same sign as g . n.
            
            g = (p1 + p2 + p3) / 3.0
            sign = np.sign(g @ n)
            if sign == 0:
                sign = 1.0
            
            C = C1 if (C1 @ n) * sign > 0 else C2
            
            # Calculate angular distance from C to p1 (or p2, p3, they are equal)
            dot_val = np.clip(C @ p1, -1.0, 1.0)
            angle = np.arccos(dot_val)
            
            # We must ensure C is actually the center of the Voronoi cell.
            # This requires C to be inside the spherical triangle.
            # If the triangle is obtuse, the circumcenter might be outside.
            # If it's outside, the covering radius for this facet is determined by the longest edge midpoint.
            
            # Check if C is inside the triangle.
            # C is inside if it can be written as a positive combination of p1, p2, p3?
            # Or simpler: Check if the angles C-p1, C-p2, C-p3 are all less than the corresponding angles of the triangle?
            # A robust check: The barycentric coordinates of C on the plane defined by p1,p2,p3 must be positive.
            
            # Project C onto the plane? No, C is on the sphere.
            # Let's check if C is "between" the vertices.
            # A point C is inside the spherical triangle if the geodesic distance from C to each edge 
            # is such that C is on the same side as the opposite vertex.
            
            # Simpler check for small triangles:
            # If angle < pi/2, it's likely inside.
            # If angle >= pi/2, it's outside (obtuse triangle).
            
            if angle < np.pi / 2:
                # Likely inside. Verify by checking if C is closer to p1 than to any other point in pts?
                # This is expensive. 
                # For now, assume if angle < pi/2, it's the circumcenter of the Voronoi cell.
                if angle > max_angle:
                    max_angle = angle
            else:
                # Circumcenter is outside. The worst point is the midpoint of the longest edge.
                d12 = np.arccos(np.clip(p1 @ p2, -1.0, 1.0))
                d13 = np.arccos(np.clip(p1 @ p3, -1.0, 1.0))
                d23 = np.arccos(np.clip(p2 @ p3, -1.0, 1.0))
                
                max_edge_angle = max(d12, d13, d23)
                # The covering radius contribution is half the max edge angle?
                # No, the midpoint is the worst point on the edge.
                # The angular distance from the midpoint to the endpoints is max_edge_angle / 2.
                # But is it possible that a point off the edge is worse?
                # If the circumcenter is outside, the Voronoi cell vertex is not the circumcenter.
                # The Voronoi cell is bounded by the perpendicular bisectors.
                # If the triangle is obtuse, the Voronoi cell does not contain the circumcenter.
                # The worst point in the Voronoi cell for an obtuse triangle is the midpoint of the longest edge.
                
                candidate_angle = max_edge_angle / 2.0
                if candidate_angle > max_angle:
                    max_angle = candidate_angle
        
        return max_angle * 180.0 / np.pi
    except:
        # Fallback to sampling if hull fails
        i = np.arange(20000) + 0.5
        z = 1.0 - 2.0 * i / 20000
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        dots = samples @ pts.T
        dots = np.clip(dots, -1.0, 1.0)
        max_dots = np.max(dots, axis=1)
        angles = np.arccos(max_dots)
        return np.max(angles) * 180.0 / np.pi

def _repulsive_relaxation(pts, max_iter=5000, dt=0.01):
    """
    Optimize points using repulsive forces.
    Force F_ij = (p_i - p_j) / ||p_i - p_j||^2
    """
    pts = _normalize(pts.copy())
    for i in range(max_iter):
        # Calculate pairwise distances
        # diff[i,j] = pts[i] - pts[j]
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        
        # Avoid division by zero
        np.fill_diagonal(dist_sq, 1.0)
        
        # Force magnitude: 1 / dist_sq
        # Direction: diff
        # F_ij = diff / dist_sq^1.5 ? No, Coulomb is 1/r^2 force, so vector is diff/r * 1/r^2 = diff/r^3.
        # Let's use 1/r^2 potential, so force is 1/r^3 * r_vec = diff / r^2.
        # Wait, potential U = 1/r. Force = -grad(U) = 1/r^2 * r_hat = diff / r^3.
        # Let's use U = 1/r^2 for stronger repulsion at close range?
        # Standard Tammes uses repulsion ~ 1/r^2 or similar.
        
        # Let's use F = diff / dist_sq^1.5 (which is diff / r^3)
        # This corresponds to potential 1/r.
        
        inv_dist_cubed = 1.0 / (dist_sq * np.sqrt(dist_sq))
        # Zero out self-interaction
        np.fill_diagonal(inv_dist_cubed, 0.0)
        
        forces = diff * inv_dist_cubed[:, :, np.newaxis]
        
        # Sum forces for each point
        net_forces = np.sum(forces, axis=1)
        
        # Update positions
        pts = pts + dt * net_forces
        pts = _normalize(pts)
        
        # Cool down step size
        dt *= 0.999
        
    return pts

def _local_optimize(pts, max_time=10.0):
    """
    Local optimization using Nelder-Mead on the exact covering radius.
    """
    start_time = time.time()
    pts = _normalize(pts.copy())
    
    # Flatten points for optimizer
    x0 = pts.flatten()
    
    def objective(x):
        p = x.reshape(N, 3)
        p = _normalize(p)
        return _covering_radius_exact(p)
    
    # Nelder-Mead is slow for 150 dimensions.
    # Better: Coordinate descent or random perturbation on the worst points.
    
    best_pts = pts.copy()
    best_cr = _covering_radius_exact(best_pts)
    
    step_size = 0.05
    
    while time.time() - start_time < max_time:
        improved = False
        
        # Identify worst points? Or just random points?
        # Random points is more robust.
        for _ in range(10):
            i = np.random.randint(N)
            direction = np.random.randn(3)
            direction /= np.linalg.norm(direction)
            
            new_pts = best_pts.copy()
            new_pt = new_pts[i] + step_size * direction
            new_pts[i] = _normalize(new_pt)
            
            cr = _covering_radius_exact(new_pts)
            if cr < best_cr - 1e-10:
                best_cr = cr
                best_pts = new_pts
                improved = True
                # Reset step size if improved
                # step_size = 0.05 
                break
        
        if not improved:
            step_size *= 0.7
            if step_size < 1e-5:
                break
    
    return best_pts, best_cr

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_pts = None
    best_cr = np.inf
    
    # Multiple restarts
    for restart in range(5):
        if time.time() > deadline - 20.0:
            break
        
        # Initialization
        if restart == 0:
            # Fibonacci spiral
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i
            r = np.sqrt(1.0 - z * z)
            pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        else:
            # Random
            pts = np.random.randn(N, 3)
        
        pts = _normalize(pts)
        
        # Phase 1: Repulsive Relaxation
        # Time budget: 10s per restart
        relax_time = min(10.0, deadline - 20.0 - (time.time() - start_time))
        if relax_time > 1.0:
            # Estimate iterations
            # 5000 iters takes ~1-2s in NumPy for N=50?
            # Let's run a fixed number or time-based.
            # We'll run a fixed number of iterations, say 2000, which should be fast.
            pts = _repulsive_relaxation(pts, max_iter=2000, dt=0.05)
        
        # Phase 2: Local Polish
        polish_time = min(10.0, deadline - 10.0 - (time.time() - start_time))
        if polish_time > 1.0:
            pts, cr = _local_optimize(pts, max_time=polish_time)
        else:
            cr = _covering_radius_exact(pts)
        
        if cr < best_cr:
            best_cr = cr
            best_pts = pts.copy()
            
        # Check time
        if time.time() > deadline - 15.0:
            break
            
    # Final verification
    if best_pts is None:
        # Fallback
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
