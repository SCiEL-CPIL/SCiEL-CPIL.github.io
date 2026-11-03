import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 130

def _fibonacci_init(seed=0):
    rng = np.random.RandomState(seed)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + rng.uniform(0, 2*np.pi)
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _soft_max_circumradii(radii, beta=50.0):
    """
    Smooth approximation of max(radii) using log-sum-exp.
    As beta -> inf, this approaches max(radii).
    """
    # Shift for numerical stability
    m = np.max(radii)
    if m < -1e15:
        return 0.0
    shifted = radii - m
    # Clip to avoid overflow in exp
    shifted = np.clip(shifted, -50.0, 50.0)
    log_sum_exp = m + np.log(np.sum(np.exp(beta * shifted))) / beta
    return log_sum_exp

def _circumcenter_robust(p1, p2, p3):
    """Compute circumcenter of triangle on unit sphere, robust to degeneracy."""
    edges = [
        (p2 - p1, p3 - p1),
        (p1 - p2, p3 - p2),
        (p1 - p3, p2 - p3)
    ]
    
    for v1, v2 in edges:
        n = np.cross(v1, v2)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-10:
            continue
        
        n = n / n_norm
        center = n.copy()
        
        dot1 = np.dot(center, p1)
        dot2 = np.dot(center, p2)
        dot3 = np.dot(center, p3)
        
        if dot1 > 0 and dot2 > 0 and dot3 > 0:
            return center
        else:
            return -center
    
    # Fallback: average of the three points, normalized
    center = (p1 + p2 + p3) / 3.0
    norm = np.linalg.norm(center)
    if norm < 1e-10:
        return p1.copy()
    return center / norm

def _compute_circumradii(pts, hull):
    """
    Compute angular circumradii for all triangular facets of the convex hull.
    Returns array of circumradii (in radians) and the circumcenters.
    """
    simplices = hull.simplices
    n_facets = len(simplices)
    radii = np.zeros(n_facets)
    centers = np.zeros((n_facets, 3))
    
    for k, tri in enumerate(simplices):
        v0 = pts[tri[0]]
        v1 = pts[tri[1]]
        v2 = pts[tri[2]]
        
        c = _circumcenter_robust(v0, v1, v2)
        centers[k] = c
        
        dot = np.clip(np.dot(c, v0), -1.0, 1.0)
        radii[k] = np.arccos(dot)
        
    return radii, centers

def _covering_radius(pts):
    """Compute the covering radius in degrees using convex hull."""
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 180.0, None, None
        
    radii, centers = _compute_circumradii(pts, hull)
    max_radius = np.max(radii)
    return np.degrees(max_radius), radii, centers

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    return pts / norms

def _optimize_fixed_hull(pts, simplices, max_iter=50):
    """
    Optimize point positions to minimize the smooth max circumradius
    given a fixed set of simplices (triangles).
    """
    def objective(x_flat):
        temp_pts = _normalize(x_flat.reshape(-1, 3))
        
        radii = np.zeros(len(simplices))
        for k, tri in enumerate(simplices):
            v0 = temp_pts[tri[0]]
            v1 = temp_pts[tri[1]]
            v2 = temp_pts[tri[2]]
            
            # Fast circumcenter calculation for objective
            # Normal vector to the plane
            n = np.cross(v1 - v0, v2 - v0)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-10:
                radii[k] = np.pi / 2
                continue
            c = n / n_norm
            
            # Ensure c is on the correct side (towards the origin or away? 
            # For convex hull of points on sphere, the circumcenter of the facet 
            # should be in the direction of the facet normal if the triangle is small.
            # Actually, the circumcenter on the sphere is the point on the great circle 
            # equidistant from v0, v1, v2.
            # The normal n is perpendicular to the plane. The circumcenter lies on the line 
            # passing through the origin and n? No, that's only for isosceles or specific cases.
            # The circumcenter of a spherical triangle is the intersection of the perpendicular bisectors.
            # For small triangles, it's close to the planar circumcenter projected.
            # A robust way:
            # The circumcenter C satisfies |C-v0|=|C-v1|=|C-v2| and C is on the sphere.
            # This is equivalent to C being the normalized cross product of (v0-v1) and (v0-v2) 
            # only if we are talking about the pole of the plane.
            # Wait, the "circumcenter" in the context of covering radius is the point on the sphere 
            # furthest from the vertices of the facet, which is the center of the empty cap.
            # This is indeed the normalized normal vector to the plane defined by the 3 points, 
            # provided the triangle is small enough (< 180 deg sides).
            # Let's verify: If v0, v1, v2 are close, the plane is close to the tangent plane.
            # The normal n points to the "outside". The point on the sphere in direction n is 
            # equidistant to v0, v1, v2?
            # Dot(n, v0) = Dot(n, v1) = Dot(n, v2) because n is perpendicular to v1-v0 and v2-v0.
            # Yes! The normal vector to the plane passing through the origin is equidistant 
            # (in dot product) to all three vertices.
            # So C = n / ||n|| is the circumcenter.
            
            dot = np.clip(np.dot(c, v0), -1.0, 1.0)
            radii[k] = np.arccos(dot)
            
        # Objective: Soft max of radii + Variance penalty
        # Variance penalty helps equalize radii
        var = np.var(radii)
        soft_max = _soft_max_circumradii(radii, beta=50.0)
        
        return soft_max + 0.1 * var

    x0 = pts.flatten()
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B', 
                          options={'maxiter': max_iter, 'gtol': 1e-6})
        return _normalize(result.x.reshape(-1, 3))
    except Exception:
        return pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_cr_overall = 180.0
    best_pts_overall = None
    
    n_restarts = 5
    time_per_restart = (deadline - start_time) / n_restarts
    
    for restart in range(n_restarts):
        restart_start = time.time()
        restart_deadline = restart_start + time_per_restart
        
        seed = restart * 100
        pts = _fibonacci_init(seed)
        
        best_cr = 180.0
        best_pts = pts.copy()
        
        # Initial Hull
        try:
            hull = ConvexHull(pts)
            radii, centers = _compute_circumradii(pts, hull)
            current_cr = np.degrees(np.max(radii))
        except Exception:
            continue
        
        if current_cr < best_cr:
            best_cr = current_cr
            best_pts = pts.copy()
            
        # Optimization Loop
        # We will iterate: 
        # 1. Optimize with fixed hull
        # 2. Recompute hull
        # 3. Check for improvement
        
        while time.time() < restart_deadline - 5.0:
            # 1. Optimize with current hull simplices
            try:
                new_pts = _optimize_fixed_hull(pts, hull.simplices, max_iter=30)
            except Exception:
                break
                
            # 2. Recompute hull with new points
            try:
                new_hull = ConvexHull(new_pts)
                new_radii, new_centers = _compute_circumradii(new_pts, new_hull)
                new_cr = np.degrees(np.max(new_radii))
            except Exception:
                # If hull fails, points might be collinear or coincident.
                # Add small noise and continue
                new_pts = _normalize(pts + np.random.normal(0, 1e-4, (N, 3)))
                continue
                
            # 3. Check improvement
            if new_cr < best_cr - 1e-6:
                best_cr = new_cr
                best_pts = new_pts.copy()
                pts = new_pts
                hull = new_hull
            else:
                # No significant improvement. 
                # If we are stuck, try a small random perturbation to escape local minima
                # Or break if we've been stuck for a while.
                # For now, just break the loop for this restart if no improvement in fixed-hull opt.
                # The fixed-hull opt should have converged. If it didn't improve the global max,
                # the topology might need to change, but L-BFGS on fixed topology can't do that.
                # So we break and rely on the next restart or a perturbation.
                
                # Let's try one perturbation step before giving up on this restart
                if time.time() < restart_deadline - 10.0:
                    perturbed_pts = _normalize(pts + np.random.normal(0, 0.01, (N, 3)))
                    try:
                        p_hull = ConvexHull(perturbed_pts)
                        p_radii, _ = _compute_circumradii(perturbed_pts, p_hull)
                        p_cr = np.degrees(np.max(p_radii))
                        if p_cr < best_cr:
                            best_cr = p_cr
                            best_pts = perturbed_pts.copy()
                            pts = perturbed_pts
                            hull = p_hull
                            continue # Try again with new topology
                    except Exception:
                        pass
                break
        
        if best_cr < best_cr_overall:
            best_cr_overall = best_cr
            best_pts_overall = best_pts.copy()
            
    # Final Polish
    if best_pts_overall is not None and time.time() < deadline - 5.0:
        try:
            final_hull = ConvexHull(best_pts_overall)
            final_pts = _optimize_fixed_hull(best_pts_overall, final_hull.simplices, max_iter=100)
            
            # Check if final polish improved
            f_hull = ConvexHull(final_pts)
            f_radii, _ = _compute_circumradii(final_pts, f_hull)
            f_cr = np.degrees(np.max(f_radii))
            
            if f_cr < best_cr_overall:
                best_cr_overall = f_cr
                best_pts_overall = final_pts
        except Exception:
            pass

    if best_pts_overall is None:
        return _fibonacci_init(0)
        
    return best_pts_overall
