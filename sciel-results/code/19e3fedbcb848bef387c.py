"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize


def _covering_radius(pts):
    """
    Compute the covering radius of points on the unit sphere.
    The covering radius is the maximum angular distance from any point on the sphere
    to its nearest chosen point. This is equivalent to the largest circumradius of the
    facets of the convex hull of the points.
    """
    try:
        hull = ConvexHull(pts)
        max_rad = 0.0
        worst_simplex = None
        for simplex in hull.simplices:
            tri = pts[simplex]
            a, b, c = tri[0], tri[1], tri[2]
            
            n = np.cross(b - a, c - a)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-12:
                continue
            n = n / n_norm
            
            centroid = (a + b + c) / 3.0
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm < 1e-12:
                continue
            centroid = centroid / centroid_norm
            
            if np.dot(n, centroid) < 0:
                n = -n
            
            cos_R = np.dot(n, a)
            cos_R = np.clip(cos_R, -1.0, 1.0)
            R = np.arccos(cos_R)
            
            if R > max_rad:
                max_rad = R
                worst_simplex = simplex
        
        return max_rad, worst_simplex
    except Exception:
        return np.pi, None


def _covering_radius_degrees(pts):
    rad, _ = _covering_radius(pts)
    return np.degrees(rad)


def _spherical_gradient_step(pts, worst_simplex, step_size):
    """
    Perform a Riemannian gradient step to minimize the worst facet's circumradius.
    The gradient of the circumradius R with respect to vertex a is:
    dR/da = -sin(R) * (n - (n·a)a) / sin(R) ... actually let's derive properly.
    
    cos(R) = n · a, where n is the unit normal to the plane of the triangle.
    R = arccos(n · a)
    dR/da = -1/sqrt(1-(n·a)^2) * n = -n / sin(R)
    
    But we need the gradient on the sphere. The Riemannian gradient is the projection
    of the Euclidean gradient onto the tangent space.
    
    For the circumradius to decrease, we want to move a in the direction that increases n·a.
    Since n is fixed (for small moves), we want to move a towards n.
    The direction on the sphere towards n is: n - (n·a)a, normalized.
    
    Similarly for b and c.
    """
    if worst_simplex is None:
        return pts.copy()
    
    new_pts = pts.copy()
    a_idx, b_idx, c_idx = worst_simplex[0], worst_simplex[1], worst_simplex[2]
    
    a = pts[a_idx]
    b = pts[b_idx]
    c = pts[c_idx]
    
    n = np.cross(b - a, c - a)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-12:
        return new_pts
    n = n / n_norm
    
    centroid = (a + b + c) / 3.0
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm < 1e-12:
        return new_pts
    centroid = centroid / centroid_norm
    
    if np.dot(n, centroid) < 0:
        n = -n
    
    cos_R = np.dot(n, a)
    cos_R = np.clip(cos_R, -1.0, 1.0)
    sin_R = np.sqrt(1.0 - cos_R * cos_R)
    if sin_R < 1e-12:
        return new_pts
    
    # Riemannian gradient direction: move each vertex towards the circumcenter n
    # to increase n·a, n·b, n·c (which decreases R)
    for idx, p in [(a_idx, a), (b_idx, b), (c_idx, c)]:
        # Direction towards n on the sphere
        proj = np.dot(n, p)
        grad_dir = n - proj * p
        grad_norm = np.linalg.norm(grad_dir)
        if grad_norm < 1e-12:
            continue
        grad_dir = grad_dir / grad_norm
        
        # Step in the direction of the gradient
        new_p = p + step_size * grad_dir
        new_p = new_p / np.linalg.norm(new_p)
        new_pts[idx] = new_p
    
    return new_pts


def _refine_with_gradient_flow(pts, max_time=30.0, initial_step=0.05):
    """
    Refine the configuration using Riemannian gradient flow on the worst facet.
    """
    start_time = time.time()
    current_pts = pts.copy()
    best_pts = pts.copy()
    best_rad, _ = _covering_radius(pts)
    
    step_size = initial_step
    no_improve_count = 0
    
    while time.time() - start_time < max_time:
        rad, worst_simplex = _covering_radius(current_pts)
        
        if rad < best_rad - 1e-10:
            best_rad = rad
            best_pts = current_pts.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if worst_simplex is None:
            break
        
        # Try gradient step
        candidate = _spherical_gradient_step(current_pts, worst_simplex, step_size)
        cand_rad, _ = _covering_radius(candidate)
        
        if cand_rad < rad - 1e-12:
            current_pts = candidate
            # Adaptive step size: increase if improving
            step_size = min(step_size * 1.1, 0.2)
        else:
            # Decrease step size
            step_size *= 0.8
            if step_size < 1e-4:
                break
            
            # If stuck, try a small random perturbation to escape
            if no_improve_count > 10:
                idx = np.random.randint(N)
                noise = np.random.randn(3) * 0.05
                candidate = current_pts.copy()
                candidate[idx] = candidate[idx] + noise
                candidate[idx] = candidate[idx] / np.linalg.norm(candidate[idx])
                cand_rad, _ = _covering_radius(candidate)
                if cand_rad < rad:
                    current_pts = candidate
                    no_improve_count = 0
                else:
                    break
    
    return best_pts


def _local_optimization(pts, max_time=20.0):
    """
    Local optimization by moving each point to minimize the covering radius.
    """
    start_time = time.time()
    current_pts = pts.copy()
    best_pts = pts.copy()
    best_rad, _ = _covering_radius(pts)
    
    for iteration in range(50):
        if time.time() - start_time > max_time:
            break
        
        improved = False
        for idx in range(N):
            if time.time() - start_time > max_time:
                break
            
            orig_pt = current_pts[idx].copy()
            pts_fixed = np.delete(current_pts, idx, axis=0)
            
            best_local = orig_pt
            best_local_rad, _ = _covering_radius(current_pts)
            
            # Try several random perturbations
            for trial in range(15):
                if time.time() - start_time > max_time:
                    break
                
                noise = np.random.randn(3) * 0.1
                candidate = orig_pt + noise
                candidate = candidate / np.linalg.norm(candidate)
                
                current_pts[idx] = candidate
                rad, _ = _covering_radius(current_pts)
                
                if rad < best_local_rad:
                    best_local_rad = rad
                    best_local = candidate.copy()
            
            current_pts[idx] = best_local
            
            if best_local_rad < best_rad - 1e-10:
                best_rad = best_local_rad
                best_pts = current_pts.copy()
                improved = True
        
        if not improved:
            break
    
    return best_pts


def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 140.0  # 10s safety margin before 150s limit
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    
    best_pts = pts.copy()
    best_rad, _ = _covering_radius(pts)
    
    # Phase 1: Gradient flow refinement
    if time.time() < deadline - 40.0:
        refined = _refine_with_gradient_flow(pts, max_time=35.0)
        rad, _ = _covering_radius(refined)
        if rad < best_rad:
            best_rad = rad
            best_pts = refined.copy()
    
    # Phase 2: Local optimization
    if time.time() < deadline - 20.0:
        locally_opt = _local_optimization(best_pts, max_time=18.0)
        rad, _ = _covering_radius(locally_opt)
        if rad < best_rad:
            best_rad = rad
            best_pts = locally_opt.copy()
    
    # Phase 3: Final gradient flow pass
    if time.time() < deadline - 5.0:
        final = _refine_with_gradient_flow(best_pts, max_time=4.0, initial_step=0.02)
        rad, _ = _covering_radius(final)
        if rad < best_rad:
            best_rad = rad
            best_pts = final.copy()
    
    # Ensure all points are on the unit sphere
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts = best_pts / norms
    
    return best_pts
# EVOLVE-BLOCK-END
