import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import least_squares

N = 130
_EPS = 1e-12

def _initial_fib():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_init(rng):
    """Generate random initial points."""
    pts = rng.standard_normal((N, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _get_hull_data(pts):
    """
    Compute convex hull and facet properties (normals, radii, vertices).
    Returns:
        hull: ConvexHull object
        faces: array of vertex indices for each facet
        normals: unit outward normals for each facet
        radii: angular circumradius for each facet
    """
    try:
        hull = ConvexHull(pts)
        faces = hull.simplices
        M = len(faces)
        
        A = pts[faces[:, 0]]
        B = pts[faces[:, 1]]
        C = pts[faces[:, 2]]
        
        AB = B - A
        AC = C - A
        cross = np.cross(AB, AC)
        area = 0.5 * np.linalg.norm(cross, axis=1)
        
        valid = area > _EPS
        if not np.any(valid):
            return None, None, None, None

        normals = cross[valid]
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        
        # Ensure normals point outward
        centroids = (A[valid] + B[valid] + C[valid]) / 3.0
        signs = np.sum(normals * centroids, axis=1, keepdims=True)
        normals = normals * np.sign(signs)
        normals = normals / (norms + _EPS)
        
        cos_theta = np.sum(normals * A[valid], axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        radii = np.arccos(cos_theta)
        
        valid_faces = faces[valid]
        
        return hull, valid_faces, normals, radii
    except Exception:
        return None, None, None, None

def _compute_residuals(x, target_radius, pts_cache=None):
    """
    Compute residuals for least_squares.
    Residuals are (current_radius - target_radius) for each facet,
    weighted by how much they exceed the target (or just all facets if we want uniform pressure).
    Actually, to make it a proper least squares problem that drives the max down,
    we can define residuals for ALL facets as (r_i - target).
    Minimizing sum of squares will try to make all r_i close to target.
    If target is smaller than current max, the large facets will have large positive residuals,
    dominating the objective and pulling them down.
    """
    pts = x.reshape(N, 3)
    # Normalize to sphere
    pts /= (np.linalg.norm(pts, axis=1, keepdims=True) + _EPS)
    
    _, faces, normals, radii = _get_hull_data(pts)
    if faces is None:
        return np.ones(N) * 10.0 # Large penalty if hull fails
        
    # Residuals: r_i - target
    # We only care about facets that are "active" (close to max or exceeding target)
    # But for least squares, we can just use all facets.
    # To focus on the worst ones, we could weight by max(0, r_i - target + margin)
    # Let's use a soft threshold: if r_i > target, residual is (r_i - target).
    # If r_i < target, residual is 0 (or small).
    # This makes it a "minimize the violation" problem.
    
    # Actually, standard least squares minimizes sum of squares.
    # If we set residual = max(0, r_i - target), the function is non-smooth at r_i=target.
    # least_squares uses finite differences or Jacobian.
    # Let's use a smooth approximation: log-sum-exp or just a softplus-like function?
    # Or simply: residual = r_i - target for all i.
    # This will try to make ALL radii equal to target.
    # This is a strong constraint that might be too rigid if target is too low.
    # But if we iteratively lower target, it should work.
    
    # Let's try: residual_i = r_i - target
    residuals = radii - target_radius
    
    # To prevent the optimizer from just making all radii huge (which is impossible on sphere)
    # or degenerating, the normalization handles the sphere constraint.
    # The geometry of the hull naturally constrains the radii.
    
    return residuals

def _jacobian(x, target_radius):
    """
    Numerical Jacobian is used by least_squares if not provided.
    We can provide an analytical one if we derive it, but numerical is safer for first pass.
    """
    pass # Let least_squares handle it via finite differences or we implement if needed.
    # Given the complexity of hull changes, finite difference is acceptable for N=130 (390 vars).
    # 390 * 2 * cost_of_eval. cost_of_eval is ~1ms? 390*2*1ms = 0.8s per step.
    # Too slow for many steps.
    # We need to be careful.
    # Maybe use '2-point' finite difference with small step.
    
    # Alternative: Use the fact that the gradient of the radius w.r.t vertices is known.
    # d r / d p_k = - (p_k - pole) / (r * sin(r)) ? Something like that.
    # Let's stick to numerical but optimize the eval speed.
    # If numerical jac is too slow, we might need to fall back to BFGS on the max radius.
    # But the prompt asks for least_squares strategy.
    
    # Let's try to implement a semi-analytical Jacobian or just accept the cost.
    # 390 dimensions. 390 evaluations per Jacobian.
    # If one eval is 2ms, that's 0.8s.
    # 10 iterations = 8s. 10 restarts = 80s. Fits in 150s.

def _solve_with_least_squares(pts, target_radius, max_iter=50, deadline=None):
    """
    Run least_squares to minimize residuals (r_i - target).
    """
    x0 = pts.flatten()
    
    # Use 'trf' method which is robust
    try:
        res = least_squares(
            _compute_residuals,
            x0,
            args=(target_radius,),
            method='trf',
            max_nfev=200, # Limit function evaluations to control time
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8
        )
        # Normalize result
        pts_opt = res.x.reshape(N, 3)
        pts_opt /= (np.linalg.norm(pts_opt, axis=1, keepdims=True) + _EPS)
        return pts_opt
    except Exception:
        return pts

def _multi_restart_search(deadline):
    """
    Run multiple restarts.
    Strategy:
    1. Start with a decent initial config (Fibonacci).
    2. Compute current max radius R_max.
    3. Set target_radius = R_max * 0.95.
    4. Run least_squares.
    5. If success (new R_max < old R_max), update best.
    6. Lower target_radius further and repeat.
    """
    rng = np.random.default_rng(42)
    best_pts = _initial_fib()
    _, _, _, radii = _get_hull_data(best_pts)
    best_r = np.max(radii) if radii is not None else np.pi
    
    # We will try to improve the best found
    current_pts = best_pts.copy()
    
    # Time budget per major iteration
    # Total 145s.
    # Let's do a loop that tries to tighten the radius.
    
    target_radius = best_r * 0.98 # Start close to current
    min_target = np.deg2rad(11.0) # Known best is ~11.31. Try to go below 11.0?
    
    last_improvement_time = time.time()
    
    while time.time() < deadline - 10.0:
        if target_radius < min_target:
            break
            
        # Run least squares
        t_start = time.time()
        new_pts = _solve_with_least_squares(
            current_pts, 
            target_radius, 
            max_iter=50, 
            deadline=deadline
        )
        
        # Evaluate new points
        _, _, _, new_radii = _get_hull_data(new_pts)
        if new_radii is not None:
            new_r = np.max(new_radii)
        else:
            new_r = np.pi
            
        if new_r < best_r - 1e-5:
            best_r = new_r
            best_pts = new_pts.copy()
            current_pts = new_pts.copy()
            last_improvement_time = time.time()
            
            # Tighten target
            # If we achieved target, we can try to go lower
            if new_r <= target_radius + 1e-4:
                # We hit the target, try to lower it
                target_radius *= 0.95
            else:
                # We didn't hit target, but improved. Try a slightly lower target?
                # Or retry with same target?
                # Let's lower it slightly to push further
                target_radius = new_r * 0.95
                
        else:
            # No improvement or stagnation
            # If we haven't improved in a while, try a random restart
            if time.time() - last_improvement_time > 20.0:
                # Random restart
                current_pts = _random_init(rng)
                # Or perturb best
                current_pts = best_pts + rng.normal(0, 0.01, (N, 3))
                current_pts /= np.linalg.norm(current_pts, axis=1, keepdims=True)
                
                _, _, _, curr_radii = _get_hull_data(current_pts)
                if curr_radii is not None:
                    curr_r = np.max(curr_radii)
                    if curr_r < best_r:
                        best_r = curr_r
                        best_pts = current_pts.copy()
                    target_radius = curr_r * 0.95
                else:
                    target_radius = best_r * 0.95
                
                last_improvement_time = time.time()
            else:
                # Try a different target? Maybe the optimizer got stuck.
                # Try a slightly higher target to escape local min?
                # Or just break and try another restart.
                target_radius *= 0.99
                if target_radius < min_target:
                    break

    # Final polish: try one more time with a very tight target on the best points
    if time.time() < deadline - 5.0:
        final_target = best_r * 0.95
        if final_target > min_target:
            final_pts = _solve_with_least_squares(
                best_pts, 
                final_target, 
                max_iter=100, 
                deadline=deadline
            )
            _, _, _, final_radii = _get_hull_data(final_pts)
            if final_radii is not None:
                final_r = np.max(final_radii)
                if final_r < best_r:
                    best_r = final_r
                    best_pts = final_pts

    return best_pts

def construct():
    deadline = time.time() + 145.0
    best_pts = _multi_restart_search(deadline)
    
    # Final normalization and validity check
    best_pts = np.array(best_pts, dtype=float)
    best_pts /= (np.linalg.norm(best_pts, axis=1, keepdims=True) + _EPS)
    
    if not np.all(np.isfinite(best_pts)) or best_pts.shape != (N, 3):
        return _initial_fib()
        
    return best_pts
