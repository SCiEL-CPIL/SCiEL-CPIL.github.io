import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
import itertools

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])

# Precompute all triplets for N=11 points
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])

DEADLINE = 145  # Safety margin from 150s limit

def _cart_to_bary(pts):
    """
    Convert Cartesian to barycentric (u, v) relative to V0, V1, V2.
    pts: (N, 2) array of [x, y]
    Returns: (N, 2) array of [u, v] where P = u*V0 + v*V1 + (1-u-v)*V2
    Note: This specific barycentric mapping is convenient for constraints.
    u = (S3H - y) / S3H
    v = 1 - 2x
    Check: 
    V0(0,0) -> u=1, v=1 -> (1,0,0) in (u,v,w)?? 
    Let's stick to standard barycentric (b0, b1, b2) summing to 1.
    b0 = 1 - x - y/S3H
    b1 = x
    b2 = y/S3H
    """
    x = pts[:, 0]
    y = pts[:, 1]
    b0 = 1.0 - x - y / S3H
    b1 = x
    b2 = y / S3H
    return np.column_stack([b0, b1, b2])

def _bary_to_cart(bary):
    """
    Convert barycentric (b0, b1, b2) to Cartesian.
    bary: (N, 3)
    """
    x = bary[:, 1] * V1[0] + bary[:, 2] * V2[0] # b0*V0 is 0
    y = bary[:, 1] * V1[1] + bary[:, 2] * V2[1]
    return np.column_stack([x, y])

def _project_to_triangle(pts):
    """
    Project Cartesian points to be inside the triangle.
    If a point is outside, we clip its barycentric coordinates to be >= 0 and sum to 1.
    """
    bary = _cart_to_bary(pts)
    # Clip to non-negative
    bary = np.clip(bary, 0.0, None)
    # Normalize to sum to 1
    sums = np.sum(bary, axis=1, keepdims=True)
    # Avoid division by zero (should not happen if we clip 0 to epsilon)
    sums = np.where(sums == 0, 1.0, sums)
    bary = bary / sums
    return _bary_to_cart(bary)

def _min_triangle_area(pts):
    """
    Compute the minimum area of any triangle formed by three points.
    pts: (N, 2)
    """
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    return np.min(areas)

def _objective(x):
    """
    Objective function for optimization.
    x: flat array of shape (2*N,)
    Returns: negative minimum area (to be minimized).
    """
    pts = x.reshape(N, 2)
    # Project to valid triangle
    pts = _project_to_triangle(pts)
    area = _min_triangle_area(pts)
    return -area

def _soft_objective(x, beta=50.0):
    """
    Soft objective using log-sum-exp for smoother gradients.
    """
    pts = x.reshape(N, 2)
    pts = _project_to_triangle(pts)
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    
    # Soft min: -log(sum(exp(-beta * areas))) / beta
    # We want to maximize min(areas), so minimize -min(areas).
    # -min(a) = - ( -log(sum(exp(-beta*a)))/beta ) = log(sum(exp(-beta*a)))/beta
    # Let's just return -soft_min_area
    # soft_min_area = -log(sum(exp(-beta * areas))) / beta
    # We want to minimize -soft_min_area = log(sum(exp(-beta * areas))) / beta
    
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    shifted = np.clip(shifted, -700, 700)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = max_a - log_sum / beta
    
    return -soft_min

def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    best_area = 0.0
    best_pts = None
    
    # Bounds for Cartesian coordinates
    # x in [0, 1], y in [0, S3H]
    # But we project anyway, so we can use a slightly larger box to allow DE to explore
    # The projection will handle the constraints.
    bounds = [(0.0, 1.0), (0.0, S3H)] * N
    
    def _elapsed_ok(margin):
        return (time.time() - start_time) < (DEADLINE - margin)

    def _update_best(pts):
        nonlocal best_area, best_pts
        pts = _project_to_triangle(pts)
        area = _min_triangle_area(pts)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()
            return True
        return False

    # Phase 1: Differential Evolution (Global Search)
    # DE is stochastic, so we run it with a decent population size.
    # maxiter is limited by time.
    # We use the objective that returns -min_area.
    
    # We need to pass a callback to check time.
    # scipy's differential_evolution doesn't have a direct callback for stopping mid-iteration 
    # easily in older versions, but we can wrap the objective or just rely on maxiter being conservative.
    # Better: Run DE with a moderate maxiter, then polish.
    
    if _elapsed_ok(60):
        try:
            # Use a relatively large population for better exploration
            popsize = 20
            # maxiter: we have ~60s. Each evaluation is fast (~1ms?). 
            # 165 triplets * 11 points. Vectorized numpy is very fast.
            # Let's estimate 10,000 evaluations in 10s. So 60s -> 60,000 evals.
            # Pop 20 * 6000 gens = 120,000 evals. Might be too much.
            # Let's try maxiter=1000, popsize=20 -> 20,000 evals.
            
            res = differential_evolution(
                _objective, 
                bounds, 
                seed=42,
                maxiter=1000,
                popsize=20,
                tol=1e-12,
                polish=True,  # DE does a local polish at the end
                init='latinhypercube',
                mutation=(0.5, 1.0),
                recombination=0.7
            )
            if res.success or res.fun < 0:
                pts = res.x.reshape(N, 2)
                _update_best(pts)
        except Exception as e:
            pass

    # Phase 2: Local Polish with Nelder-Mead on the best point found
    if best_pts is not None and _elapsed_ok(30):
        try:
            # Nelder-Mead on the raw Cartesian coordinates
            # We define a local objective that projects
            def nm_obj(x):
                pts = x.reshape(N, 2)
                pts = _project_to_triangle(pts)
                return -_min_triangle_area(pts)
            
            res_nm = minimize(
                nm_obj, 
                best_pts.flatten(), 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if res_nm.fun < 0:
                pts = res_nm.x.reshape(N, 2)
                _update_best(pts)
        except Exception:
            pass

    # Phase 3: Multi-start Nelder-Mead from random points if time remains
    # This helps if DE got stuck or if the global optimum is in a different basin.
    if _elapsed_ok(20):
        n_restarts = 10
        for i in range(n_restarts):
            if not _elapsed_ok(2):
                break
            
            # Random initial points
            pts_init = rng.random((N, 2)) * np.array([1.0, S3H])
            pts_init = _project_to_triangle(pts_init)
            
            def nm_obj(x):
                pts = x.reshape(N, 2)
                pts = _project_to_triangle(pts)
                return -_min_triangle_area(pts)
            
            try:
                res_nm = minimize(
                    nm_obj, 
                    pts_init.flatten(), 
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                if res_nm.fun < 0:
                    pts = res_nm.x.reshape(N, 2)
                    _update_best(pts)
            except Exception:
                pass

    # Phase 4: Final fine polish if time allows
    if best_pts is not None and _elapsed_ok(10):
        try:
            def nm_obj(x):
                pts = x.reshape(N, 2)
                pts = _project_to_triangle(pts)
                return -_min_triangle_area(pts)
            
            res_nm = minimize(
                nm_obj, 
                best_pts.flatten(), 
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            if res_nm.fun < 0:
                pts = res_nm.x.reshape(N, 2)
                _update_best(pts)
        except Exception:
            pass

    if best_pts is None:
        # Fallback: simple grid or symmetric
        # Use the center and vertices
        pts = np.zeros((N, 2))
        pts[0] = V0
        pts[1] = V1
        pts[2] = V2
        pts[3] = (V0+V1)/2
        pts[4] = (V1+V2)/2
        pts[5] = (V2+V0)/2
        pts[6] = (V0+V1+V2)/3
        # Fill rest randomly
        pts[7:] = rng.random((4, 2)) * np.array([1.0, S3H])
        best_pts = _project_to_triangle(pts)
        
    return best_pts
