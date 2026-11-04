import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

# Precompute regular 13-gon vertices on unit circle
_REG_ANGLES = np.linspace(0, 2*np.pi, N, endpoint=False)
_REG_COS = np.cos(_REG_ANGLES)
_REG_SIN = np.sin(_REG_ANGLES)
_REG_PTS = np.column_stack([_REG_COS, _REG_SIN])

# Precompute all triangle index triplets
_TRIPLETS = np.array(list(combinations(range(N), 3)))
_N_TRIPLES = _TRIPLETS.shape[0]
_I, _J, _K = _TRIPLETS.T

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by 3 points."""
    a = pts[_I]
    b = pts[_J]
    c = pts[_K]
    areas = 0.5 * np.abs(a[:, 0]*(b[:, 1]-c[:, 1]) + b[:, 0]*(c[:, 1]-a[:, 1]) + c[:, 0]*(a[:, 1]-b[:, 1]))
    return float(np.min(areas))

def _hull_area(pts):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 1e-10

def _score(pts):
    """Calculate the objective score: min_triangle_area / hull_area."""
    min_area = _min_triangle_area(pts)
    h_area = _hull_area(pts)
    if h_area < 1e-10:
        return 0.0
    return min_area / h_area

def _all_triangle_areas(pts):
    """Compute areas of all C(N,3) triangles as a 1D array (vectorized)."""
    a = pts[_I]
    b = pts[_J]
    c = pts[_K]
    areas = 0.5 * np.abs(a[:, 0]*(b[:, 1]-c[:, 1]) + b[:, 0]*(c[:, 1]-a[:, 1]) + c[:, 0]*(a[:, 1]-b[:, 1]))
    return areas

def _soft_min_score(pts, beta=50.0):
    """
    Smooth approximation of min_triangle_area / hull_area using log-sum-exp.
    soft_min(x) = -(1/beta) * log(sum(exp(-beta * x_i)))
    This is differentiable and approximates min as beta -> inf.
    """
    areas = _all_triangle_areas(pts)
    areas = np.maximum(areas, 1e-15)
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
    soft_min_area = -log_sum_exp / beta
    
    h_area = _hull_area(pts)
    if h_area < 1e-10:
        return 0.0
    return soft_min_area / h_area

def _neg_soft_min_score(x, beta=50.0):
    """Negative soft-min score for minimization."""
    pts = x.reshape(N, 2)
    return -_soft_min_score(pts, beta)

def _regular_polygon(n, radius=1.0):
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _apply_elliptic(params):
    """Apply symmetric 2x2 matrix M = [[a, b], [b, c]] to regular polygon."""
    a, b, c = params
    x = _REG_COS
    y = _REG_SIN
    new_x = a * x + b * y
    new_y = b * x + c * y
    return np.column_stack([new_x, new_y])

def _elliptic_soft_min(params, beta=50.0):
    """Soft-min score for elliptically distorted regular polygon."""
    pts = _apply_elliptic(params)
    return -_soft_min_score(pts, beta)

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin
    
    best_score = 0.0
    best_points = None
    
    def update_best(pts, score):
        nonlocal best_score, best_points
        if score > best_score:
            best_score = score
            best_points = pts.copy()
    
    # Phase 1: Optimize in the 3D elliptic parameter space
    # M = [[a, b], [b, c]] applied to regular 13-gon
    # For PSD: a > 0, c > 0, ac > b^2
    # We optimize over (a, b, c) with bounds to ensure reasonable shapes
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial parameter sets
    param_inits = []
    
    # Identity (circle)
    param_inits.append(np.array([1.0, 0.0, 1.0]))
    
    # Various elliptic distortions
    for a in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4]:
        for c in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4]:
            for b in [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]:
                if a * c > b * b * 1.01:  # Ensure PSD with margin
                    param_inits.append(np.array([a, b, c]))
    
    # Random PSD matrices
    for _ in range(30):
        a = rng.uniform(0.6, 1.5)
        c = rng.uniform(0.6, 1.5)
        max_b = np.sqrt(a * c) * 0.8
        b = rng.uniform(-max_b, max_b)
        param_inits.append(np.array([a, b, c]))
    
    # Evaluate initial candidates
    for params in param_inits:
        if time.time() > deadline:
            break
        pts = _apply_elliptic(params)
        s = _score(pts)
        update_best(pts, s)
    
    # Score and sort initial params
    scored_params = []
    for params in param_inits:
        pts = _apply_elliptic(params)
        s = _score(pts)
        scored_params.append((s, params))
    scored_params.sort(key=lambda x: -x[0])
    
    top_params = [p for _, p in scored_params[:min(15, len(scored_params))]]
    
    # Progressive beta optimization in 3D param space
    betas = [20.0, 50.0, 100.0, 200.0, 400.0]
    
    for beta in betas:
        if time.time() > deadline:
            break
        
        for init_params in top_params:
            if time.time() > deadline:
                break
            
            try:
                res = minimize(
                    _elliptic_soft_min,
                    init_params,
                    args=(beta,),
                    method='L-BFGS-B',
                    bounds=[(0.3, 2.0), (-1.0, 1.0), (0.3, 2.0)],
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
                )
                
                res_params = res.x
                # Check PSD condition
                a, b, c = res_params
                if a * c > b * b:
                    res_pts = _apply_elliptic(res_params)
                    res_score = _score(res_pts)
                    update_best(res_pts, res_score)
                    # Add to top params for next beta iteration
                    top_params.append(res_params.copy())
            except Exception:
                pass
        
        # Keep top 10 params for next iteration
        if best_points is not None:
            # Re-score and keep top
            scored_params2 = []
            for p in top_params:
                pts = _apply_elliptic(p)
                s = _score(pts)
                scored_params2.append((s, p))
            scored_params2.sort(key=lambda x: -x[0])
            top_params = [p for _, p in scored_params2[:10]]
    
    # Phase 2: Full 26D optimization from best elliptic result
    if best_points is not None and time.time() < deadline - 20:
        # Use the best elliptic result as starting point for full optimization
        full_init = best_points.flatten()
        
        for beta in [100.0, 200.0, 500.0]:
            if time.time() > deadline:
                break
            try:
                res = minimize(
                    _neg_soft_min_score,
                    full_init,
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
                )
                res_pts = res.x.reshape(N, 2)
                res_score = _score(res_pts)
                update_best(res_pts, res_score)
                full_init = res.x.copy()
            except Exception:
                pass
    
    # Phase 3: Final Nelder-Mead polish on exact score
    if best_points is not None and time.time() < deadline - 5:
        try:
            res = minimize(
                lambda x: -_score(x.reshape(N, 2)),
                best_points.flatten(),
                method='Nelder-Mead',
                options={'xatol': 1e-7, 'fatol': 1e-10, 'maxiter': 5000}
            )
            res_pts = res.x.reshape(N, 2)
            res_score = _score(res_pts)
            update_best(res_pts, res_score)
        except Exception:
            pass
    
    # Fallback
    if best_points is None:
        best_points = _regular_polygon(N, 1.0)
        best_score = _score(best_points)
    
    return best_points

# EVOLVE-BLOCK-END
