import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = np.sqrt(3) / 4.0

# Precompute indices for all triples
_IDX_TRIPLES = None
def _get_indices():
    global _IDX_TRIPLES
    if _IDX_TRIPLES is None:
        n = N
        _IDX_TRIPLES = np.array([(i, j, k) for i in range(n) for j in range(i+1, n) for k in range(j+1, n)])
    return _IDX_TRIPLES

def _all_areas(points):
    """Compute all triangle areas, vectorized."""
    idx = _get_indices()
    pi = points[idx[:, 0]]
    pj = points[idx[:, 1]]
    pk = points[idx[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    return 0.5 * np.abs(cross)

def _min_triangle_area_vectorized(points):
    return np.min(_all_areas(points))

def _bary_to_cart(bary):
    x = bary[:, 0] * V1[0] + bary[:, 1] * V2[0]
    y = bary[:, 0] * V1[1] + bary[:, 1] * V2[1]
    return np.column_stack([x, y])

def _cart_to_bary(points):
    u = points[:, 0] - 0.5 * points[:, 1]
    v = 2.0 * points[:, 1] / np.sqrt(3)
    return np.column_stack([u, v])

def _clamp_bary(bary):
    bary = bary.copy()
    bary = np.clip(bary, 0.0, None)
    s = bary[:, 0] + bary[:, 1]
    mask = s > 1.0
    if np.any(mask):
        bary[mask] = bary[mask] / s[mask, None]
    return bary

def _objective_smooth(x, beta=50.0):
    """Smooth approximation of negative min area using softmin."""
    bary = x.reshape(-1, 2)
    bary = _clamp_bary(bary)
    points = _bary_to_cart(bary)
    areas = _all_areas(points)
    # Softmin: -log(sum(exp(-beta*a)))/beta approximates min for large beta
    # But for stability, use: min + (1/beta)*log(sum(exp(beta*(a-min))))
    a_min = np.min(areas)
    # Use a numerically stable softmin
    shifted = beta * (areas - a_min)
    # Clip to avoid overflow
    shifted = np.clip(shifted, 0, 50)
    log_sum = np.log(np.sum(np.exp(shifted)))
    softmin_val = a_min + log_sum / beta
    return -softmin_val

def _objective_bary(x):
    """Objective: negative min triangle area."""
    bary = x.reshape(-1, 2)
    bary = _clamp_bary(bary)
    points = _bary_to_cart(bary)
    return -_min_triangle_area_vectorized(points)

def _make_structured_starts():
    """Generate diverse structured starting points in barycentric coordinates."""
    starts = []
    
    # Start 1: Hexagonal lattice pattern
    bary1 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.0],
        [0.5, 0.5],
        [0.0, 0.5],
        [1/3, 1/3],
        [0.25, 0.0],
        [0.75, 0.0],
        [0.25, 0.25],
        [0.5, 0.75],
    ])
    starts.append(_clamp_bary(bary1.copy()))
    
    # Start 2: Concentric hexagons
    bary2 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.333, 0.0],
        [0.667, 0.0],
        [0.0, 0.333],
        [0.0, 0.667],
        [0.333, 0.333],
        [0.667, 0.333],
        [0.333, 0.667],
        [0.5, 0.5],
    ])
    starts.append(_clamp_bary(bary2.copy()))
    
    # Start 3: Near-optimal known configuration (symmetric)
    # Based on known best for 11 points in equilateral triangle
    bary3 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.25, 0.0],
        [0.75, 0.0],
        [0.0, 0.25],
        [0.0, 0.75],
        [0.25, 0.25],
        [0.5, 0.25],
        [0.25, 0.5],
        [0.5, 0.5],
    ])
    starts.append(_clamp_bary(bary3.copy()))
    
    # Start 4: Optimized hexagonal packing
    bary4 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.3, 0.0],
        [0.7, 0.0],
        [0.0, 0.3],
        [0.0, 0.7],
        [0.3, 0.3],
        [0.7, 0.3],
        [0.3, 0.7],
        [0.5, 0.5],
    ])
    starts.append(_clamp_bary(bary4.copy()))
    
    # Start 5: Slightly inset corners
    bary5 = np.array([
        [0.02, 0.02],
        [0.98, 0.02],
        [0.02, 0.98],
        [0.45, 0.02],
        [0.02, 0.45],
        [0.53, 0.45],
        [0.33, 0.33],
        [0.25, 0.55],
        [0.55, 0.20],
        [0.20, 0.25],
        [0.45, 0.25],
    ])
    starts.append(_clamp_bary(bary5.copy()))
    
    # Start 6: Random with structured bias
    rng = np.random.RandomState(456)
    bary6 = []
    bary6.append([0.0, 0.0])
    bary6.append([1.0, 0.0])
    bary6.append([0.0, 1.0])
    bary6.append([1/3, 1/3])
    for _ in range(7):
        u = rng.uniform(0.1, 0.9)
        v = rng.uniform(0.1, 0.9)
        if u + v < 0.85:
            bary6.append([u, v])
        else:
            bary6.append([u * 0.8, v * 0.8])
    starts.append(_clamp_bary(np.array(bary6[:N])))
    
    # Start 7: Symmetric grid with jitter
    rng2 = np.random.RandomState(123)
    bary7 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.25, 0.0],
        [0.75, 0.0],
        [0.0, 0.25],
        [0.0, 0.75],
        [0.25, 0.25],
        [0.5, 0.25],
        [0.25, 0.5],
        [0.5, 0.5],
    ])
    bary7 += rng2.normal(0, 0.02, size=bary7.shape)
    starts.append(_clamp_bary(bary7.copy()))
    
    return starts

def _local_optimize_nm(x0, maxiter=300, xatol=1e-8, fatol=1e-8):
    try:
        res = minimize(
            _objective_bary, x0, method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': xatol, 'fatol': fatol, 'adaptive': True}
        )
        return res.x
    except Exception:
        return x0

def _local_optimize_lbfgs(x0, maxiter=200, gtol=1e-9):
    try:
        res = minimize(
            _objective_bary, x0, method='L-BFGS-B',
            bounds=[(0.0, 1.0)] * (2 * N),
            options={'maxiter': maxiter, 'gtol': gtol}
        )
        return res.x
    except Exception:
        return x0

def _local_optimize_smooth(x0, beta=50.0, maxiter=300):
    """Optimize using smooth objective for gradient-based methods."""
    try:
        res = minimize(
            lambda x: _objective_smooth(x, beta), x0, method='L-BFGS-B',
            bounds=[(0.0, 1.0)] * (2 * N),
            options={'maxiter': maxiter, 'gtol': 1e-10}
        )
        return res.x
    except Exception:
        return x0

def _coordinate_descent(bary, n_iter=50, time_left=None, deadline=None):
    """Coordinate descent refinement on barycentric coordinates."""
    bary = bary.copy()
    best_score = _min_triangle_area_vectorized(_bary_to_cart(_clamp_bary(bary)))
    
    for it in range(n_iter):
        if time_left is not None and time_left() < 2:
            break
        improved = False
        for i in range(N):
            for coord in range(2):
                best_val = bary[i, coord]
                best_local_score = best_score
                # Try small perturbations
                for delta in [-0.01, -0.005, -0.001, 0.001, 0.005, 0.01]:
                    bary[i, coord] = best_val + delta
                    bary_clamped = _clamp_bary(bary)
                    score = _min_triangle_area_vectorized(_bary_to_cart(bary_clamped))
                    if score > best_local_score:
                        best_local_score = score
                        best_val = bary[i, coord]
                bary[i, coord] = best_val
                bary_clamped = _clamp_bary(bary)
                new_score = _min_triangle_area_vectorized(_bary_to_cart(bary_clamped))
                if new_score > best_score:
                    best_score = new_score
                    improved = True
        if not improved:
            break
    return bary, best_score

def _refine_with_multiple_methods(bary_start, time_left, deadline):
    """Refine a starting point using multiple optimization methods."""
    best_bary = bary_start.copy()
    best_score = _min_triangle_area_vectorized(_bary_to_cart(_clamp_bary(bary_start)))
    
    # Try Nelder-Mead
    if time_left() > 5:
        x_nm = _local_optimize_nm(bary_start.flatten(), maxiter=200)
        bary_nm = _clamp_bary(x_nm.reshape(-1, 2))
        score_nm = _min_triangle_area_vectorized(_bary_to_cart(bary_nm))
        if score_nm > best_score:
            best_score = score_nm
            best_bary = bary_nm.copy()
    
    # Try smooth L-BFGS-B
    if time_left() > 5:
        x_smooth = _local_optimize_smooth(best_bary.flatten(), beta=50.0, maxiter=200)
        bary_smooth = _clamp_bary(x_smooth.reshape(-1, 2))
        score_smooth = _min_triangle_area_vectorized(_bary_to_cart(bary_smooth))
        if score_smooth > best_score:
            best_score = score_smooth
            best_bary = bary_smooth.copy()
    
    # Try L-BFGS-B from the best so far
    if time_left() > 5:
        x_lbfgs = _local_optimize_lbfgs(best_bary.flatten(), maxiter=200)
        bary_lbfgs = _clamp_bary(x_lbfgs.reshape(-1, 2))
        score_lbfgs = _min_triangle_area_vectorized(_bary_to_cart(bary_lbfgs))
        if score_lbfgs > best_score:
            best_score = score_lbfgs
            best_bary = bary_lbfgs.copy()
    
    # Coordinate descent
    if time_left() > 3:
        bary_cd, score_cd = _coordinate_descent(best_bary, n_iter=30, time_left=time_left, deadline=deadline)
        if score_cd > best_score:
            best_score = score_cd
            best_bary = bary_cd.copy()
    
    # Fine-tune with Nelder-Mead from the best
    if time_left() > 3:
        x_ft = _local_optimize_nm(best_bary.flatten(), maxiter=300, xatol=1e-10, fatol=1e-10)
        bary_ft = _clamp_bary(x_ft.reshape(-1, 2))
        score_ft = _min_triangle_area_vectorized(_bary_to_cart(bary_ft))
        if score_ft > best_score:
            best_score = score_ft
            best_bary = bary_ft.copy()
    
    return best_bary, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    def time_left():
        return deadline - time.time()
    
    best_points = None
    best_score = -np.inf
    best_bary = None
    
    # Generate structured starting points
    starts = _make_structured_starts()
    
    # Evaluate each starting point
    for bary_start in starts:
        if time_left() < 5:
            break
        bary_clamped = _clamp_bary(bary_start)
        points = _bary_to_cart(bary_clamped)
        score = _min_triangle_area_vectorized(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
            best_bary = bary_clamped.copy()
    
    # Multi-start optimization
    rng = np.random.RandomState(42)
    max_restarts = 150
    restart = 0
    
    while restart < max_restarts and time_left() > 15.0:
        restart += 1
        
        # Choose starting point
        if restart % 7 == 0:
            # Random restart
            bary_start = []
            for _ in range(N):
                u = rng.uniform(0.05, 0.95)
                v = rng.uniform(0.05, 0.95)
                if u + v < 0.9:
                    bary_start.append([u, v])
                else:
                    bary_start.append([u * 0.8, v * 0.8])
            bary_start = _clamp_bary(np.array(bary_start[:N]))
        elif restart % 3 == 0:
            # Perturb best solution with moderate scale
            scale = 0.03 + 0.02 * (1.0 - restart / max_restarts)
            bary_pert = best_bary + rng.normal(0, scale, size=best_bary.shape)
            bary_start = _clamp_bary(bary_pert)
        else:
            # Perturb best solution with larger scale
            scale = 0.08 + 0.04 * (1.0 - restart / max_restarts)
            bary_pert = best_bary + rng.normal(0, scale, size=best_bary.shape)
            bary_start = _clamp_bary(bary_pert)
        
        # Refine using multiple methods
        bary_opt, score_opt = _refine_with_multiple_methods(bary_start, time_left, deadline)
        
        if score_opt > best_score:
            best_score = score_opt
            best_points = _bary_to_cart(bary_opt).copy()
            best_bary = bary_opt.copy()
    
    # Final fine-tuning from the best solution
    if time_left() > 8.0 and best_bary is not None:
        for ft in range(15):
            if time_left() < 4.0:
                break
            scale = 0.003
            bary_ft = best_bary + rng.normal(0, scale, size=best_bary.shape)
            bary_ft = _clamp_bary(bary_ft)
            x_ft = _local_optimize_nm(bary_ft.flatten(), maxiter=400, xatol=1e-11, fatol=1e-11)
            bary_opt = _clamp_bary(x_ft.reshape(-1, 2))
            points_opt = _bary_to_cart(bary_opt)
            score_opt = _min_triangle_area_vectorized(points_opt)
            
            if score_opt > best_score:
                best_score = score_opt
                best_points = points_opt.copy()
                best_bary = bary_opt.copy()
        
        # Final coordinate descent
        if time_left() > 3.0:
            bary_cd, score_cd = _coordinate_descent(best_bary, n_iter=50, time_left=time_left, deadline=deadline)
            if score_cd > best_score:
                best_score = score_cd
                best_points = _bary_to_cart(bary_cd).copy()
                best_bary = bary_cd.copy()
    
    # Fallback if no valid point was found
    if best_points is None:
        bary_fallback = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
            [0.5, 0.0], [0.5, 0.5], [0.0, 0.5],
            [1/3, 1/3], [0.25, 0.0], [0.75, 0.0],
            [0.25, 0.25], [0.5, 0.75],
        ])
        best_points = _bary_to_cart(_clamp_bary(bary_fallback))
    
    return best_points
