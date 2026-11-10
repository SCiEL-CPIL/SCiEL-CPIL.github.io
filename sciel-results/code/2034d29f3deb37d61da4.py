"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random
import numpy as np
from scipy.optimize import minimize, differential_evolution

def _log_score_vec(pts):
    """
    Fast vectorized computation of log(Delta / n^n) after scaling to diameter 2.
    pts: numpy array of shape (8, 2)
    Returns the log of the score (higher is better).
    """
    n = pts.shape[0]
    if n < 2:
        return -1e10
    # Compute all pairwise squared distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    # Mask diagonal (self-distances)
    np.fill_diagonal(d2, 1.0)  # temporary, will be excluded
    d = np.sqrt(d2)
    # Diameter = max pairwise distance
    D = np.max(d)
    if D < 1e-12:
        return -1e10
    s = 2.0 / D
    # Log of product of all ordered-pair distances after scaling
    d_scaled = d * s
    np.fill_diagonal(d_scaled, 1.0)
    d_scaled = np.clip(d_scaled, 1e-15, None)
    log_p = np.sum(np.log(d_scaled))
    log_score = log_p - n * math.log(n)
    return log_score

def _two_squares(r1, r2, theta1, theta2, dy):
    """
    Generate 8 points as two concentric, rotated regular squares.
    Square 1: radius r1, rotation theta1, centered at (0, dy)
    Square 2: radius r2, rotation theta2, centered at (0, 0)
    """
    # Square 1 vertices (rotated by theta1)
    a1 = np.array([0, math.pi/2, math.pi, 3*math.pi/2]) + theta1
    sq1 = np.column_stack([r1 * np.cos(a1), r1 * np.sin(a1)])
    sq1[:, 1] += dy  # vertical offset
    
    # Square 2 vertices (rotated by theta2)
    a2 = np.array([0, math.pi/2, math.pi, 3*math.pi/2]) + theta2
    sq2 = np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)])
    
    return np.vstack([sq1, sq2])

def _param_objective(params):
    """
    Objective for the 5-parameter two-squares model.
    params: [r1, r2, theta1, theta2, dy]
    """
    r1, r2, theta1, theta2, dy = params
    # Ensure positive radii
    r1 = max(abs(r1), 1e-10)
    r2 = max(abs(r2), 1e-10)
    pts = _two_squares(r1, r2, theta1, theta2, dy)
    ls = _log_score_vec(pts)
    return -ls

def _objective(pts_flat):
    """
    Objective for scipy minimization (we minimize negative log score).
    pts_flat: array of 16 floats (8 points x 2 coords)
    """
    pts = pts_flat.reshape(8, 2)
    ls = _log_score_vec(pts)
    return -ls

def _score_from_pts(pts_list):
    """Compute the actual score for a list of [x,y] points."""
    pts = np.array(pts_list, dtype=float)
    ls = _log_score_vec(pts)
    return math.exp(ls)

def construct():
    """
    Find 8 distinct points maximizing the product of distances at fixed diameter.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin for 150s limit

    def regular_octagon(radius=1.0):
        angles = 2 * math.pi * np.arange(8) / 8
        return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

    best_pts = regular_octagon()
    best_log_score = _log_score_vec(best_pts)

    # Phase 1: Global search over 5-parameter two-squares manifold using differential_evolution
    if time.time() < deadline - 60.0:
        try:
            # Bounds: r1, r2 in [0.1, 3.0], theta1, theta2 in [0, 2*pi], dy in [-2, 2]
            bounds = [(0.1, 3.0), (0.1, 3.0), (0, 2*math.pi), (0, 2*math.pi), (-2.0, 2.0)]
            de_result = differential_evolution(
                _param_objective, bounds,
                maxiter=100, popsize=15, tol=1e-8,
                seed=42, init='sobol'
            )
            if de_result.success:
                params_best = de_result.x
                r1, r2, theta1, theta2, dy = params_best
                r1 = max(abs(r1), 1e-10)
                r2 = max(abs(r2), 1e-10)
                pts_de = _two_squares(r1, r2, theta1, theta2, dy)
                ls_de = _log_score_vec(pts_de)
                if ls_de > best_log_score:
                    best_log_score = ls_de
                    best_pts = pts_de
        except Exception:
            pass

    # Phase 2: Local refinement from the two-squares result and regular octagon
    initializations = []
    
    # Regular octagon
    initializations.append(regular_octagon().copy())
    
    # Perturbed regular octagons
    random.seed(12345)
    np.random.seed(12345)
    for pert in [0.05, 0.1, 0.2, 0.3, 0.5]:
        for trial in range(2):
            base = regular_octagon()
            noise = np.random.randn(8, 2) * pert
            init = base + noise
            initializations.append(init)
    
    # Random points on unit circle
    for trial in range(3):
        angles = np.sort(np.random.uniform(0, 2 * math.pi, 8))
        init = np.column_stack([np.cos(angles), np.sin(angles)])
        initializations.append(init)
    
    # Add the two-squares result if it improved
    if best_log_score > _log_score_vec(regular_octagon()):
        initializations.append(best_pts.copy())
    
    # Random configurations
    for trial in range(3):
        init = np.random.randn(8, 2)
        D = np.max(np.sqrt(np.sum((init[:, np.newaxis, :] - init[np.newaxis, :])**2, axis=2)))
        if D > 1e-12:
            init = init * (2.0 / D)
        initializations.append(init)

    # Run Powell and Nelder-Mead from each initialization
    remaining_time = deadline - time.time()
    num_inits = len(initializations)
    time_budget_per_init = min(8.0, remaining_time / max(1, num_inits))
    
    for i, init in enumerate(initializations):
        if time.time() > deadline - 10.0:
            break
        
        # Check distinctness
        d2_mat = np.sum((init[:, np.newaxis, :] - init[np.newaxis, :])**2, axis=2)
        np.fill_diagonal(d2_mat, 1.0)
        if np.min(d2_mat) < 1e-8:
            continue
        
        init_flat = init.flatten()
        
        # Powell method
        try:
            res_p = minimize(
                _objective, init_flat,
                method='Powell',
                options={'maxiter': 200, 'xtol': 1e-10, 'ftol': 1e-10}
            )
            if res_p.success or res_p.fun < -best_log_score:
                pts_p = res_p.x.reshape(8, 2)
                ls_p = _log_score_vec(pts_p)
                if ls_p > best_log_score:
                    best_log_score = ls_p
                    best_pts = pts_p
        except Exception:
            pass
        
        if time.time() > deadline - 10.0:
            break
        
        # Nelder-Mead from the Powell result
        start_nm = res_p.x if 'res_p' in dir() and res_p.success else init_flat
        try:
            res_nm = minimize(
                _objective, start_nm,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            if res_nm.success or res_nm.fun < -best_log_score:
                pts_nm = res_nm.x.reshape(8, 2)
                ls_nm = _log_score_vec(pts_nm)
                if ls_nm > best_log_score:
                    best_log_score = ls_nm
                    best_pts = pts_nm
        except Exception:
            pass

    # Phase 3: High-precision polish on the best solution found
    if time.time() < deadline - 5.0:
        init_flat = best_pts.flatten()
        try:
            res_polish = minimize(
                _objective, init_flat,
                method='Powell',
                options={'maxiter': 500, 'xtol': 1e-12, 'ftol': 1e-12}
            )
            if res_polish.success:
                pts_polish = res_polish.x.reshape(8, 2)
                ls_polish = _log_score_vec(pts_polish)
                if ls_polish > best_log_score:
                    best_log_score = ls_polish
                    best_pts = pts_polish
        except Exception:
            pass

    # Phase 4: Final Nelder-Mead polish
    if time.time() < deadline - 3.0:
        init_flat = best_pts.flatten()
        try:
            res_nm = minimize(
                _objective, init_flat,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True}
            )
            if res_nm.success:
                pts_nm = res_nm.x.reshape(8, 2)
                ls_nm = _log_score_vec(pts_nm)
                if ls_nm > best_log_score:
                    best_log_score = ls_nm
                    best_pts = pts_nm
        except Exception:
            pass

    # Convert back to list of [x, y]
    best_list = [[float(p[0]), float(p[1])] for p in best_pts]
    
    # Ensure distinctness
    for a in range(8):
        for b in range(a + 1, 8):
            d = math.hypot(best_list[a][0] - best_list[b][0], best_list[a][1] - best_list[b][1])
            if d < 1e-6:
                best_list[b][0] += 1e-5 * (b - a + 1)
                best_list[b][1] += 1e-5 * (b - a + 2)
    
    # Ensure coordinate bounds
    for p in best_list:
        if abs(p[0]) > 1e6:
            p[0] = math.copysign(1e6 - 1, p[0])
        if abs(p[1]) > 1e6:
            p[1] = math.copysign(1e6 - 1, p[1])
    
    return best_list
# EVOLVE-BLOCK-END
