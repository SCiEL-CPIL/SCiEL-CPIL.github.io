"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random
import numpy as np
from scipy.optimize import minimize

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
    # sum_{i != j} log(d_ij * s) = sum_{i != j} log(d_ij) + n*(n-1)*log(s)
    # Exclude diagonal by setting those entries to 1 (log=0)
    d_scaled = d * s
    np.fill_diagonal(d_scaled, 1.0)
    
    # Numerical stability: clip very small distances to avoid log(0)
    d_scaled = np.clip(d_scaled, 1e-15, None)
    
    log_p = np.sum(np.log(d_scaled))
    log_score = log_p - n * math.log(n)
    return log_score

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
    deadline = start_time + 140.0  # Safety margin for 150s limit

    def regular_octagon(radius=1.0):
        angles = 2 * math.pi * np.arange(8) / 8
        return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

    best_pts = regular_octagon()
    best_log_score = _log_score_vec(best_pts)

    # Generate diverse initializations
    random.seed(12345)
    np.random.seed(12345)
    
    initializations = []
    
    # 1. Regular octagon
    initializations.append(regular_octagon().copy())
    
    # 2. Perturbed regular octagons (small to medium noise)
    for pert in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8]:
        for trial in range(2):
            base = regular_octagon()
            noise = np.random.randn(8, 2) * pert
            init = base + noise
            initializations.append(init)
            
    # 3. Symmetry-breaking configurations
    # Example: Two squares rotated by a small angle, not 45 degrees
    for angle_offset in [0.1, 0.2, 0.3, 0.5]:
        a1 = 2 * math.pi * np.arange(4) / 4
        a2 = 2 * math.pi * np.arange(4) / 4 + angle_offset
        pts1 = np.column_stack([np.cos(a1), np.sin(a1)])
        pts2 = np.column_stack([np.cos(a2), np.sin(a2)])
        init = np.vstack([pts1, pts2])
        initializations.append(init)
        
    # 4. Random points on unit circle
    for trial in range(5):
        angles = np.sort(np.random.uniform(0, 2 * math.pi, 8))
        init = np.column_stack([np.cos(angles), np.sin(angles)])
        initializations.append(init)
        
    # 5. Random points in unit disk
    for trial in range(5):
        init = np.random.randn(8, 2)
        D = np.max(np.sqrt(np.sum((init[:, np.newaxis, :] - init[np.newaxis, :])**2, axis=2)))
        if D > 1e-12:
            init = init * (2.0 / D)
        initializations.append(init)

    # Shuffle initializations to mix up the search
    random.shuffle(initializations)

    # Run optimization
    # We use Powell for global-ish search and Nelder-Mead for local polish
    # To save time, we run Powell on all inits, then NM on the top few results
    
    candidates = []
    
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
                options={'maxiter': 100, 'xtol': 1e-9, 'ftol': 1e-9}
            )
            pts_p = res_p.x.reshape(8, 2)
            ls_p = _log_score_vec(pts_p)
            candidates.append((ls_p, pts_p))
            
            if ls_p > best_log_score:
                best_log_score = ls_p
                best_pts = pts_p
        except Exception:
            pass

    # Sort candidates by score and take the top 5 for Nelder-Mead polish
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    top_candidates = candidates[:5]
    
    for ls_c, pts_c in top_candidates:
        if time.time() > deadline - 5.0:
            break
        
        start_nm = pts_c.flatten()
        try:
            res_nm = minimize(
                _objective, start_nm,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            pts_nm = res_nm.x.reshape(8, 2)
            ls_nm = _log_score_vec(pts_nm)
            
            if ls_nm > best_log_score:
                best_log_score = ls_nm
                best_pts = pts_nm
        except Exception:
            pass

    # Final high-precision polish on the absolute best
    if time.time() < deadline - 1.0:
        try:
            res_final = minimize(
                _objective, best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True}
            )
            pts_final = res_final.x.reshape(8, 2)
            ls_final = _log_score_vec(pts_final)
            
            if ls_final > best_log_score:
                best_log_score = ls_final
                best_pts = pts_final
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
