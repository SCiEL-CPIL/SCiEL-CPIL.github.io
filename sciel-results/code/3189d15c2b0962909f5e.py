import numpy as np
import time
import scipy.optimize as optimize

def kakeya_needle_3d8() -> np.ndarray:
    start_time = time.time()
    timeout = 580  # safety margin before 630s limit
    
    N = 8
    
    def compute_1d_volume(p):
        """Compute integral of union length of 8 intervals over c in [0,1].
        p has shape (8,). Interval i at height c: [p[i] + c*i/N, p[i] + c*i/N + 1/N]
        """
        # Use Gauss-Legendre quadrature for accuracy
        c_vals, w_vals = np.polynomial.legendre.leggauss(64)
        c_vals = 0.5 * (c_vals + 1)  # map to [0,1]
        w_vals = 0.5 * w_vals
        
        # For each c, compute union length of 8 intervals
        # starts[i] = p[i] + c * i / N
        # ends[i] = starts[i] + 1/N
        idx = np.arange(N)
        # Vectorize over c: shape (64, 8)
        starts = p[None, :] + c_vals[:, None] * idx[None, :] / N
        ends = starts + 1.0 / N
        
        # Sort by start for each c
        order = np.argsort(starts, axis=1)
        s = np.take_along_axis(starts, order, axis=1)
        e = np.take_along_axis(ends, order, axis=1)
        
        # Merge intervals: for each row, compute union length
        # cur_s = s[:, 0], cur_e = e[:, 0]
        cur_s = s[:, 0].copy()
        cur_e = e[:, 0].copy()
        total_len = np.zeros(len(c_vals))
        
        for k in range(1, N):
            # If s[:, k] <= cur_e, overlap
            overlap = s[:, k] <= cur_e
            cur_e = np.where(overlap, np.maximum(cur_e, e[:, k]), e[:, k])
            cur_s = np.where(overlap, cur_s, s[:, k])
            # When not overlapping, add previous segment
            total_len += np.where(~overlap, cur_e - cur_s, 0.0)
            # When overlapping, cur_s stays, cur_e updates
            # Actually need to be more careful:
        
        # Let me redo the merge properly
        total_len = np.zeros(len(c_vals))
        for ci in range(len(c_vals)):
            s_ci = s[ci]
            e_ci = e[ci]
            cs = s_ci[0]
            ce = e_ci[0]
            length = 0.0
            for k in range(1, N):
                if s_ci[k] <= ce:
                    ce = max(ce, e_ci[k])
                else:
                    length += ce - cs
                    cs = s_ci[k]
                    ce = e_ci[k]
            length += ce - cs
            total_len[ci] = length
        
        return np.sum(total_len * w_vals)
    
    def compute_2d_volume(px, py):
        """Compute 2D volume as product of 1D volumes (separable case)."""
        vx = compute_1d_volume(px)
        vy = compute_1d_volume(py)
        return vx * vy
    
    # Initial guess: linear spread
    # For 1D: intervals [p_i + ci/8, p_i + ci/8 + 1/8]
    # At c=0: intervals are [p_i, p_i + 1/8]
    # At c=1: intervals are [p_i + i/8, p_i + i/8 + 1/8]
    # To minimize overlap, we want intervals to be spread out
    # Try p_i = -i/8 * alpha for some alpha
    
    best_vol = float('inf')
    best_px = -np.arange(N) / N
    best_py = -np.arange(N) / N
    
    # Multi-start optimization for 1D problem
    def obj_1d(p_opt):
        p_full = np.concatenate(([0.0], p_opt))
        return compute_1d_volume(p_full)
    
    # Try several initial configurations
    init_configs = [
        -np.arange(N) / N,           # linear negative
        -np.arange(N) / (N * 0.5),   # wider spread
        -np.arange(N) / (N * 1.5),   # tighter spread
        np.linspace(0, -1, N),       # another linear
        -np.arange(N)**1.5 / (N**1.5), # nonlinear
        np.sin(np.arange(N) * np.pi / N) * 0.5 - 0.25,
    ]
    
    for init in init_configs:
        if time.time() - start_time > timeout - 60:
            break
        p_init_opt = init[1:] - init[0]
        try:
            res = optimize.minimize(obj_1d, p_init_opt, method='Nelder-Mead',
                                     options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-9})
            if res.success or res.fun < best_vol:
                px_candidate = np.concatenate(([0.0], res.x))
                vol_candidate = compute_1d_volume(px_candidate)
                if vol_candidate < best_vol:
                    best_vol = vol_candidate
                    best_px = px_candidate.copy()
                    best_py = px_candidate.copy()
        except Exception:
            pass
    
    # Refine with more iterations if time allows
    if time.time() - start_time < timeout - 30:
        try:
            p_init_opt = best_px[1:] - best_px[0]
            res = optimize.minimize(obj_1d, p_init_opt, method='Nelder-Mead',
                                     options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            px_candidate = np.concatenate(([0.0], res.x))
            vol_candidate = compute_1d_volume(px_candidate)
            if vol_candidate < best_vol:
                best_vol = vol_candidate
                best_px = px_candidate.copy()
                best_py = px_candidate.copy()
        except Exception:
            pass
    
    # Construct positions: row k, i = k//8, j = k%8
    # x_k = best_px[i], y_k = best_py[j]
    positions = np.zeros((N * N, 2))
    for k in range(N * N):
        i = k // N
        j = k % N
        positions[k, 0] = best_px[i]
        positions[k, 1] = best_py[j]
    
    # Validate
    if not np.all(np.isfinite(positions)):
        # Fallback to simple construction
        positions = np.zeros((N * N, 2))
        for k in range(N * N):
            i = k // N
            j = k % N
            positions[k, 0] = -i / N
            positions[k, 1] = -j / N
    
    return positions
