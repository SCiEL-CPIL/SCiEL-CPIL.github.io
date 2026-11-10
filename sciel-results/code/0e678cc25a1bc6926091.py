import numpy as np
import time
from scipy.optimize import minimize

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    delta = 1.0 / N
    
    # Keich binary construction for initial guess
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    X0 = keich.copy()
    Y0 = keich.copy()
    
    i_coeff = np.arange(N, dtype=np.float64) / N
    j_coeff = np.arange(N, dtype=np.float64) / N
    
    # Pre-compute a dense set of c-points for quadrature
    # Use Gauss-Legendre quadrature with many points for smoothness
    n_quad = 256
    c_quad, w_quad = np.polynomial.legendre.leggauss(n_quad)
    # Map from [-1,1] to [0,1]
    c_quad = 0.5 * (c_quad + 1.0)
    w_quad = 0.5 * w_quad
    
    def _union_length_1d_vec(A, B, c_arr):
        """Vectorized union length for multiple c values.
        A: (N,), B: (N,), c_arr: (M,) -> returns (M,)
        """
        M = len(c_arr)
        starts = A[None, :] + B[None, :] * c_arr[:, None]  # (M, N)
        ends = starts + delta
        # Sort by start
        order = np.argsort(starts, axis=1)  # (M, N)
        s = np.take_along_axis(starts, order, axis=1)
        e = np.take_along_axis(ends, order, axis=1)
        # Sweep line
        total = np.zeros(M)
        cur_s = s[:, 0].copy()
        cur_e = e[:, 0].copy()
        for k in range(1, N):
            gap = s[:, k] - cur_e
            # Where there's a gap, add the previous segment
            total += np.where(gap > 0, cur_e - cur_s, 0.0)
            # Update cur_s, cur_e
            cur_s = np.where(gap > 0, s[:, k], cur_s)
            cur_e = np.maximum(cur_e, e[:, k])
        total += cur_e - cur_s
        return total
    
    def compute_volume_smooth(X, Y):
        """Volume via dense quadrature - smooth in X,Y."""
        Lx = _union_length_1d_vec(X, i_coeff, c_quad)  # (n_quad,)
        Ly = _union_length_1d_vec(Y, j_coeff, c_quad)  # (n_quad,)
        # Volume = integral of Lx(c)*Ly(c) dc
        vol = np.sum(w_quad * Lx * Ly)
        return vol
    
    def objective(params):
        X = params[:N]
        Y = params[N:]
        return compute_volume_smooth(X, Y)
    
    p0 = np.concatenate([X0, Y0])
    
    start_time = time.time()
    deadline = start_time + 590.0  # 40s safety margin
    
    best_p = p0.copy()
    best_v = objective(p0)
    
    # Multi-restart with diverse strategies
    rng_opt = np.random.default_rng(42)
    n_restarts = 20
    
    for r in range(n_restarts):
        if time.time() > deadline - 90.0:
            break
        
        if r == 0:
            p_init = p0.copy()
        elif r == 1:
            # Try symmetric: X = Y
            p_init = np.concatenate([X0, X0])
        elif r == 2:
            # Try reversed
            p_init = np.concatenate([X0[::-1], Y0])
        elif r == 3:
            # Try shifted
            p_init = np.concatenate([X0 + 0.1, Y0 - 0.1])
        else:
            # Perturb from best with decreasing noise
            noise_scale = 0.2 / (1 + (r - 4) * 0.2)
            p_init = best_p + rng_opt.normal(0, noise_scale, 2 * N)
        
        try:
            res = minimize(objective, p_init, method='Powell',
                          options={'maxiter': 800, 'xtol': 1e-13, 'ftol': 1e-13})
            if res.fun < best_v:
                best_v = res.fun
                best_p = res.x.copy()
        except Exception:
            pass
    
    # L-BFGS-B polish with numerical gradient
    if time.time() < deadline - 60.0:
        try:
            res2 = minimize(objective, best_p, method='L-BFGS-B',
                           options={'maxiter': 2000, 'gtol': 1e-14, 'ftol': 1e-14,
                                    'maxfun': 50000})
            if res2.fun < best_v:
                best_v = res2.fun
                best_p = res2.x.copy()
        except Exception:
            pass
    
    # Nelder-Mead polish
    if time.time() < deadline - 30.0:
        try:
            res3 = minimize(objective, best_p, method='Nelder-Mead',
                           options={'maxiter': 8000, 'xatol': 1e-13, 'fatol': 1e-13})
            if res3.fun < best_v:
                best_v = res3.fun
                best_p = res3.x.copy()
        except Exception:
            pass
    
    # Final high-precision Powell polish
    if time.time() < deadline - 10.0:
        try:
            res4 = minimize(objective, best_p, method='Powell',
                           options={'maxiter': 3000, 'xtol': 1e-15, 'ftol': 1e-15})
            if res4.fun < best_v:
                best_v = res4.fun
                best_p = res4.x.copy()
        except Exception:
            pass
    
    X_opt = best_p[:N]
    Y_opt = best_p[N:]
    
    # Build the 64 positions: row k has i=k//8, j=k%8
    positions = np.empty((64, 2), dtype=np.float64)
    for k in range(64):
        i = k // N
        j = k % N
        positions[k, 0] = X_opt[i]
        positions[k, 1] = Y_opt[j]
    
    return positions
