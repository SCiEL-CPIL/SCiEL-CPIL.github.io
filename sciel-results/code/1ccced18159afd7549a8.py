import numpy as np
from scipy import optimize
import time

_N = 8
_FRAC = 1.0 / _N
_IDX8 = np.arange(_N) / _N

# Use a moderate grid size for speed. 1001 points is sufficient for optimization.
_C_GRID = np.linspace(0.0, 1.0, 1001)

def _union_len(C, base):
    """
    Compute the union length of intervals [base[i] + C*c_i, base[i] + C*c_i + 1/N]
    for each c in C.
    Returns an array of union lengths.
    """
    shifts = C[:, None] * _IDX8[None, :]
    l = base[None, :] + shifts
    r = l + _FRAC
    # Sort intervals by left endpoint
    order = np.argsort(l, axis=1)
    ls = np.take_along_axis(l, order, 1)
    rs = np.take_along_axis(r, order, 1)
    
    # Initialize union length with the first interval
    union = (rs[:, 0] - ls[:, 0]).copy()
    cur = rs[:, 0:1]
    
    # Iterate through the rest
    for k in range(1, _N):
        s = ls[:, k:k + 1]
        e = rs[:, k:k + 1]
        ne = np.maximum(cur, e)
        # Add the non-overlapping part
        union = union + np.maximum(ne - np.maximum(cur, s), 0.0)
        cur = ne
    return union

def _v_to_fg(v):
    """
    v has length 14: [f_1..f_7, g_1..g_7]
    f_0 = 0, g_0 = 0
    """
    f = np.zeros(_N)
    g = np.zeros(_N)
    f[1:] = v[:7]
    g[1:] = v[7:]
    return f, g

def _obj(v):
    """
    Exact 3-D union volume for the separable (f, g) ansatz.
    V = integral_0^1 Lx(c) * Ly(c) dc
    """
    f, g = _v_to_fg(v)
    uf = _union_len(_C_GRID, f)
    ug = _union_len(_C_GRID, g)
    
    # Use trapezoidal rule for integration
    val = float(np.trapz(uf * ug, _C_GRID))
    if not np.isfinite(val) or val <= 0:
        return 1e9
    return val

def kakeya_needle_3d8() -> np.ndarray:
    N = _N
    start_time = time.time()
    time_limit = 600.0
    safety_margin = 10.0
    
    def build(f, g):
        i = np.arange(N * N) // N
        j = np.arange(N * N) % N
        return np.stack([f[i], g[j]], axis=1)

    best_v = None
    best_val = 1e9
    
    def consider(v_candidate):
        nonlocal best_v, best_val
        if v_candidate is None:
            return
        val = _obj(v_candidate)
        if np.isfinite(val) and val < best_val:
            best_val = val
            best_v = np.asarray(v_candidate, dtype=float).copy()

    bounds = [(-5.0, 5.0)] * 14
    
    # 1. Multiple L-BFGS-B restarts from diverse initializations
    rng = np.random.default_rng(123)
    
    # Initial guesses
    inits = []
    # Zero
    inits.append(np.zeros(14))
    # Linear shear (positive)
    inits.append(np.concatenate([np.arange(1, N) / N, np.arange(1, N) / N]))
    # Linear shear (negative)
    inits.append(np.concatenate([-np.arange(1, N) / N, -np.arange(1, N) / N]))
    # Keich-like structured guess
    # f(i) = -i/N * 0.5, g(j) = -j/N * 0.5
    inits.append(np.concatenate([-np.arange(1, N) / (2*N), -np.arange(1, N) / (2*N)]))
    
    # Random restarts
    for _ in range(50):
        inits.append(rng.uniform(-2.0, 2.0, 14))

    for init in inits:
        if time.time() > start_time + time_limit - safety_margin:
            break
        try:
            res = optimize.minimize(_obj, init, method='L-BFGS-B', bounds=bounds, 
                                    options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10})
            consider(res.x)
        except Exception:
            pass

    # 2. Local refinement with Nelder-Mead if time remains
    if best_v is not None:
        if time.time() < start_time + time_limit - safety_margin:
            try:
                res = optimize.minimize(_obj, best_v, method='Nelder-Mead', 
                                        options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
                consider(res.x)
            except Exception:
                pass

    # 3. Final L-BFGS-B polish
    if best_v is not None:
        if time.time() < start_time + time_limit - safety_margin:
            try:
                res = optimize.minimize(_obj, best_v, method='L-BFGS-B', bounds=bounds,
                                        options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-12})
                consider(res.x)
            except Exception:
                pass

    # Construct final result
    if best_v is None:
        # Fallback
        f = np.zeros(N)
        g = np.zeros(N)
    else:
        f, g = _v_to_fg(best_v)

    pos = build(f, g)
    
    # Sanity check
    if pos.shape != (N * N, 2) or not np.all(np.isfinite(pos)):
        f = np.zeros(N)
        g = np.zeros(N)
        pos = build(f, g)
        
    return pos
