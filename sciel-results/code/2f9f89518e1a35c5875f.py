# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _keich_axis(N):
    """Keich binary construction (baseline) for one axis: N base positions."""
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    out = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        out[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    return out


def _union_len_sq(motion, base, M=200):
    """
    Calculate the integral of the squared 1D union length over c in [0, 1].
    motion: (K,) array of slopes.
    base: (K,) array of base positions.
    Returns the value of the integral.
    """
    K = len(base)
    width = 1.0 / 8.0
    c = np.linspace(0.0, 1.0, M)
    dc = 1.0 / (M - 1)
    L_sq_sum = 0.0
    
    for m in range(M):
        lo = base + c[m] * motion
        hi = lo + width
        o = np.argsort(lo)
        s_lo = lo[o]
        s_hi = hi[o]
        
        L = 0.0
        cur_max_hi = -np.inf
        for t in range(K):
            if s_hi[t] > cur_max_hi:
                start = max(s_lo[t], cur_max_hi)
                L += s_hi[t] - start
                cur_max_hi = s_hi[t]
        L_sq_sum += L * L
        
    return dc * L_sq_sum


def _solve_axis(motion, base0, iters=300):
    """
    Minimize the integral of squared union length using L-BFGS-B with finite differences.
    """
    K = len(base0)
    
    def fun(x):
        return _union_len_sq(motion, x)
        
    res = minimize(fun, base0.copy(), method="L-BFGS-B",
                   options={"maxiter": iters, "ftol": 1e-12, "gtol": 1e-8, "maxls": 20})
    return res.x


def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    # x-positions depend only on i (cluster), y only on j.
    motion = np.arange(N, dtype=np.float64) / N  # c*i/8

    base_x = _keich_axis(N).copy()
    base_y = _keich_axis(N).copy()

    # Try to optimize
    try:
        base_x = _solve_axis(motion, base_x, iters=300)
    except Exception:
        pass
    try:
        base_y = _solve_axis(motion, base_y, iters=300)
    except Exception:
        pass

    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()

    positions = np.stack([base_x[i_flat], base_y[j_flat]], axis=1)
    return positions


# EVOLVE-BLOCK-END
