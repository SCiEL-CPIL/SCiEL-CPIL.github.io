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


def _union_len_sq(motion, base, M=800):
    """Exact integral over c in [0,1] of (union length of intervals
    [base[k]+c*motion[k], base[k]+c*motion[k]+1/8])^2.
    motion/base: (K,) arrays. Returns (value, gradient w.r.t. base)."""
    K = len(base)
    width = 1.0 / 8.0
    c = np.linspace(0.0, 1.0, M)
    dc = 1.0 / (M - 1)
    L = np.empty(M, dtype=np.float64)
    for m in range(M):
        lo = base + c[m] * motion
        hi = lo + width
        o = np.argsort(lo)
        s_lo = lo[o]
        s_hi = hi[o]
        L[m] = 0.0
        cur = -np.inf
        for t in range(K):
            if s_lo[t] > cur:
                cur = s_lo[t]
            if s_hi[t] > cur:
                L[m] += s_hi[t] - cur
                cur = s_hi[t]
    val = dc * np.sum(L * L)
    g = np.zeros(K, dtype=np.float64)
    for k in range(K):
        for m in range(M):
            lo = base + c[m] * motion
            hi = lo + width
            o = np.argsort(lo)
            s_lo = lo[o]
            s_hi = hi[o]
            cur = -np.inf
            contrib = 0.0
            for t in range(K):
                if s_lo[t] > cur:
                    cur = s_lo[t]
                if s_hi[t] > cur:
                    contrib += s_hi[t] - cur
                    cur = s_hi[t]
            g[k] += 2.0 * L[m] * contrib
    g *= dc
    return val, g


def _solve_axis(motion, base0, iters=150):
    """Minimize exact int L^2 dc with a non-overlap barrier at c=0."""
    K = len(base0)
    width = 1.0 / 8.0
    lam = 1e6
    g0 = np.zeros(K, dtype=np.float64)
    for k in range(K):
        for l in range(K):
            if k < l:
                g0[k] -= lam
                g0[l] += lam
            elif k > l:
                g0[k] += lam
                g0[l] -= lam

    def fun(x):
        v, g = _union_len_sq(motion, x)
        # barrier: penalize overlap at c=0
        for k in range(K):
            for l in range(K):
                if k < l:
                    d = x[k] - (x[l] + width)
                    if d < 0:
                        v += lam * d * d
                        g[k] += 2 * lam * d
                        g[l] -= 2 * lam * d
                elif k > l:
                    d = x[k] - (x[l] + width)
                    if d < 0:
                        v += lam * d * d
                        g[k] += 2 * lam * d
                        g[l] -= 2 * lam * d
        return v, g

    res = minimize(fun, base0.copy(), jac=True, method="L-BFGS-B",
                   options={"maxiter": iters, "ftol": 1e-14, "gtol": 1e-10})
    return res.x


def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    # x-positions depend only on i (cluster), y only on j.
    motion = np.arange(N, dtype=np.float64) / N  # c*i/8

    base_x = _keich_axis(N).copy()
    base_y = _keich_axis(N).copy()

    try:
        base_x = _solve_axis(motion, base_x, iters=150)
    except Exception:
        pass
    try:
        base_y = _solve_axis(motion, base_y, iters=150)
    except Exception:
        pass

    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()

    positions = np.stack([base_x[i_flat], base_y[j_flat]], axis=1)
    return positions


# EVOLVE-BLOCK-END
