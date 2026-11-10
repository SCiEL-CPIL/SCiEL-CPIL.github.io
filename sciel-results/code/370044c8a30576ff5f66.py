# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_START = time.time()
_LIMIT = 630.0
_MARGIN = 30.0


def _ok():
    return (time.time() - _START) < (_LIMIT - _MARGIN)


def _union_len(lo, hi):
    """Length of the union of intervals [lo[t], hi[t]]."""
    n = len(lo)
    if n == 0:
        return 0.0
    order = np.argsort(lo)
    lo = lo[order]
    hi = hi[order]
    total = 0.0
    cl = lo[0]
    ch = hi[0]
    for t in range(1, n):
        if lo[t] > ch:
            total += ch - cl
            cl = lo[t]
            ch = hi[t]
        elif hi[t] > ch:
            ch = hi[t]
    total += ch - cl
    return total


def _exact_volume(f, g, n_c=128):
    """Exact 3-D union volume for product structure x=f(i), y=g(j)."""
    N = len(f)
    idx = np.arange(N, dtype=np.float64)
    cs = np.linspace(0.0, 1.0, n_c)
    vals = np.empty(n_c)
    for k in range(n_c):
        c = cs[k]
        lo_x = f + c * idx / N
        hi_x = f + (1.0 + idx) / N
        lo_y = g + c * idx / N
        hi_y = g + (1.0 + idx) / N
        vals[k] = _union_len(lo_x, hi_x) * _union_len(lo_y, hi_y)
    h = 1.0 / (n_c - 1)
    return h * (0.5 * vals[0] + vals[1:-1].sum() + 0.5 * vals[-1])


def _keich(N):
    """Keich binary construction for 1-D Kakeya."""
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1, dtype=np.float64)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        bits = list(bin(idx)[2:])
        bits = [0] * (pos - len(bits)) + bits
        eps = np.array(bits, dtype=np.float64)
        keich[idx] = np.sum((1.0 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    return keich


def _mc_volume(f, g, n_pts=50000, seed=12345):
    """Fast Monte Carlo volume estimate for product structure."""
    N = len(f)
    idx = np.arange(N, dtype=np.float64)
    i_grid, j_grid = np.meshgrid(idx, idx, indexing="ij")
    base_x = f[i_grid.flatten()]
    base_y = g[j_grid.flatten()]
    X0 = np.min(base_x)
    X1 = np.max(base_x + (1.0 + i_grid.flatten()) / N)
    Y0 = np.min(base_y)
    Y1 = np.max(base_y + (1.0 + j_grid.flatten()) / N)
    if (X1 - X0) <= 1e-9 or (Y1 - Y0) <= 1e-9:
        return 1e6
    
    rng = np.random.default_rng(seed)
    a = rng.uniform(X0, X1, n_pts)
    b = rng.uniform(Y0, Y1, n_pts)
    c = rng.uniform(0.0, 1.0, n_pts)
    
    # For each point, check if it's inside any tube
    # Tube (i,j): x in [f[i] + c*i/N, f[i] + (1+i)/N], y in [g[j] + c*j/N, g[j] + (1+j)/N]
    # Vectorize over c for each (i,j)
    inside = np.zeros(n_pts, dtype=bool)
    for i in range(N):
        for j in range(N):
            lo_x = f[i] + c * i / N
            hi_x = f[i] + (1.0 + i) / N
            lo_y = g[j] + c * j / N
            hi_y = g[j] + (1.0 + j) / N
            mask = (a >= lo_x) & (a <= hi_x) & (b >= lo_y) & (b <= hi_y)
            inside |= mask
    frac = inside.mean()
    return frac * (X1 - X0) * (Y1 - Y0)


def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    keich = _keich(N)
    
    # Objective over 16 params: first 8 = f (x by i), next 8 = g (y by j).
    def obj(fg):
        f = fg[:N]
        g = fg[N:]
        return _exact_volume(f, g)
    
    # Initial best: Keich
    best = np.concatenate([keich, keich])
    best_v = obj(best)
    
    # Multi-start Nelder-Mead with diverse initializations
    rng = np.random.default_rng(42)
    
    # Strategy: try many random restarts with different scales
    restarts = 60
    for r in range(restarts):
        if not _ok():
            break
        
        if r == 0:
            x0 = best.copy()
        elif r == 1:
            # Small perturbation of best
            x0 = best + rng.normal(0.0, 0.02, size=2 * N)
        elif r == 2:
            # Moderate perturbation
            x0 = best + rng.normal(0.0, 0.05, size=2 * N)
        elif r == 3:
            # Larger perturbation
            x0 = best + rng.normal(0.0, 0.1, size=2 * N)
        elif r == 4:
            # Different structure: try symmetric
            x0 = np.concatenate([keich, keich[::-1]])
        elif r == 5:
            # Try reversed
            x0 = np.concatenate([keich[::-1], keich[::-1]])
        elif r == 6:
            # Try shifted
            x0 = np.concatenate([keich + 0.1, keich - 0.1])
        elif r == 7:
            # Try scaled
            x0 = np.concatenate([keich * 0.8, keich * 1.2])
        else:
            # Random restarts with varying scale
            scale = 0.03 + 0.07 * (r % 5) / 4.0
            x0 = best + rng.normal(0.0, scale, size=2 * N)
        
        try:
            res = minimize(obj, x0, method="Nelder-Mead",
                           options={"maxiter": 3000, "xatol": 1e-9, "fatol": 1e-11,
                                    "adaptive": True})
            if np.all(np.isfinite(res.x)) and res.fun < best_v:
                best = res.x.copy()
                best_v = res.fun
        except Exception:
            pass
    
    # Final refinement with tighter tolerances
    if _ok():
        try:
            res = minimize(obj, best, method="Nelder-Mead",
                           options={"maxiter": 5000, "xatol": 1e-10, "fatol": 1e-12,
                                    "adaptive": True})
            if np.all(np.isfinite(res.x)) and res.fun < best_v:
                best = res.x.copy()
                best_v = res.fun
        except Exception:
            pass
    
    # Second pass: try different initializations from scratch
    if _ok():
        for r in range(30):
            if not _ok():
                break
            # Random starting points
            x0 = rng.uniform(-0.5, 0.5, size=2 * N)
            try:
                res = minimize(obj, x0, method="Nelder-Mead",
                               options={"maxiter": 2000, "xatol": 1e-9, "fatol": 1e-11,
                                        "adaptive": True})
                if np.all(np.isfinite(res.x)) and res.fun < best_v:
                    best = res.x.copy()
                    best_v = res.fun
            except Exception:
                pass
    
    f = best[:N]
    g = best[N:]
    
    # Safety checks
    if not (np.all(np.isfinite(f)) and np.all(np.isfinite(g))):
        f = keich
        g = keich
    
    # Guard: ensure a non-degenerate box
    x_lo = np.min(f)
    x_hi = np.max(f + (1.0 + np.arange(N)) / N)
    y_lo = np.min(g)
    y_hi = np.max(g + (1.0 + np.arange(N)) / N)
    if (x_hi - x_lo) <= 1e-9 or (y_hi - y_lo) <= 1e-9:
        f = keich
        g = keich
    
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    base_x = f[i_grid.flatten()]
    base_y = g[j_grid.flatten()]
    positions = np.stack([base_x, base_y], axis=1)
    return positions


# EVOLVE-BLOCK-END
