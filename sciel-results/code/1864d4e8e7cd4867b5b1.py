# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = time.time() + 595.0  # wall-clock safety margin (limit is 630 s)
_N = 8
_S = 1.0 / _N


def _union_len(p):
    """Exact union length of the N moving intervals (piecewise-linear)."""
    p = np.asarray(p, dtype=np.float64)
    idx = np.arange(_N)
    starts = p[None, :] + idx[None, :] * _S
    ends = starts + _S
    cand = np.unique(np.concatenate([starts, ends]))
    if cand.size < 2:
        return 0.0
    L = cand[-1] - cand[0]
    for k in range(cand.size - 1):
        a, b = cand[k], cand[k + 1]
        m = 0.5 * (a + b)
        L -= np.sum((m > starts) & (m < ends)) * (b - a)
    return float(L)


def _grad_union(p, s):
    """Smooth surrogate (log-sum-exp) union length + gradient."""
    p = np.asarray(p, dtype=np.float64)
    idx = np.arange(_N)
    starts = p[None, :] + idx[None, :] * _S
    ends = starts + _S
    cand = np.unique(np.concatenate([starts, ends]))
    if cand.size < 2:
        return 0.0, np.zeros_like(p)
    L = cand[-1] - cand[0]
    sm = s * np.log(_N)
    g = np.zeros_like(p)
    for k in range(cand.size - 1):
        a, b = cand[k], cand[k + 1]
        m = 0.5 * (a + b)
        inx = (m > starts) & (m < ends)
        c = int(np.count_nonzero(inx))
        if c == 0:
            L -= (b - a)
            continue
        if c == _N:
            continue
        w = (b - a) / sm
        L -= np.exp(-c * w) * sm
        e = np.exp(-w * inx)
        e_sum = e.sum()
        g += w * (e / e_sum) * inx
    return float(L), g


def _smooth_obj(p):
    val, _ = _grad_union(p, 1e-4)
    return val


def _smooth_grad(p):
    _, g = _grad_union(p, 1e-4)
    return g


def _nm_obj(p):
    return _union_len(p)


def _gd_from(p, s0, iters, step):
    p = p.copy()
    s = s0
    best = _union_len(p)
    for _ in range(iters):
        f, g = _grad_union(p, s)
        g = g / (np.linalg.norm(g) + 1e-12)
        p_new = p - step * g
        if _union_len(p_new) < best:
            p = p_new
            best = _union_len(p)
            s = max(s * 0.9, 1e-6)
        else:
            s = min(s * 1.3, 1.0)
    return p, best


def _optimize_1d():
    """Minimize the 1-D union length over the N base positions."""
    best_p, best_l = None, np.inf
    # (1) Greedy chain: order intervals to minimize the span, then polish.
    order = list(range(_N))
    for _ in range(20):
        cur = order[-1]
        nxt = min(range(_N), key=lambda k: abs((cur + k * _S) - (order[-1] + cur * _S)))
        order.append(nxt)
    p = np.empty(_N)
    for pos, i in enumerate(order):
        p[i] = pos * _S
    p, l = _gd_from(p, 1e-4, 400, 0.05)
    best_p, best_l = p, l
    # (2) Random restarts: L-BFGS-B on the smoothed surrogate + Nelder-Mead polish.
    for r in range(60):
        if time.time() > _DEADLINE - 10:
            break
        p0 = np.random.default_rng(1000 + r).uniform(-2.0, 2.0, _N)
        try:
            res = minimize(_smooth_obj, p0, jac=_smooth_grad, method="L-BFGS-B",
                           options={"maxiter": 300, "gtol": 1e-5})
            p = res.x
        except Exception:
            p = p0
        p, l = _gd_from(p, 1e-4, 300, 0.05)
        try:
            res = minimize(_nm_obj, p, method="Nelder-Mead",
                           options={"maxiter": 800, "xatol": 1e-9, "fatol": 1e-12,
                                    "adaptive": True})
            p2, l2 = res.x, float(res.fun)
        except Exception:
            p2, l2 = p, l
        if l2 < best_l - 1e-12:
            best_p, best_l = p2, l2
    return best_p, best_l


def kakeya_needle_3d8() -> np.ndarray:
    """Return base positions (x_k, y_k) for the 64 sheared tubes (N = 8).

    The union volume factors as Lx * Ly * 1 (x-union over i times y-union over j),
    and the two 1-D problems are identical, so we solve one 1-D problem and use the
    same base positions for both axes.
    """
    p, _ = _optimize_1d()
    p = np.asarray(p, dtype=np.float64)
    i_grid, j_grid = np.meshgrid(np.arange(_N), np.arange(_N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    base_x = p[i_flat]
    base_y = p[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    if not np.all(np.isfinite(positions)):
        # Fallback to the Keich construction (guaranteed finite).
        pos = int(np.log2(_N))
        j_terms = np.arange(1, pos + 1)
        keich = np.empty(_N, dtype=np.float64)
        for idx in range(_N):
            eps = np.array(list(str(bin(idx))[2:]), dtype=int)
            eps = np.pad(eps, (pos - len(eps), 0), "constant")
            keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
        base_x = keich[i_flat]
        base_y = keich[j_flat]
        positions = np.stack([base_x, base_y], axis=1)
    return positions


# EVOLVE-BLOCK-END
