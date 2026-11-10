# EVOLVE-BLOCK-START
import time
import numpy as np

_N = 8
_NF = float(_N)
_W = 1.0 / _NF          # interval width = 1/8
_C_SAMPLES = 64
_START = time.time()
_DEADLINE = 585.0       # safety margin before 630s kill


def _union_length_1d(p, c):
    """Total covered length of N intervals over c (vectorized over c)."""
    # p: (N,), c: (M,)
    a = p[None, :] + c[:, None] * (np.arange(_N) / _NF)   # (M, N)
    b = a + _W
    order = np.argsort(a, axis=1)
    a_sorted = np.take_along_axis(a, order, axis=1)
    b_sorted = np.take_along_axis(b, order, axis=1)
    lo = np.maximum.reduce(b_sorted[:, :-1])
    hi = a_sorted[:, 1:]
    seg = np.maximum(hi - lo, 0.0)
    total = seg.sum(axis=1) + _W
    return total


def _cost_1d(p):
    """Smooth surrogate for the 1D union length (log-sum-exp over c)."""
    c = np.linspace(0.0, 1.0, _C_SAMPLES)
    u = _union_length_1d(p, c)
    m = u.max()
    return (m + np.log(np.exp(u - m).sum())) / _C_SAMPLES


def _optimize_1d(p0, max_iters, deadline):
    """BFGS on the smooth 1D surrogate, with deadline guard."""
    from scipy.optimize import fmin_bfgs
    best = p0.copy()
    best_c = _cost_1d(best)
    try:
        res = fmin_bfgs(_cost_1d, p0, maxfun=max_iters, full_output=True, disp=False)
        x, fx, ier, msg = res
        if np.all(np.isfinite(x)) and np.isfinite(fx) and fx < best_c:
            best, best_c = x, fx
    except Exception:
        pass
    return best, best_c


def _refine_1d(p, budget):
    """Local polish: alternating anneal restarts + BFGS within budget."""
    best = p.copy()
    best_c = _cost_1d(best)
    rng = np.random.default_rng(7)
    n = 0
    while time.time() < budget:
        # occasional restart with a perturbation to escape local minima
        if n % 4 == 3:
            cand = best + rng.normal(0.0, 0.03, _N)
        else:
            cand = best + rng.normal(0.0, 0.01, _N)
        cand_c = _cost_1d(cand)
        # accept with annealing temperature
        if cand_c < best_c:
            best, best_c = cand, cand_c
        else:
            T = 0.02 * (0.8 ** n)
            if np.log(np.random.rand()) > (cand_c - best_c) / max(T, 1e-12):
                best, best_c = cand, cand_c
        # polish with BFGS
        polished, pc = _optimize_1d(best, 120, budget)
        if pc < best_c:
            best, best_c = polished, pc
        n += 1
    return best, best_c


def kakeya_needle_3d8() -> np.ndarray:
    """Optimize the 1D shear-union length, use it for both x and y axes."""
    # initial guess: Keich binary construction (good baseline)
    pos = int(np.log2(_N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(_N, dtype=np.float64)
    for idx in range(_N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)

    # initial BFGS polish from the baseline
    p, c = _optimize_1d(keich.copy(), 300, _DEADLINE)

    # refine with annealing + BFGS within the deadline budget
    p, c = _refine_1d(p, _DEADLINE)

    # build 64 base positions: x depends on i, y depends on j (product structure)
    i_grid, j_grid = np.meshgrid(np.arange(_N), np.arange(_N), indexing="ij")
    base_x = p[i_grid.flatten()]
    base_y = p[j_grid.flatten()]
    positions = np.stack([base_x, base_y], axis=1)
    return positions


# EVOLVE-BLOCK-END
