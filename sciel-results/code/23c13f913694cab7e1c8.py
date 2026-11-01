"""Circle packing: 50 equal circles in the unit square, maximize common radius."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 50
DEADLINE = 150.0          # hard wall-clock limit
SAFETY = 8.0              # stop searching this many seconds before the limit
START_TIME = time.time()


def _elapsed():
    return time.time() - START_TIME


def _radius(c):
    """Exact score: min pairwise half-distance and min distance to walls."""
    d = c[:, None, :] - c[None, :, :]
    dist = np.sqrt((d * d).sum(-1))
    np.fill_diagonal(dist, np.inf)
    pair = 0.5 * dist.min()
    wall = 0.5 - np.abs(c).max()
    return min(float(pair), float(wall))


def _softmin_penalty(c, beta):
    """Smooth surrogate for the min-distance objective (negative of soft-min)."""
    d = c[:, None, :] - c[None, :, :]
    dist = np.sqrt((d * d).sum(-1))
    np.fill_diagonal(dist, np.inf)
    w = 0.5 * dist
    logsum = np.logaddexp.reduce(w, axis=1)
    pair = np.logaddexp.reduce(logsum)
    # wall: distance of each point to nearest wall (0.5 - |x|, 0.5 - |y|)
    m = 0.5 - np.abs(c)
    logsum_w = np.logaddexp.reduce(m, axis=1)
    wall = np.logaddexp.reduce(logsum_w)
    # soft-min over the two groups
    val = np.logaddexp(pair, wall) / beta
    return val


def _objective(c, beta):
    return _softmin_penalty(c, beta)


def _grad(c, beta):
    d = c[:, None, :] - c[None, :, :]
    dist = np.sqrt((d * d).sum(-1))
    np.fill_diagonal(dist, np.inf)
    w = 0.5 * dist
    logsum = np.logaddexp.reduce(w, axis=1)
    pair = np.logaddexp.reduce(logsum)
    m = 0.5 - np.abs(c)
    logsum_w = np.logaddexp.reduce(m, axis=1)
    wall = np.logaddexp.reduce(logsum_w)
    val = np.logaddexp(pair, wall) / beta
    # softmax weights
    pair_w = np.exp(w - logsum[:, None])          # (N, N)
    pair_w = np.nan_to_num(pair_w, nan=0.0)
    p_top = np.exp(pair - val * beta)             # scalar weight on pair group
    w_top = np.exp(wall - val * beta)             # scalar weight on wall group
    grad = np.zeros_like(c)
    # pair contribution: grad_i = p_top * sum_j pair_w[i,j] * (c_i - c_j)
    diff = c[:, None, :] - c[None, :, :]
    pair_grad = (pair_w[:, None, :, None] * diff).sum((1, 2))
    # wall contribution: grad_i = w_top * sum over k of (1 - sign(c_ik)) * m_weight
    # m[i,k] = 0.5 - |c_ik|; d m[i,k]/d c_ik = -sign(c_ik)
    mw = np.exp(m - logsum_w[:, None])            # (N,2)
    mw = np.nan_to_num(mw, nan=0.0)
    wall_grad = -mw * np.sign(c)
    grad = p_top * pair_grad + w_top * wall_grad
    return grad / beta


def _random_start(seed):
    rng = np.random.default_rng(seed)
    c = rng.uniform(-0.5, 0.5, size=(N, 2))
    return c


def _jitter(c, seed, amt):
    rng = np.random.default_rng(seed)
    return c + amt * rng.standard_normal(c.shape)


def construct():
    bounds = [(-0.5, 0.5)] * (2 * N)

    # Grid seed (known-good starting point)
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])[:N]

    best = grid.copy()
    best_r = _radius(best)

    # Stage 1: coarse search with moderate beta
    beta_coarse = 1500.0
    start_seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    start_seeds = [None] + start_seeds  # None => grid seed first

    for s in start_seeds:
        if _elapsed() > DEADLINE - SAFETY:
            break
        if s is None:
            c0 = best.copy()
        else:
            c0 = _random_start(s)
        try:
            res = minimize(_objective, c0, jac=lambda x: _grad(x, beta_coarse),
                           method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 300, 'maxfun': 5000,
                                    'ftol': 1e-12, 'gtol': 1e-8, 'iprint': 0})
            c = np.clip(res.x, -0.5, 0.5)
            r = _radius(c)
            if np.isfinite(r) and r > best_r:
                best_r = r
                best = c.copy()
        except Exception:
            pass

    # Stage 2: polish the best solution with high beta and tight tolerances
    if _elapsed() < DEADLINE - SAFETY:
        try:
            res = minimize(_objective, best, jac=lambda x: _grad(x, 8000.0),
                           method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 2000, 'maxfun': 10000,
                                    'ftol': 1e-15, 'gtol': 1e-12, 'iprint': 0})
            c = np.clip(res.x, -0.5, 0.5)
            r = _radius(c)
            if np.isfinite(r) and r > best_r:
                best_r = r
                best = c.copy()
        except Exception:
            pass

    # Stage 3: local perturbations around the best, re-polish
    if _elapsed() < DEADLINE - SAFETY:
        rng = np.random.default_rng(12345)
        for attempt in range(40):
            if _elapsed() > DEADLINE - SAFETY:
                break
            amt = 0.01 / (1.0 + attempt * 0.5)
            c0 = _jitter(best, 100000 + attempt, amt)
            c0 = np.clip(c0, -0.5, 0.5)
            try:
                res = minimize(_objective, c0, jac=lambda x: _grad(x, 8000.0),
                               method='L-BFGS-B', bounds=bounds,
                               options={'maxiter': 800, 'maxfun': 4000,
                                        'ftol': 1e-15, 'gtol': 1e-12, 'iprint': 0})
                c = np.clip(res.x, -0.5, 0.5)
                r = _radius(c)
                if np.isfinite(r) and r > best_r:
                    best_r = r
                    best = c.copy()
            except Exception:
                pass

    # Final safety: ensure valid finite output
    best = np.clip(best, -0.5, 0.5)
    if not np.all(np.isfinite(best)) or best.shape != (N, 2):
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best = np.array([(x, y) for y in ticks for x in ticks])[:N]
    return best
