"""Rewritten solver: smooth soft-min objective + L-BFGS-B + repulsion polish."""
import time

N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _radius(c):
    """Largest valid common radius for centres c (N,3)."""
    c = np.asarray(c, dtype=np.float64)
    d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    pair = 0.5 * d.min()
    wall = 0.5 - np.abs(c).max()
    return float(min(pair, wall))


def _score(c):
    c = np.asarray(c, dtype=np.float64)
    if c.shape != (N, 3):
        return -np.inf
    if not np.all(np.isfinite(c)):
        return -np.inf
    if np.any(np.abs(c) > 0.5 + 1e-12):
        return -np.inf
    r = _radius(c)
    if not np.isfinite(r) or r < 0:
        return -np.inf
    return r


def _softmin(c, s):
    """Smooth surrogate for the true radius; returns (value, gradient)."""
    c = np.asarray(c, dtype=np.float64)
    n = c.shape[0]
    d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    # pair terms: 0.5*d_ij
    x_pair = 0.5 * d
    # wall terms: 0.5 - |c_ik|
    x_wall = 0.5 - np.abs(c)
    x = np.concatenate([x_pair.ravel(), x_wall.ravel()])
    # log-sum-exp softmin (soft min = -soft max)
    m = x.min()
    e = np.exp(s * (x - m))
    softmax = -m - np.log(e.sum()) / s
    # gradient wrt x of softmax
    p = e / e.sum()
    gx = -p / s
    # pair gradient
    gp = gx[: x_pair.size].reshape(n, n)
    diff = c[:, None, :] - c[None, :, :]
    inv = np.where(d > 1e-12, 1.0 / np.maximum(d, 1e-12), 0.0)
    # d(0.5*d_ij)/dc_i = 0.5 * (c_i - c_j)/d_ij
    grad = np.einsum('ij,ijk->ik', gp, diff * inv * 0.5)
    # wall gradient: d(0.5-|c_ik|)/dc_ik = -sign(c_ik)
    gw = gx[x_pair.size:].reshape(n, 3)
    grad = grad - np.sign(c) * gw
    return softmax, grad


def _grid_seed():
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()


def _repulsion_polish(c, iters, alpha):
    """Local polish: move each centre away from its nearest neighbours, keep inside."""
    c = np.array(c, dtype=np.float64)
    n = c.shape[0]
    for _ in range(iters):
        d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        # repulsion from a few nearest neighbours
        k = 6
        idx = np.argsort(d, axis=1)[:, :k]
        disp = np.zeros_like(c)
        for i in range(n):
            for j in idx[i]:
                v = c[i] - c[j]
                dd = np.linalg.norm(v)
                if dd > 1e-12:
                    disp[i] += v / dd
        c += alpha * disp
        c = np.clip(c, -0.5, 0.5)
    return c


def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # safety margin under 150s

    best = _grid_seed()
    best_r = _radius(best)

    rng = np.random.default_rng(0)

    def time_left():
        return deadline - time.time()

    # Phase 1: multi-restart smooth L-BFGS-B on soft-min objective
    s_list = [40.0, 80.0, 160.0, 320.0]
    restarts = 0
    while time_left() > 25.0:
        # start from best (occasionally from a jittered grid)
        if restarts % 3 == 0:
            x0 = best + rng.normal(0, 0.02, best.shape)
        else:
            x0 = best.copy()
        x0 = np.clip(x0, -0.5, 0.5)
        for s in s_list:
            if time_left() < 5.0:
                break
            res = minimize(
                lambda v: -_softmin(v, s)[0],
                x0,
                jac=lambda v: -_softmin(v, s)[1],
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10,
                         'maxls': 20},
            )
            cand = np.clip(res.x, -0.5, 0.5)
            cand = _repulsion_polish(cand, iters=50, alpha=0.01)
            r = _radius(cand)
            if r > best_r:
                best_r = r
                best = cand.copy()
        restarts += 1

    # Phase 2: final high-precision polish from the best configuration
    if time_left() > 3.0:
        c = best.copy()
        c = _repulsion_polish(c, iters=300, alpha=0.003)
        # tighten with a fine L-BFGS-B at large s
        res = minimize(
            lambda v: -_softmin(v, 512.0)[0],
            c,
            jac=lambda v: -_softmin(v, 512.0)[1],
            method='L-BFGS-B',
            options={'maxiter': 400, 'ftol': 1e-13, 'gtol': 1e-11, 'maxls': 20},
        )
        cand = np.clip(res.x, -0.5, 0.5)
        cand = _repulsion_polish(cand, iters=200, alpha=0.002)
        r = _radius(cand)
        if r > best_r:
            best_r = r
            best = cand.copy()

    # Final safety clip and validation
    best = np.clip(best, -0.5, 0.5)
    if _score(best) < 0:
        # fall back to the guaranteed-valid grid
        best = _grid_seed()
    return best
# EVOLVE-BLOCK-END
