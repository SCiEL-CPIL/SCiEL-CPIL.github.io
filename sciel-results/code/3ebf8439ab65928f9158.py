"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _score(c):
    """Compute the largest common radius for centres c (shape (N,3))."""
    if c.shape != (N, 3):
        return -1.0
    if not np.all(np.isfinite(c)):
        return -1.0
    # distance to walls
    wall = np.min(np.min(0.5 - np.abs(c), axis=1), axis=0)
    # pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    return min(wall, min_dist / 2.0)

def _init_grid():
    """Simple cubic grid as starting point."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def _init_random():
    """Random initialisation inside the cube."""
    return np.random.uniform(-0.5, 0.5, size=(N, 3))

# ---------------------------------------------------------------------------
# Smooth (soft-min) objective for L-BFGS-B
# ---------------------------------------------------------------------------
def _softmin_val(c, beta):
    """Value of the soft-min objective (larger = better packing)."""
    c = np.asarray(c, dtype=float)
    # wall distances: min over coords of (0.5 - |c|)
    wall = 0.5 - np.abs(c)          # (N,3)
    w = np.min(wall, axis=1)        # (N,)
    # pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist2 = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist2, np.inf)
    d = np.sqrt(np.maximum(dist2, 0.0))
    # soft-min over the union {wall distances} U {pair distances}
    vals = np.concatenate([w, d.ravel()])
    vals = np.maximum(vals, 1e-12)
    m = np.min(vals)
    return m + np.log(np.sum(np.exp(-beta * (vals - m)))) / beta

def _softmin_grad(c, beta):
    """Gradient of the soft-min objective (ascent direction)."""
    c = np.asarray(c, dtype=float)
    wall = 0.5 - np.abs(c)
    w = np.min(wall, axis=1)

    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist2 = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist2, np.inf)
    d = np.sqrt(np.maximum(dist2, 0.0))

    vals = np.concatenate([w, d.ravel()])
    vals = np.maximum(vals, 1e-12)
    m = np.min(vals)
    ex = np.exp(-beta * (vals - m))
    S = np.sum(ex)
    # d softmin / d vals = -(ex / S)   (softmin = m + log(S)/beta)
    grad_vals = -(ex / S)

    g = np.zeros_like(c)

    # wall part: vals[0:N] = w = min_k (0.5 - |c_i,k|)
    # d w_i / d c_i = -sign(c_i) for the argmin coordinate
    gw = grad_vals[:N]
    argmin_k = np.argmin(wall, axis=1)
    signs = np.sign(c)
    signs[signs == 0] = 1.0
    for i in range(N):
        g[i, argmin_k[i]] += gw[i] * (-signs[i, argmin_k[i]])

    # pair part: vals[N:] = d (flattened, upper triangle order of the full matrix)
    gp = grad_vals[N:].reshape(N, N)
    # d d_ij / d c_i = (c_i - c_j) / d_ij
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            dij = d[i, j]
            if dij > 1e-12:
                g[i] += gp[i, j] * (diff[i, j] / dij)
    return g

def _lfgs(x, beta):
    """Negative objective + gradient for scipy (minimize)."""
    x = np.reshape(x, (N, 3))
    val = _softmin_val(x, beta)
    grad = _softmin_grad(x, beta)
    return -val, -np.reshape(grad, -1)

def _polish(c, time_limit, iters=1200):
    """Pairwise-repulsion + wall local polish (steepest on true min)."""
    c = c.copy()
    start = time.time()
    for it in range(iters):
        if time.time() > time_limit:
            break
        # full pairwise distance matrix
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)

        g = np.zeros_like(c)
        # wall repulsion
        for i in range(N):
            for k in range(3):
                if c[i, k] > 0:
                    wd = 0.5 - c[i, k]
                    g[i, k] -= 1.0 / (wd + 1e-8)
                else:
                    wd = c[i, k] + 0.5
                    g[i, k] += 1.0 / (wd + 1e-8)
        # pairwise repulsion (only nearest neighbours matter but do all for correctness)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                dij = dist[i, j]
                if dij > 1e-10:
                    g[i] += diff[i, j] / (dij**2)
        # step with adaptive learning rate
        gnorm = np.linalg.norm(g)
        if gnorm < 1e-12:
            break
        lr = 0.004
        new_c = c + lr * g
        new_c = np.clip(new_c, -0.4999, 0.4999)
        if _score(new_c) > _score(c) - 1e-12:
            c = new_c
        else:
            # try smaller step
            new_c = c + 0.25 * lr * g
            new_c = np.clip(new_c, -0.4999, 0.4999)
            if _score(new_c) > _score(c) - 1e-12:
                c = new_c
    return c

def construct():
    start_time = time.time()
    deadline = 135.0  # safety margin before 150s kill

    # Candidate starts: grid + a few random
    starts = [_init_grid()]
    for _ in range(4):
        starts.append(_init_random())

    best_c = starts[0].copy()
    best_score = _score(best_c)

    beta = 120.0  # soft-min sharpness
    bounds = [(-0.5, 0.5)] * (N * 3)

    for s in starts:
        if time.time() > deadline - 25.0:
            break
        x0 = np.reshape(s, -1)
        try:
            res = minimize(_lfgs, x0, method='L-BFGS-B',
                           jac=True, bounds=bounds,
                           options={'maxiter': 400, 'maxfun': 20000,
                                    'ftol': 1e-12, 'gtol': 1e-9})
            xopt = np.reshape(res.x, (N, 3))
            xopt = np.clip(xopt, -0.4999, 0.4999)
        except Exception:
            xopt = s.copy()

        # local polish to escape soft-min plateau
        xopt = _polish(xopt, deadline - 5.0, iters=1200)

        sc = _score(xopt)
        if sc > best_score:
            best_score = sc
            best_c = xopt.copy()

    # Final short polish on the best
    best_c = _polish(best_c, deadline - 2.0, iters=600)

    return best_c
# EVOLVE-BLOCK-END
