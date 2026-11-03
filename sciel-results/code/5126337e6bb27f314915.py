"""Spherical packing: stochastic local search + simulated annealing + polish."""
import numpy as np
from scipy.optimize import minimize
import time

N = 100
HALF = 0.5

# --- scoring helpers -------------------------------------------------------
def _score(c):
    """Largest common radius (min pairwise dist/2, min wall clearance)."""
    c = np.asarray(c, dtype=np.float64)
    # validity guard
    if c.shape != (N, 3) or not np.all(np.isfinite(c)):
        return 0.0
    if np.any(c < -HALF - 1e-12) or np.any(c > HALF + 1e-12):
        return 0.0
    d = c[:, None, :] - c[None, :, :]
    dist = np.sqrt((d * d).sum(-1))
    np.fill_diagonal(dist, np.inf)
    min_pair = dist.min()
    wall = (HALF - np.abs(c)).min()
    return min(min_pair * 0.5, wall)


def _energy(c, k=12.0):
    """Smooth surrogate: large when min radius is small.  E = -softmin."""
    c = np.asarray(c, dtype=np.float64)
    d = c[:, None, :] - c[None, :, :]
    dist = np.sqrt((d * d).sum(-1))
    np.fill_diagonal(dist, np.inf)
    # soft-min over pairwise distances
    m = dist.min()
    # shift for stability
    shifted = dist - m
    soft = m - np.logaddexp.reduce(np.log(np.exp(-k * shifted))) / k
    # wall terms
    wall = (HALF - np.abs(c))
    wm = wall.min()
    wshift = wall - wm
    softwall = wm - np.logaddexp.reduce(np.log(np.exp(-k * wshift))) / k
    return -(min(soft, softwall))


# --- simulated annealing ---------------------------------------------------
def _anneal(start, rng, t_end, iters):
    c = start.copy()
    cur = _score(c)
    best = c.copy()
    best_s = cur
    t0 = time.time()
    n = N
    for it in range(iters):
        if time.time() - t0 > t_end:
            break
        # temperature schedule (geometric)
        T = 1e-5 * (0.9995 ** it)
        # pick a random sphere
        i = rng.integers(n)
        # random small move
        step = 0.02 * (1.0 + 0.3 * np.sin(it * 0.001))
        delta = rng.normal(0, step, size=3)
        old = c[i].copy()
        c[i] = old + delta
        # clamp inside cube (with small margin)
        c[i] = np.clip(c[i], -HALF + 1e-9, HALF - 1e-9)
        new = _score(c)
        if new >= cur or rng.random() < np.exp((new - cur) / max(T, 1e-12)):
            cur = new
            if new > best_s:
                best_s = new
                best = c.copy()
        else:
            c[i] = old
    return best, best_s


# --- polish with L-BFGS-B on a smooth objective ---------------------------
def _polish(c0, rng, t_end):
    """Maximize min radius via minimizing a smooth surrogate."""
    c = c0.copy()
    best = c.copy()
    best_s = _score(c)
    t0 = time.time()
    for restart in range(3):
        if time.time() - t0 > t_end:
            break
        # add tiny perturbation for restarts
        if restart > 0:
            c = best + rng.normal(0, 1e-4, size=c.shape)
            c = np.clip(c, -HALF + 1e-9, HALF - 1e-9)

        def obj(x):
            x = x.reshape(N, 3)
            return _energy(x, k=20.0)

        res = minimize(obj, c.ravel(), method='L-BFGS-B',
                       options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
        cand = np.clip(res.x.reshape(N, 3), -HALF + 1e-9, HALF - 1e-9)
        s = _score(cand)
        if s > best_s:
            best_s = s
            best = cand.copy()
    return best, best_s


def construct():
    t_start = time.time()
    deadline = t_start + 110.0  # safety margin vs 150s kill
    rng = np.random.default_rng(12345)

    # --- initialisation: simple grid ---
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    start = grid[:N].copy()

    best = start.copy()
    best_s = _score(best)

    # --- multi-start annealing ---
    n_starts = 6
    for s in range(n_starts):
        if time.time() > deadline - 30.0:
            break
        # perturb start
        if s == 0:
            cand0 = start.copy()
        else:
            cand0 = start + rng.normal(0, 0.05, size=start.shape)
            cand0 = np.clip(cand0, -HALF + 1e-9, HALF - 1e-9)
        # anneal
        c, sc = _anneal(cand0, rng, deadline - 30.0, iters=20000)
        if sc > best_s:
            best_s = sc
            best = c.copy()

    # --- polish the best ---
    c, sc = _polish(best, rng, deadline - 5.0)
    if sc > best_s:
        best_s = sc
        best = c.copy()

    # final validity clamp
    best = np.clip(best, -HALF + 1e-9, HALF - 1e-9)
    return best
