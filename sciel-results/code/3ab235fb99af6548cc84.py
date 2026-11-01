"""Packing of 50 equal circles in the unit square [-1/2,1/2]^2 via annealing."""
import numpy as np
import time
from scipy.spatial import cKDTree

N = 50
HALF = 0.5
TARGET = 0.071377103865


def _score(c):
    """Largest common radius for centres c (N,2)."""
    if c.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    d = cKDTree(c)
    dd = d.query_pairs(1.0)
    if not dd:
        mind = np.inf
    else:
        mind = min(np.linalg.norm(c[i] - c[j]) for i, j in dd)
    if not np.isfinite(mind):
        mind = 0.0
    r = min(mind / 2.0, np.min(HALF - np.abs(c)))
    return float(r) if np.isfinite(r) else 0.0


def _init_grid():
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    pts = np.array([(x, y) for y in ticks for x in ticks])[:N]
    return pts + np.random.randn(N, 2) * 0.01


def _init_hex():
    rows = []
    x0 = 0.05
    dx = 0.13
    dy = 0.11
    for i in range(8):
        off = 0.065 if i % 2 else 0.0
        for j in range(8):
            rows.append((x0 + j * dx + off, -0.45 + i * dy))
    pts = np.array(rows[:N])
    return pts + np.random.randn(N, 2) * 0.01


def _init_random():
    return np.random.uniform(-HALF + 0.05, HALF - 0.05, size=(N, 2))


def _anneal(c0, t_end, start, deadline):
    c = c0.copy()
    r = _score(c)
    best = c.copy()
    best_r = r
    step = 0.02
    step_min = 1e-5
    i = 0
    n = len(t_end)
    while i < n:
        if time.time() >= deadline:
            break
        t = t_end[i]
        # a few moves per temperature step
        for _ in range(20):
            if time.time() >= deadline:
                break
            idx = np.random.randint(N)
            delta = np.random.randn(2) * step
            c[idx] += delta
            c[idx] = np.clip(c[idx], -HALF + 1e-9, HALF - 1e-9)
            r_new = _score(c)
            if r_new >= r or np.random.rand() < np.exp((r_new - r) / max(t, 1e-12)):
                r = r_new
                if r_new > best_r:
                    best_r = r_new
                    best = c.copy()
            else:
                c[idx] -= delta
        i += 1
        step = max(step * 0.98, step_min)
    return best, best_r


def construct():
    start = time.time()
    deadline = start + 145.0
    rng = np.random.default_rng(0)
    np.random.seed(0)

    inits = [_init_grid(), _init_hex(), _init_random()]

    best = None
    best_r = 0.0

    # coarse temperature schedule per restart
    t0 = 0.003
    for it in range(6):
        if time.time() >= deadline:
            break
        c0 = inits[it % len(inits)]
        # shrink target temperature each outer iteration
        t_end = t0 * (0.5 ** it)
        b, br = _anneal(c0, t_end, start, deadline)
        if br > best_r:
            best_r = br
            best = b

    # final local polish on the best: very small steps, deterministic-ish
    if best is not None:
        c = best.copy()
        r = best_r
        step = 2e-4
        for _ in range(4000):
            if time.time() >= deadline:
                break
            idx = np.random.randint(N)
            delta = np.random.randn(2) * step
            c[idx] += delta
            c[idx] = np.clip(c[idx], -HALF + 1e-9, HALF - 1e-9)
            r_new = _score(c)
            if r_new >= r:
                r = r_new
                if r_new > best_r:
                    best_r = r_new
                    best = c.copy()
            else:
                c[idx] -= delta

    if best is None:
        best = _init_grid()
    return best
