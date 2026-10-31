"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii.

Structural idea (current): differential evolution over [aspect, x1..yN].
Refinement machinery (inspiration): vectorized per-circle max-radius objective
that allows different radii, plus a fast vectorized local refinement.
"""
import numpy as np
import time
import random

N = 21
DEADLINE = 142.0  # safety margin before 150s kill

# Precompute pairwise index pairs once (i<j)
_PAIRS = np.array([(i, j) for i in range(N) for j in range(i + 1, N)], dtype=int)
_PI = _PAIRS[:, 0]
_PJ = _PAIRS[:, 1]


def _max_radii(centers, w, h):
    """Vectorized: max feasible radius for each circle given centers in [0,w]x[0,h]."""
    r = np.minimum(
        np.minimum(centers[:, 0], w - centers[:, 0]),
        np.minimum(centers[:, 1], h - centers[:, 1]),
    )
    # pairwise distances, halved
    d = np.sqrt(
        (centers[_PI, 0] - centers[_PJ, 0]) ** 2 +
        (centers[_PI, 1] - centers[_PJ, 1]) ** 2
    )
    half = d / 2.0
    for i in range(N):
        # distances from circle i to all others
        mask = (_PI == i) | (_PJ == i)
        others = half[mask]
        if others.size:
            r[i] = min(r[i], others.min())
    return np.maximum(r, 0.0)


def _objective(x):
    """Negative sum of max feasible radii."""
    w = x[0]
    h = 2.0 - w
    if w < 0.2 or h < 0.2:
        return 1e6
    centers = x[1:].reshape(N, 2)
    # clamp
    centers[:, 0] = np.clip(centers[:, 0], 1e-4, w - 1e-4)
    centers[:, 1] = np.clip(centers[:, 1], 1e-4, h - 1e-4)
    r = _max_radii(centers, w, h)
    return -float(np.sum(r))


def _differential_evolution(objective, bounds, pop_size=30, max_iter=600,
                            mutation=(0.5, 1.0), crossover_prob=0.7,
                            time_limit=100.0):
    n_dims = len(bounds)
    start = time.time()
    bmin = np.array([b[0] for b in bounds])
    bmax = np.array([b[1] for b in bounds])

    pop = bmin + (bmax - bmin) * np.random.rand(pop_size, n_dims)
    fitness = np.array([objective(ind) for ind in pop])

    best_idx = int(np.argmin(fitness))
    best_x = pop[best_idx].copy()
    best_f = fitness[best_idx]

    for gen in range(max_iter):
        if time.time() - start > time_limit:
            break
        for i in range(pop_size):
            r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
            F = random.uniform(*mutation)
            mutant = pop[r1] + F * (pop[r2] - pop[r3])
            mutant = np.clip(mutant, bmin, bmax)
            trial = pop[i].copy()
            mask = np.random.rand(n_dims) < crossover_prob
            trial[mask] = mutant[mask]
            f_trial = objective(trial)
            if f_trial <= fitness[i]:
                pop[i] = trial
                fitness[i] = f_trial
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = trial.copy()
    return best_x, best_f


def _local_refine(centers, w, h, max_iters=400, step_size=0.02):
    """Coordinate-descent local refinement on sum of max radii."""
    centers = centers.copy()
    cur = _max_radii(centers, w, h)
    cur_score = float(np.sum(cur))

    for it in range(max_iters):
        improved = False
        for i in range(N):
            for axis in (0, 1):
                base = centers[i, axis]
                hi = w if axis == 0 else h
                for s in (-step_size, step_size):
                    nx = base + s
                    if nx < 1e-4 or nx > hi - 1e-4:
                        continue
                    centers[i, axis] = nx
                    r = _max_radii(centers, w, h)
                    sc = float(np.sum(r))
                    if sc > cur_score + 1e-11:
                        cur = r
                        cur_score = sc
                        improved = True
                    else:
                        centers[i, axis] = base
        if not improved:
            step_size *= 0.5
            if step_size < 1e-5:
                break
    return centers, cur, cur_score


def _build_result(centers, w, h, radii):
    """Emit (x, y, r) with a tiny shrink for strict disjointness."""
    # recompute feasible radii and shrink slightly
    r = np.minimum(radii, _max_radii(centers, w, h))
    r = r * (1.0 - 1e-6)
    out = np.zeros((N, 3))
    out[:, 0] = centers[:, 0]
    out[:, 1] = centers[:, 1]
    out[:, 2] = r
    return out


def _grid_fallback():
    """Simple valid packing: 5x5 grid in a square, drop 4 corners."""
    w = h = 1.0
    side = 5
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < N:
                cells.append(((i + 0.5) / side, (j + 0.5) / side))
    r = (0.5 / side) * (1.0 - 1e-4)
    out = np.zeros((N, 3))
    for i, (cx, cy) in enumerate(cells):
        out[i, 0] = cx * w
        out[i, 1] = cy * h
        out[i, 2] = r
    return out


def construct():
    start_time = time.time()

    # Bounds: aspect (w) in [0.4, 1.6], centers in loose [0, 2] box
    bounds = [(0.4, 1.6)]
    for _ in range(N * 2):
        bounds.append((0.0, 2.0))

    best_result = None
    best_score = -np.inf

    def _consider(centers, w, h):
        nonlocal best_result, best_score
        centers = centers.copy()
        centers[:, 0] = np.clip(centers[:, 0], 1e-4, w - 1e-4)
        centers[:, 1] = np.clip(centers[:, 1], 1e-4, h - 1e-4)
        r = _max_radii(centers, w, h)
        sc = float(np.sum(r))
        if sc > best_score:
            best_score = sc
            best_result = _build_result(centers, w, h, r)

    # --- Differential evolution (main structural search) ---
    de_time = max(5.0, DEADLINE - time.time() - 25.0)
    if de_time > 10:
        try:
            best_x, best_f = _differential_evolution(
                _objective, bounds,
                pop_size=30, max_iter=800,
                mutation=(0.5, 1.0), crossover_prob=0.7,
                time_limit=de_time,
            )
            w = best_x[0]
            h = 2.0 - w
            centers = best_x[1:].reshape(N, 2)
            _consider(centers, w, h)
        except Exception:
            pass

    # --- Local refinement on the best found so far ---
    if best_result is not None and time.time() < DEADLINE - 5.0:
        # Recover centers/w/h from best_result
        w = float(np.max(best_result[:, 0])) + 1e-9
        # better: reconstruct from stored aspect? we didn't store it; recompute box
        # Use bounding approach: set w,h from the rectangle that fits centers.
        # Simpler: re-run a short refinement using the centers we have.
        cx = best_result[:, 0]
        cy = best_result[:, 1]
        # infer w,h as the rectangle: perimeter 4, choose aspect from extent
        ext_w = float(np.max(cx)) + float(np.max(best_result[:, 2]))
        ext_h = float(np.max(cy)) + float(np.max(best_result[:, 2]))
        # normalize to perimeter 4
        s = 4.0 / (2.0 * (ext_w + ext_h))
        w = ext_w * s
        h = ext_h * s
        centers = np.column_stack([cx, cy])
        try:
            centers2, r2, sc2 = _local_refine(centers, w, h,
                                              max_iters=300, step_size=0.02)
            _consider(centers2, w, h)
        except Exception:
            pass

    # --- Fallback ---
    if best_result is None:
        best_result = _grid_fallback()

    return best_result
