"""
50 equal spheres in a unit cube – maximise common radius.

Strategy (combined):
  1. Fast initialisation: a near-optimal grid / layered seed.
  2. Smooth surrogate optimisation (log-sum-exp soft-min) via L-BFGS-B,
     annealed over temperature – navigates the rugged landscape without
     getting stuck at non-differentiable kinks.
  3. Exact local polish: Nelder-Mead on the true min-radius objective
     with very tight tolerances.
  4. Metropolis Monte Carlo (a few thousand iters) to escape local minima,
     followed by another smooth + exact polish cycle.
  5. Deadline guard: every loop checks elapsed time and returns the best
     configuration found so far with a safety margin.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 50
DEADLINE = 140.0  # safety margin before 150 s kill


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_radius(c):
    """Exact common radius for centres c (N,3) in cube [-0.5,0.5]^3."""
    if c.shape != (N, 3) or not np.all(np.isfinite(c)):
        return -1.0
    if np.any(np.abs(c) > 0.5 + 1e-12):
        return -1.0
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    min_wall = np.min(0.5 - np.max(np.abs(c), axis=1))
    return min(min_pair, min_wall)


def _pair_dist_matrix(c):
    """(N,N) pairwise distance matrix, diagonal = inf."""
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    return d


def _softmin_radius(c, t):
    """Smooth approximation of the true radius via log-sum-exp over all
    constraints (pair/2 and wall)."""
    d = _pair_dist_matrix(c)
    pair = d / 2.0
    wall = 0.5 - np.abs(c)
    vals = np.concatenate([pair.ravel(), wall.ravel()])
    vals = vals[np.isfinite(vals)]
    m = np.min(vals)
    # Numerically stable log-sum-exp
    shifted = vals - m
    s = np.sum(np.exp(-shifted / t))
    return m - t * np.log(s)


def _smooth_obj(x, t):
    """Objective to MINIMISE: negative smooth radius plus boundary penalty."""
    c = x.reshape(N, 3)
    r = _softmin_radius(c, t)
    pen = np.sum(np.maximum(0.0, np.abs(c) - 0.5 + 1e-6) ** 2)
    return -r + 1e3 * pen


def _neg_score(centers):
    """Negative exact score for minimisation."""
    return -_compute_radius(centers)


def _time_left(deadline):
    return deadline - time.time()


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def _make_grid(m, n_use):
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:n_use]


def _initial_configs():
    """Return a list of diverse starting configurations."""
    configs = []

    # 1. Simple cubic grid (5^3 = 125, take first 50)
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    configs.append(_make_grid(m, N).copy())

    # 2. Perturbed grid
    rng = np.random.default_rng(42)
    p = _make_grid(m, N) + rng.uniform(-0.01, 0.01, (N, 3))
    configs.append(np.clip(p, -0.49, 0.49))

    # 3. Layered arrangement: 5 layers × 10 points
    for seed in [11, 22, 33]:
        rng = np.random.default_rng(seed)
        centers = np.zeros((N, 3))
        idx = 0
        for layer_i in range(5):
            z = -0.4 + 0.8 * layer_i / 4.0
            for j in range(10):
                x = -0.4 + 0.8 * (j % 5) / 4.0
                y = -0.4 + 0.8 * (j // 5) / 1.0
                centers[idx] = [x, y, z]
                idx += 1
        centers += rng.uniform(-0.02, 0.02, (N, 3))
        configs.append(np.clip(centers, -0.49, 0.49))

    # 4. Random initialisations
    for seed in [123, 456, 789, 101, 202]:
        rng = np.random.default_rng(seed)
        configs.append(rng.uniform(-0.4, 0.4, (N, 3)))

    return configs


# ---------------------------------------------------------------------------
# Optimisation routines
# ---------------------------------------------------------------------------

def _smooth_polish(c, deadline):
    """Annealed L-BFGS-B on the smooth soft-min surrogate."""
    c = np.clip(c.copy(), -0.49, 0.49)
    best = c.copy()
    best_r = _compute_radius(c)
    x0 = c.flatten()
    bounds = [(-0.49, 0.49)] * (N * 3)

    temperatures = [0.05, 0.02, 0.008, 0.003, 0.001, 0.0005, 0.0001]
    for t in temperatures:
        if _time_left(deadline) < 8.0:
            break
        try:
            res = minimize(
                _smooth_obj, x0, method='L-BFGS-B', bounds=bounds,
                args=(t,),
                options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-11,
                         'maxls': 60},
            )
            c = np.clip(res.x.reshape(N, 3), -0.49, 0.49)
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best = c.copy()
            x0 = c.flatten()
        except Exception:
            pass
    return best, best_r


def _exact_polish(c, deadline):
    """Nelder-Mead on the exact min-radius objective, very tight."""
    c = np.clip(c.copy(), -0.49, 0.49)
    best = c.copy()
    best_r = _compute_radius(c)

    try:
        res = minimize(
            _neg_score, c, method='Nelder-Mead',
            options={'maxiter': 8000, 'xatol': 1e-12, 'fatol': 1e-14,
                     'adaptive': True},
        )
        c = np.clip(res.x, -0.49, 0.49)
        r = _compute_radius(c)
        if r > best_r:
            best_r = r
            best = c
    except Exception:
        pass

    # Second pass with even tighter tolerance
    if _time_left(deadline) > 10.0:
        try:
            res = minimize(
                _neg_score, best, method='Nelder-Mead',
                options={'maxiter': 6000, 'xatol': 1e-13, 'fatol': 1e-15,
                         'adaptive': True},
            )
            c = np.clip(res.x, -0.49, 0.49)
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best = c
        except Exception:
            pass

    return best, best_r


def _metropolis_mc(c, deadline, iters=3000):
    """Metropolis Monte Carlo to escape local minima."""
    c = np.clip(c.copy(), -0.499, 0.499)
    best_r = _compute_radius(c)
    best_c = c.copy()
    rng = np.random.default_rng(123)

    T0 = 0.003
    T_min = 0.0002

    for it in range(iters):
        if _time_left(deadline) < 8.0:
            break

        frac = it / max(iters - 1, 1)
        T = T0 * (T_min / T0) ** frac

        # Random displacement
        trial = c + rng.normal(0, 0.01 * (1.0 - 0.5 * frac), (N, 3))
        trial = np.clip(trial, -0.499, 0.499)

        r_trial = _compute_radius(trial)
        r_cur = _compute_radius(c)

        if r_trial >= r_cur or rng.random() < np.exp((r_trial - r_cur) / T):
            c = trial
            if r_trial > best_r:
                best_r = r_trial
                best_c = c.copy()

    return best_c, best_r


def _repulsion_step(c, deadline, steps=2000):
    """Vectorised repulsion + wall-attraction gradient descent (fast)."""
    c = np.clip(c.copy(), -0.49, 0.49)
    for it in range(steps):
        if _time_left(deadline) < 8.0:
            break
        d = _pair_dist_matrix(c)
        inv = 1.0 / np.maximum(d, 1e-6) ** 2
        diff = c[:, None, :] - c[None, :, :]
        F = np.sum(diff * inv[:, :, None], axis=1)

        wall_force = np.zeros_like(c)
        for k in range(3):
            sgn = np.sign(c[:, k])
            sgn[sgn == 0] = 1.0
            wall_force[:, k] = sgn * (0.5 - np.abs(c[:, k]))

        frac = it / max(steps - 1, 1)
        rep_scale = 1e-3 * (1.0 - 0.7 * frac)
        wall_scale = 1e-3 * (0.3 + 1.0 * frac)
        lr = 1e-2 * (1.0 - 0.5 * frac)
        c = c + lr * (rep_scale * F + wall_scale * wall_force)
        c = np.clip(c, -0.49, 0.49)
    return c


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def construct():
    start = time.time()
    deadline = start + DEADLINE

    best_centers = None
    best_score = -1.0

    def _update(c):
        nonlocal best_centers, best_score
        r = _compute_radius(c)
        if r > best_score:
            best_score = r
            best_centers = c.copy()

    # --- Phase 0: quick repulsion from each seed to get a reasonable start ---
    seeds = _initial_configs()
    for i, s in enumerate(seeds):
        if _time_left(deadline) < 15.0:
            break
        c = _repulsion_step(s, deadline, steps=800)
        _update(c)

    # --- Phase 1: smooth surrogate + exact polish on top seeds ---
    # Rank seeds by current best score after repulsion; take top 4
    seed_scores = []
    for s in seeds:
        c = _repulsion_step(s, deadline, steps=400)
        seed_scores.append((_compute_radius(c), c))
    seed_scores.sort(key=lambda x: -x[0])

    for i, (_, c) in enumerate(seed_scores[:4]):
        if _time_left(deadline) < 20.0:
            break
        _update(c)
        c2, r2 = _smooth_polish(c, deadline)
        _update(c2)
        c3, r3 = _exact_polish(c2, deadline)
        _update(c3)

    # --- Phase 2: Metropolis MC on the current best, then re-polish ---
    if _time_left(deadline) > 30.0 and best_centers is not None:
        c_mc, r_mc = _metropolis_mc(best_centers, deadline, iters=3000)
        _update(c_mc)
        if _time_left(deadline) > 15.0:
            c4, r4 = _smooth_polish(c_mc, deadline)
            _update(c4)
            c5, r5 = _exact_polish(c4, deadline)
            _update(c5)

    # --- Phase 3: final high-precision polish ---
    if _time_left(deadline) > 8.0 and best_centers is not None:
        c6, r6 = _exact_polish(best_centers, deadline)
        _update(c6)

    # Safety: ensure we return a valid array
    if best_centers is None:
        best_centers = _make_grid(5, N)

    return np.clip(best_centers, -0.5, 0.5)
