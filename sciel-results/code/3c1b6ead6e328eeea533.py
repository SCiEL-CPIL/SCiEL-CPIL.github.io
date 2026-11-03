"""Packing 50 equal circles in a unit square: maximize the common radius.

Design family: smooth soft-min (log-sum-exp) surrogate for the minimum-radius
objective, optimized with L-BFGS-B under box constraints, with a progressive
beta (sharpness) schedule and a final high-precision polish pass.  Multiple
structured starts are tried; the best result is returned before the deadline.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
HALF = 0.5
DEADLINE = 140.0          # wall-clock safety margin (evaluator kills at 150s)


# ---------------------------------------------------------------------------
# Objective helpers
# ---------------------------------------------------------------------------
def _radius(pts):
    """Exact minimum radius for a point set (used for scoring / polish)."""
    pts = np.asarray(pts, dtype=float)
    n = pts.shape[0]
    # pairwise distances
    d = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = dist.min() / 2.0
    # distance to each of the four walls
    wall = np.minimum(np.minimum(HALF - np.abs(pts[:, 0]),
                                  HALF - np.abs(pts[:, 1])),
                       np.minimum(HALF - np.abs(pts[:, 0]),
                                  HALF - np.abs(pts[:, 1])))
    # simpler: per-point min wall distance
    wall = np.minimum(HALF - np.abs(pts[:, 0]), HALF - np.abs(pts[:, 1]))
    min_wall = wall.min()
    return min(min_pair, min_wall)


def _soft_min_radius_neg(pts, beta):
    """Negative of a soft-min surrogate of the radius (to be minimized).

    Uses log-sum-exp for numerical stability.  The surrogate
    r_soft = -logsumexp(-beta * s_i) / beta  where s_i are the relevant
    separation distances (pair distances and wall distances).  Minimizing
    -r_soft is equivalent to maximizing the soft-min radius.
    """
    pts = np.asarray(pts, dtype=float)
    n = pts.shape[0]

    # pair distances (only off-diagonal, symmetric)
    d = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    # exclude self
    np.fill_diagonal(dist, np.inf)
    # each unordered pair counted once: take upper triangle
    iu = np.triu_indices(n, k=1)
    pair = dist[iu] / 2.0

    # wall distances per point
    wall = np.minimum(HALF - np.abs(pts[:, 0]), HALF - np.abs(pts[:, 1]))

    # combine all separations into one vector
    s = np.concatenate([pair, wall])
    # clip for stability (avoid log of 0 / overflow in exp)
    s = np.clip(s, 1e-12, None)

    # soft min via log-sum-exp:  m = -logsumexp(-beta*s)/beta
    x = -beta * s
    x = x - x.max()
    logsum = x.max() + np.log(np.sum(np.exp(x)))
    r_soft = -logsum / beta
    return -r_soft


def _soft_min_grad(pts, beta):
    """Gradient of _soft_min_radius_neg w.r.t. pts (central-difference-free)."""
    pts = np.asarray(pts, dtype=float)
    n = pts.shape[0]

    d = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    iu = np.triu_indices(n, k=1)
    pair = dist[iu] / 2.0
    wall = np.minimum(HALF - np.abs(pts[:, 0]), HALF - np.abs(pts[:, 1]))
    s = np.concatenate([np.clip(pair, 1e-12, None), np.clip(wall, 1e-12, None)])

    x = -beta * s
    x = x - x.max()
    w = np.exp(x)
    w = w / w.sum()          # softmax weights over separations

    grad = np.zeros_like(pts)

    # pair separations:  s_k = |p_i - p_j|/2,  d s_k / d p_i = (p_i - p_j)/|p_i-p_j|
    # weight w_k * beta * (contribution to -r_soft)  ->  d(-r_soft)/d p = beta * sum_k w_k * ds_k/dp
    # careful with sign:  r_soft = -logsumexp(-beta s)/beta
    #   d r_soft / d s_k = w_k   (softmax weight, since d/ds_k of -logsumexp(-beta s)/beta = w_k)
    #   so d(-r_soft)/d p = -beta * sum_k w_k * (d s_k / d p)
    sign = -beta

    # pairs
    for k in range(len(iu[0])):
        i, j = iu[0][k], iu[1][k]
        wkt = w[k]
        if wkt <= 1e-15:
            continue
        diff = pts[i] - pts[j]
        norm = dist[i, j]
        if norm < 1e-12:
            continue
        unit = diff / norm
        ds_di = unit / 2.0
        ds_dj = -unit / 2.0
        grad[i] += sign * wkt * ds_di
        grad[j] += sign * wkt * ds_dj

    # walls: s = min(HALF - |x|, HALF - |y|)  -> use the active wall
    for i in range(n):
        wx = HALF - np.abs(pts[i, 0])
        wy = HALF - np.abs(pts[i, 1])
        if wx <= wy:
            # active wall in x
            ds_dx = -np.sign(pts[i, 0])
            ds_dy = 0.0
        else:
            ds_dx = 0.0
            ds_dy = -np.sign(pts[i, 1])
        # wall weight index offset
        wkt = w[len(iu[0]) + i]
        if wkt > 1e-15:
            grad[i, 0] += sign * wkt * ds_dx
            grad[i, 1] += sign * wkt * ds_dy

    return grad


# ---------------------------------------------------------------------------
# Initializations
# ---------------------------------------------------------------------------
def _init_grid(perturb=0.0, seed=0):
    rng = np.random.default_rng(seed)
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    pts = grid[:N].copy()
    if perturb > 0:
        pts += rng.normal(0, perturb, size=pts.shape)
        pts = np.clip(pts, -HALF + 1e-3, HALF - 1e-3)
    return pts


def _init_spiral(seed=0):
    """Golden-angle phyllotaxis spiral scaled to the square."""
    theta = np.arange(N) * 2.399963229728653
    r = np.sqrt(np.arange(N) / N) * 0.48
    pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    return pts


def _init_jitter_grid(seed=0):
    rng = np.random.default_rng(seed)
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    pts = grid[:N].copy()
    pts += rng.normal(0, 0.02, size=pts.shape)
    pts = np.clip(pts, -HALF + 1e-3, HALF - 1e-3)
    return pts


def _init_rows(seed=0):
    """Row-based layout: distribute into rows of ~10 with slight stagger."""
    rng = np.random.default_rng(seed)
    rows = 5
    cols = 10
    pts = []
    for i in range(rows):
        y = -HALF + (i + 0.5) * (1.0 / rows)
        for j in range(cols):
            x = -HALF + (j + 0.5) * (1.0 / cols)
            pts.append([x, y])
    pts = np.array(pts[:N])
    pts += rng.normal(0, 0.01, size=pts.shape)
    pts = np.clip(pts, -HALF + 1e-3, HALF - 1e-3)
    return pts


# ---------------------------------------------------------------------------
# Optimization driver
# ---------------------------------------------------------------------------
def _optimize_start(pts0, deadline, start_beta=8.0, end_beta=2000.0,
                    n_stages=6, ftol=1e-12, gtol=1e-10):
    """Run progressive-beta L-BFGS-B from a starting point.

    Returns the best (radius, points) found across the stages.
    """
    bounds = [(-HALF, HALF)] * (2 * N)
    best_pts = pts0.copy()
    best_r = _radius(pts0)

    # geometric progression of beta from start to end
    if n_stages < 1:
        n_stages = 1
    betas = start_beta * (end_beta / start_beta) ** (np.arange(n_stages) / max(1, n_stages - 1))

    for beta in betas:
        if time.time() > deadline:
            break
        try:
            res = minimize(
                _soft_min_radius_neg,
                best_pts,
                args=(beta,),
                jac=_soft_min_grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 400, 'ftol': ftol, 'gtol': gtol,
                         'maxls': 40, 'maxfun': 5000},
            )
            cand = res.x
            r = _radius(cand)
            if r > best_r:
                best_r = r
                best_pts = cand.copy()
        except Exception:
            pass

    return best_pts, best_r


def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE

    # candidate initializers
    inits = []
    inits.append(_init_grid(perturb=0.0, seed=0))
    inits.append(_init_jitter_grid(seed=1))
    inits.append(_init_rows(seed=2))
    inits.append(_init_spiral(seed=3))
    inits.append(_init_grid(perturb=0.015, seed=4))
    inits.append(_init_grid(perturb=0.03, seed=5))
    inits.append(_init_jitter_grid(seed=6))
    inits.append(_init_rows(seed=7))

    best_pts = inits[0].copy()
    best_r = _radius(best_pts)

    # ---- Stage 1: quick passes on all inits (coarse beta) ----
    for k, p0 in enumerate(inits):
        if time.time() > deadline:
            break
        # leave room: if many inits remain, use fewer stages
        remaining = len(inits) - k
        if remaining > 3:
            n_stages = 3
            end_beta = 200.0
        else:
            n_stages = 5
            end_beta = 1000.0
        try:
            pts, r = _optimize_start(
                p0, deadline,
                start_beta=8.0, end_beta=end_beta, n_stages=n_stages,
                ftol=1e-10, gtol=1e-8,
            )
            if r > best_r:
                best_r = r
                best_pts = pts.copy()
        except Exception:
            pass

    # ---- Stage 2: deep polish on the best result (high beta, tight tol) ----
    if time.time() < deadline:
        try:
            pts, r = _optimize_start(
                best_pts, deadline,
                start_beta=50.0, end_beta=4000.0, n_stages=5,
                ftol=1e-13, gtol=1e-11,
            )
            if r > best_r:
                best_r = r
                best_pts = pts.copy()
        except Exception:
            pass

    # ---- Final exact local polish: tiny perturbation search around best ----
    if time.time() < deadline - 5.0:
        rng = np.random.default_rng(42)
        for _ in range(3):
            if time.time() > deadline - 3.0:
                break
            jittered = best_pts + rng.normal(0, 0.002, size=best_pts.shape)
            jittered = np.clip(jittered, -HALF + 1e-4, HALF - 1e-4)
            try:
                pts, r = _optimize_start(
                    jittered, deadline,
                    start_beta=200.0, end_beta=5000.0, n_stages=3,
                    ftol=1e-13, gtol=1e-11,
                )
                if r > best_r:
                    best_r = r
                    best_pts = pts.copy()
            except Exception:
                pass

    # Ensure validity: clip to interior and finite
    best_pts = np.clip(np.asarray(best_pts, dtype=float), -HALF + 1e-9, HALF - 1e-9)
    if not np.all(np.isfinite(best_pts)):
        best_pts = _init_grid(perturb=0.0, seed=0)
    return best_pts
