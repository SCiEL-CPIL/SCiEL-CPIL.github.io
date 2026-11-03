"""Packing 50 equal circles in unit disk.

Strategy (combined):
- Structural idea: ring topology (1 + 6 + 12 + 18 + 13) with one centre,
  giving a natural, low-symmetry breaking of the circular container.
- Refinement machinery: smooth soft-min (log-sum-exp) surrogate of the
  nonsmooth min/max objective, multi-stage L-BFGS-B annealing over the
  full point coordinates, a constraint-projection push pass, and a final
  binary-search polish that drives the feasible radius up to machine
  precision.
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
DEADLINE_MARGIN = 20.0  # seconds of safety before the 150 s hard limit


def _pairwise_distances(pts):
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def _score(pts):
    if pts is None or len(pts) != N:
        return -1.0
    pts = np.asarray(pts, dtype=float)
    if not np.all(np.isfinite(pts)):
        return -1.0
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    min_boundary = np.min(1.0 - np.sqrt(np.sum(pts ** 2, axis=1)))
    val = min(min_pair, min_boundary)
    if val <= 0:
        return -1.0
    return float(val)


def _softmin_logsumexp(values, beta):
    if len(values) == 0:
        return 0.0
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 0.0
    neg = -values * beta
    neg_max = np.max(neg)
    if not np.isfinite(neg_max):
        return 0.0
    exp_sum = np.sum(np.exp(neg - neg_max))
    if exp_sum <= 0.0:
        return 0.0
    return -(neg_max + np.log(exp_sum)) / beta


def _smooth_objective(x, beta):
    pts = x.reshape(N, 2)
    if not np.all(np.isfinite(pts)):
        return 1e10
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    boundary = 1.0 - np.sqrt(np.sum(pts ** 2, axis=1))
    pair_half = d / 2.0
    all_vals = np.concatenate([pair_half.ravel(), boundary])
    all_vals = all_vals[np.isfinite(all_vals)]
    if len(all_vals) == 0:
        return 1e10
    return -_softmin_logsumexp(all_vals, beta)


def _ring_init():
    """Ring topology 1 + 6 + 12 + 18 + 13 with evenly spaced angles."""
    ring_sizes = [1, 6, 12, 18, 13]
    radii = np.array([0.0, 0.27, 0.50, 0.72, 0.90])
    pts = [np.array([0.0, 0.0])]
    for i in range(1, len(ring_sizes)):
        r = radii[i]
        n = ring_sizes[i]
        a = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
        pts.append(np.column_stack([r * np.cos(a), r * np.sin(a)]))
    return np.vstack(pts)


def _hex_init(scale=0.12):
    points = []
    for i in range(-12, 13):
        for j in range(-12, 13):
            x = scale * (i + 0.5 * j)
            y = scale * (np.sqrt(3) / 2 * j)
            if x * x + y * y <= 0.80:
                points.append([x, y])
    points = np.array(points)
    if len(points) == 0:
        return None
    dists = np.sqrt(np.sum(points ** 2, axis=1))
    idx = np.argsort(dists)
    points = points[idx[:N]]
    return points


def _anneal(pts, deadline, max_betas=14):
    current = pts.copy()
    current_score = _score(current)
    betas = [3.0, 8.0, 20.0, 50.0, 120.0, 300.0, 700.0, 1500.0,
             3000.0, 6000.0, 12000.0, 25000.0, 50000.0, 100000.0]
    betas = betas[:max_betas]
    for beta in betas:
        if time.time() > deadline - 6:
            break
        try:
            res = minimize(
                _smooth_objective,
                current.ravel(),
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 1500, 'ftol': 1e-16, 'gtol': 1e-15,
                         'maxfun': 40000}
            )
            new_pts = res.x.reshape(N, 2)
            if not np.all(np.isfinite(new_pts)):
                continue
            s = _score(new_pts)
            if s > current_score:
                current = new_pts
                current_score = s
        except Exception:
            pass
    return current, current_score


def _constraint_push(pts, r_target, deadline, max_iters=600):
    pts = pts.copy()
    for _ in range(max_iters):
        if time.time() > deadline - 4:
            break
        d = _pairwise_distances(pts)
        np.fill_diagonal(d, np.inf)
        pair_margins = d / 2.0
        boundary_margins = 1.0 - np.sqrt(np.sum(pts ** 2, axis=1))
        min_pair = np.min(pair_margins)
        min_bound = np.min(boundary_margins)
        if min_pair < min_bound - 1e-14:
            i, j = np.unravel_index(np.argmin(pair_margins), pair_margins.shape)
            if i == j:
                break
            di = pts[i] - pts[j]
            dist = np.linalg.norm(di)
            if dist < 1e-15:
                break
            di /= dist
            push = (2.0 * r_target - dist) / 2.0
            if push > 1e-15:
                pts[i] += push * di
                pts[j] -= push * di
        else:
            i = int(np.argmin(boundary_margins))
            pi = pts[i]
            dist = np.linalg.norm(pi)
            if dist < 1e-15:
                pts[i] += 1e-4 * np.array([1.0, 0.0])
            else:
                di = pi / dist
                push = (1.0 - r_target - dist)
                if push > 1e-15:
                    pts[i] += push * di
        d = _pairwise_distances(pts)
        np.fill_diagonal(d, np.inf)
        r = min(np.min(d) / 2.0, np.min(1.0 - np.sqrt(np.sum(pts ** 2, axis=1))))
        if r >= r_target - 1e-12:
            break
    return pts


def _binary_search_polish(pts, deadline):
    current = pts.copy()
    low = _score(current) - 1e-8
    high = _score(current) + 2e-4
    for _ in range(45):
        if time.time() > deadline - 2:
            break
        mid = (low + high) / 2.0
        polished = _constraint_push(current, mid, deadline, max_iters=300)
        s = _score(polished)
        if s >= mid - 1e-10:
            low = mid
            current = polished
        else:
            high = mid
    return current, _score(current)


def construct():
    t0 = time.time()
    deadline = t0 + 150.0 - DEADLINE_MARGIN
    best_points = None
    best_score = -1.0

    inits = []
    ring = _ring_init()
    if ring is not None and len(ring) == N:
        inits.append(ring)
    hexp = _hex_init()
    if hexp is not None and len(hexp) == N:
        inits.append(hexp)
    # small perturbations of the ring to escape symmetric saddles
    for k in range(3):
        rng = np.random.default_rng(1000 + k)
        pert = ring + rng.normal(0.0, 0.01, size=ring.shape)
        inits.append(pert)

    for init in inits:
        if time.time() > deadline - 12:
            break
        try:
            ann, s = _anneal(init, deadline, max_betas=10)
            if s > best_score:
                best_score = s
                best_points = ann.copy()
        except Exception:
            pass

    if best_points is None:
        best_points = ring
        best_score = _score(ring)

    # Final high-precision polish on the best candidate.
    try:
        polished, s = _binary_search_polish(best_points, deadline)
        if s > best_score:
            best_score = s
            best_points = polished
    except Exception:
        pass

    return np.asarray(best_points, dtype=float)
