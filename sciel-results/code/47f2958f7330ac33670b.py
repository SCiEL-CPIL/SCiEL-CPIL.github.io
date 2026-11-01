"""Circle packing in a unit disk: soft-min L-BFGS-B + binary-search feasibility + high-beta polish."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
DEADLINE_SECONDS = 140.0  # safety margin before 150s kill


def _time_left():
    return DEADLINE_SECONDS - (time.time() - _START)


_START = time.time()


def _score(points):
    """Exact score: min(half min pairwise dist, min margin to boundary)."""
    if points.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    d2 = np.sum(points * points, axis=1)
    if np.any(d2 > 1.0 + 1e-12):
        return 0.0
    diff = points[:, None, :] - points[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    min_margin = 1.0 - np.max(np.sqrt(d2))
    r = min(min_pair / 2.0, min_margin)
    return r if r > 0 else 0.0


def _soft_min(x, beta):
    """Smooth minimum: -log(sum exp(-beta*x))/beta. Gradients handled analytically."""
    xb = beta * x
    # shift for stability
    xb = xb - np.min(xb)
    s = np.sum(np.exp(-xb))
    return -np.log(s) / beta


def _objective_and_grad(points_flat, beta):
    """Returns (value, grad) of soft-min objective. Lower = better (we want to maximize min)."""
    pts = points_flat.reshape(N, 2)
    # Pairwise squared distances
    diff = pts[:, None, :] - pts[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, np.inf)
    # Only upper triangle to avoid double counting
    iu = np.triu_indices(N, k=1)
    d2u = d2[iu]
    # distances
    dists = np.sqrt(d2u)
    # Boundary margins
    d2c = np.sum(pts * pts, axis=1)
    margins = 1.0 - np.sqrt(d2c)

    # Candidate set: pairwise distances (each contributes to min_pair/2 via dist) and margins
    # We want to maximize  min( min_pair/2, min_margin )
    # = maximize min( dists/2, margins )
    # So soft-min over  [dists/2, margins]
    vals = np.concatenate([dists / 2.0, margins])

    # soft min
    xb = beta * vals
    xb = xb - np.min(xb)
    exp_neg = np.exp(-xb)
    s = np.sum(exp_neg)
    val = -np.log(s) / beta
    # weights w_k = exp(-beta x_k)/sum
    w = exp_neg / s

    # gradient of soft_min w.r.t. vals = -w  (since d/dx_k [-log(sum exp(-beta x))/beta] = -exp(-beta x_k)/sum)
    # Actually: d/dx_k softmin = -w_k
    # Now chain rule to points.
    grad_flat = np.zeros_like(points_flat)

    # contributions from pairwise dists/2
    # vals[0:npairs] = dists/2 ; d(dists/2)/d(pts) 
    w_pair = w[:len(iu)]
    # d dist / d pts_i = (pts_i - pts_j)/dist ; d (dist/2)/d pts_i = (pts_i-pts_j)/(2*dist)
    # accumulate
    for idx in range(len(iu)):
        i, j = iu[0][idx], iu[1][idx]
        dij = diff[i, j]
        d = dists[idx]
        if d < 1e-12:
            continue
        # d (dist/2) / d pts_i = dij/(2d)
        # d (dist/2) / d pts_j = -dij/(2d)
        coef = w_pair[idx] / (2.0 * d)
        # grad of softmin = -w * d(vals)/d(pts)
        grad_flat[2 * i:2 * i + 2] += -coef * dij
        grad_flat[2 * j:2 * j + 2] += coef * dij

    # contributions from margins
    w_marg = w[len(iu):]
    # d margin / d pts_i = -pts_i/|pts_i|
    for i in range(N):
        r_i = np.sqrt(d2c[i])
        if r_i < 1e-12:
            continue
        coef = w_marg[i] / r_i
        grad_flat[2 * i:2 * i + 2] += -coef * pts[i]

    return val, grad_flat


def _optimize_lbfgs(points, beta, max_iter=200):
    """Run L-BFGS-B to minimize soft-min objective (maximize min)."""
    def fun_and_grad(x):
        v, g = _objective_and_grad(x, beta)
        return v, g
    res = minimize(
        fun_and_grad,
        points.flatten(),
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-8},
    )
    return res.x.reshape(N, 2)


def _get_hex_lattice():
    points = []
    spacing = 0.14
    while len(points) < N:
        k = int(1.0 / spacing) + 2
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x * x + y * y <= (1.0 - spacing / 2.0) ** 2:
                    points.append([x, y])
        if len(points) >= N:
            break
        spacing *= 0.95
    points = np.array(points)
    radii = np.sqrt(np.sum(points * points, axis=1))
    idx = np.argsort(radii)
    return points[idx[:N]]


def _get_ring_init():
    points = [[0.0, 0.0]]
    for ring in range(1, 9):
        r = ring * 0.12
        n_pts = min(6 * ring, N - len(points))
        if n_pts <= 0:
            break
        angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        for a in angles:
            points.append([r * np.cos(a), r * np.sin(a)])
    return np.array(points[:N])


def _get_random_init(seed):
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0, 2 * np.pi, N)
    radii = rng.uniform(0, 1.0, N)
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _polish(points, beta, max_iter=300):
    """Polish with a given beta."""
    try:
        return _optimize_lbfgs(points, beta, max_iter=max_iter)
    except Exception:
        return points


def _high_beta_polish(points):
    """Final polish with very high beta to sharpen min."""
    for beta in [50.0, 100.0, 200.0]:
        if _time_left() < 3.0:
            break
        pts = _polish(points, beta, max_iter=200)
        s = _score(pts)
        if s > _score(points):
            points = pts
    return points


def _feasibility_search(points, r_lo, r_hi, beta, iters=12):
    """Binary search on radius: check feasibility of min >= r."""
    def feasible(r):
        # objective: soft-min of (dists/2 - r, margins - r) ; we want all >= 0
        # minimize max(0, ...)  -> use soft-min of negatives
        def obj(x):
            pts = x.reshape(N, 2)
            diff = pts[:, None, :] - pts[None, :, :]
            d2 = np.sum(diff * diff, axis=2)
            np.fill_diagonal(d2, np.inf)
            iu = np.triu_indices(N, k=1)
            dists = np.sqrt(d2[iu])
            margins = 1.0 - np.sqrt(np.sum(pts * pts, axis=1))
            vals = np.concatenate([dists / 2.0 - r, margins - r])
            return _soft_min(vals, beta)
        res = minimize(obj, points.flatten(), method='L-BFGS-B',
                       options={'maxiter': 80, 'ftol': 1e-14})
        # feasible if soft-min (approx min) > -small
        x = res.x.reshape(N, 2)
        diff = x[:, None, :] - x[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        np.fill_diagonal(d2, np.inf)
        iu = np.triu_indices(N, k=1)
        dists = np.sqrt(d2[iu])
        margins = 1.0 - np.sqrt(np.sum(x * x, axis=1))
        minval = min(np.min(dists) / 2.0, np.min(margins))
        return minval >= r - 1e-9, x

    best = points
    best_score = _score(points)
    for _ in range(iters):
        if _time_left() < 4.0:
            break
        r_mid = 0.5 * (r_lo + r_hi)
        ok, x = feasible(r_mid)
        s = _score(x)
        if s > best_score:
            best_score = s
            best = x
        if ok:
            r_lo = r_mid
        else:
            r_hi = r_mid
    return best, best_score


def construct():
    global _START
    _START = time.time()

    # Multi-start: hex, ring, a few random
    starts = [_get_hex_lattice(), _get_ring_init()]
    for s in range(3, 10):
        starts.append(_get_random_init(s))

    best_points = starts[0].copy()
    best_score = _score(best_points)

    # Phase 1: moderate-beta polish from each start
    for st in starts:
        if _time_left() < 8.0:
            break
        pts = _polish(st, beta=15.0, max_iter=150)
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()

    # Phase 2: binary-search feasibility refinement from best
    if _time_left() > 15.0:
        # estimate current r
        r_cur = best_score
        r_lo = max(0.0, r_cur - 0.02)
        r_hi = r_cur + 0.03
        pts, s = _feasibility_search(best_points, r_lo, r_hi, beta=20.0, iters=10)
        if s > best_score:
            best_score = s
            best_points = pts.copy()

    # Phase 3: high-beta polish
    if _time_left() > 6.0:
        pts = _high_beta_polish(best_points)
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()

    # Final safety: ensure valid
    if best_points is None or not np.all(np.isfinite(best_points)):
        best_points = _get_hex_lattice()

    return best_points
