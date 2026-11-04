"""
Heilbronn convex regions: 13 points maximizing min triangle area / hull area.

Strategy:
- Use a smooth soft-min objective (log-sum-exp of negative scaled areas) to make
  the max-min problem amenable to gradient-based optimization (L-BFGS-B).
- Multi-start from diverse initializations (regular polygon, concentric rings, Halton).
- Polish the best candidate with Nelder-Mead on the exact (non-smmooth) objective.
- Enforce convex position implicitly by optimizing in 2D and relying on the
  convex hull area normalization; the soft-min encourages all triangles to be
  reasonably large, which pushes points toward the hull.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13
_TRIPLETS = np.array(list(combinations(range(N), 3)))
_N_TRIPLES = _TRIPLETS.shape[0]
_I, _J, _K = _TRIPLETS.T


def _triangle_areas(pts):
    """Compute all C(n,3) triangle areas vectorized."""
    a = pts[_I]
    b = pts[_J]
    c = pts[_K]
    areas = 0.5 * np.abs(
        (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) -
        (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
    )
    return areas


def _convex_hull_area(pts):
    """Compute area of convex hull."""
    try:
        if len(pts) < 3:
            return 0.0
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area <= 0:
        return 0.0
    areas = _triangle_areas(pts)
    if len(areas) == 0:
        return 0.0
    return np.min(areas) / hull_area


def _soft_min_objective(x, beta=80.0):
    """
    Negative soft-min of (min triangle area / hull area) for minimization.
    Uses log-sum-exp for numerical stability.
    x: (2N,) flattened point coordinates.
    """
    pts = x.reshape(-1, 2)
    hull_area = _convex_hull_area(pts)
    if hull_area <= 1e-12:
        return 1e6
    areas = _triangle_areas(pts)
    if len(areas) == 0:
        return 1e6
    # We want to maximize min(areas) / hull_area
    # Equivalent to minimizing -min(areas)/hull_area
    # Soft-min of areas: approx min(areas) via -logsumexp(-beta * areas) / beta
    neg_scaled = -beta * areas
    max_neg = np.max(neg_scaled)
    soft_min_areas = -(max_neg + np.log(np.sum(np.exp(neg_scaled - max_neg)))) / beta
    return -soft_min_areas / hull_area


def _exact_objective(x):
    """Negative exact score for minimization (non-smooth)."""
    pts = x.reshape(-1, 2)
    return -_score(pts)


def _halton_sequence(n, base):
    """Generate first n Halton sequence values with given base."""
    result = []
    for i in range(1, n + 1):
        f = 1.0
        r = 0.0
        idx = i
        while idx > 0:
            f /= base
            r += f * (idx % base)
            idx //= base
        result.append(r)
    return np.array(result)


def _halton_points_2d(n, seed=0):
    """Generate n 2D points using Halton sequences (bases 2 and 3)."""
    rng = np.random.RandomState(seed)
    offset = rng.randint(1, 1000)
    h2 = _halton_sequence(n + offset, 2)
    h3 = _halton_sequence(n + offset, 3)
    pts = np.column_stack([h2[offset - 1:offset - 1 + n],
                           h3[offset - 1:offset - 1 + n]])
    return pts


def _concentric_ring_init(seed=0, n_inner=5, n_outer=8):
    """Generate points on two concentric rings for diverse initialization."""
    rng = np.random.RandomState(seed)
    pts = np.zeros((N, 2))
    angles_in = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + rng.uniform(0, 2 * np.pi)
    r_in = 0.4 + rng.uniform(-0.1, 0.1)
    pts[:n_inner, 0] = r_in * np.cos(angles_in)
    pts[:n_inner, 1] = r_in * np.sin(angles_in)
    angles_out = np.linspace(0, 2 * np.pi, n_outer, endpoint=False) + rng.uniform(0, 2 * np.pi)
    r_out = 1.0 + rng.uniform(-0.1, 0.1)
    pts[n_inner:, 0] = r_out * np.cos(angles_out)
    pts[n_inner:, 1] = r_out * np.sin(angles_out)
    return pts


def _regular_polygon_init(seed=0):
    """Regular polygon with small perturbation."""
    rng = np.random.RandomState(seed)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + rng.uniform(0, 2 * np.pi)
    r = 1.0 + rng.normal(0, 0.05, N)
    pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
    return pts


def _nelder_mead_polish(pts, max_iter=500, tol=1e-10):
    """Polish using Nelder-Mead on the exact (non-smooth) objective."""
    x0 = pts.flatten()
    try:
        result = minimize(
            _exact_objective,
            x0,
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': tol, 'fatol': tol, 'adaptive': True}
        )
        return result.x.reshape(-1, 2)
    except Exception:
        return pts


def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    best_score = 0.0
    best_pts = _regular_polygon_init(0)

    # Phase 1: Multi-start with soft-min L-BFGS-B on full 2D coordinates
    init_strategies = [
        _regular_polygon_init,
        _concentric_ring_init,
        lambda s: _halton_points_2d(N, seed=s) * 0.8,
    ]

    n_starts = 30
    for i in range(n_starts):
        if time.time() > deadline - 30:
            break

        strategy_idx = i % len(init_strategies)
        seed = i
        try:
            pts0 = init_strategies[strategy_idx](seed)
            x0 = pts0.flatten()

            result = minimize(
                _soft_min_objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )

            pts = result.x.reshape(-1, 2)
            score = _score(pts)

            if score > best_score:
                best_score = score
                best_pts = pts.copy()
        except Exception:
            continue

    # Phase 2: Nelder-Mead polish on the best solution
    if time.time() < deadline - 15:
        polished = _nelder_mead_polish(best_pts, max_iter=800)
        score = _score(polished)
        if score > best_score:
            best_score = score
            best_pts = polished.copy()

    # Phase 3: A few more diverse starts with full polish if time permits
    for seed in range(10):
        if time.time() > deadline - 5:
            break

        try:
            pts0 = _concentric_ring_init(seed + 100)
            x0 = pts0.flatten()

            result = minimize(
                _soft_min_objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-13, 'gtol': 1e-8}
            )

            pts = result.x.reshape(-1, 2)
            pts = _nelder_mead_polish(pts, max_iter=200)
            score = _score(pts)

            if score > best_score:
                best_score = score
                best_pts = pts.copy()
        except Exception:
            continue

    # Final polish if time remains
    if time.time() < deadline - 2:
        final_polish = _nelder_mead_polish(best_pts, max_iter=300)
        score = _score(final_polish)
        if score > best_score:
            best_score = score
            best_pts = final_polish.copy()

    return best_pts
