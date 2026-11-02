"""Rewritten program: L-BFGS-B with soft-min (log-sum-exp) surrogate for min triangle area / hull area."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 14
DEADLINE_SECONDS = 120.0  # safety margin before 150s kill
_START_TIME = time.time()


def _remaining_time():
    return DEADLINE_SECONDS - (time.time() - _START_TIME)


def _triangle_areas_vectorized(pts):
    """Return array of areas for all C(n,3) triangles. Shape (m,)."""
    n = pts.shape[0]
    # Use the shoelace formula: area = 0.5 * |cross(p2-p1, p3-p1)|
    # Vectorize over triples using indices
    i, j, k = np.indices((n, n, n))
    mask = (i < j) & (j < k)
    i, j, k = i[mask], j[mask], k[mask]
    p1 = pts[i]
    p2 = pts[j]
    p3 = pts[k]
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
            (p3[:, 0] - p1[:, 0]) * (p2[:, 1] - p1[:, 1])
    areas = 0.5 * np.abs(cross)
    return areas


def _hull_area(pts):
    """Convex hull area of points; returns 0 if degenerate."""
    if pts.shape[0] < 3:
        return 0.0
    try:
        hull = ConvexHull(pts)
        return hull.volume  # in 2D, volume == area
    except Exception:
        return 0.0


def _min_ratio(pts):
    """True objective: min triangle area / hull area."""
    pts = np.asarray(pts, dtype=np.float64)
    hull_area = _hull_area(pts)
    if hull_area <= 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(pts)
    if areas.size == 0:
        return 0.0
    return float(areas.min() / hull_area)


def _soft_min_value(values, temperature):
    """Soft-min of an array of positive values via log-sum-exp."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    # soft_min = - (1/t) * log(sum(exp(-t * x)))
    # For numerical stability, shift by min
    v = values - values.min()
    # Now v >= 0, soft_min = values.min() - (1/t)*log(sum(exp(-t*v)))
    exp_term = np.exp(-temperature * v)
    s = np.sum(exp_term)
    if s <= 0:
        return float(values.min())
    return float(values.min() - np.log(s) / temperature)


def _neg_objective(pts_flat, temperature):
    """Negative of soft_min(triangle_areas) / hull_area. L-BFGS-B minimizes."""
    pts = pts_flat.reshape(-1, 2)
    hull_area = _hull_area(pts)
    if hull_area <= 1e-12:
        return 1e10
    areas = _triangle_areas_vectorized(pts)
    if areas.size == 0:
        return 1e10
    soft_min_area = _soft_min_value(areas, temperature)
    ratio = soft_min_area / hull_area
    return -ratio


def _optimize_from_start(start_pts, max_iter, time_budget):
    """Run L-BFGS-B from a start, with annealed temperature. Returns best pts found."""
    best_pts = start_pts.copy()
    best_score = _min_ratio(best_pts)
    n = best_pts.shape[0]

    # Annealing schedule: start with low temperature (smooth), increase to sharp
    # We'll do multiple phases with increasing temperature
    phases = [
        (5.0, max_iter // 3),
        (20.0, max_iter // 3),
        (100.0, max_iter - 2 * (max_iter // 3)),
    ]

    current = start_pts.copy().ravel()
    for temp, iters in phases:
        if time_budget() <= 2.0:
            break
        res = minimize(
            _neg_objective,
            current,
            args=(temp,),
            method='L-BFGS-B',
            options={'maxiter': iters, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50},
            bounds=[(0.0, 1.0)] * (2 * n),
        )
        if res.x is not None:
            cand = res.x.reshape(-1, 2)
            score = _min_ratio(cand)
            if score > best_score:
                best_score = score
                best_pts = cand.copy()
            current = res.x.copy()

    # Final polish with very high temperature (close to true min)
    if time_budget() > 3.0:
        res = minimize(
            _neg_objective,
            current,
            args=(500.0,),
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100},
            bounds=[(0.0, 1.0)] * (2 * n),
        )
        if res.x is not None:
            cand = res.x.reshape(-1, 2)
            score = _min_ratio(cand)
            if score > best_score:
                best_score = score
                best_pts = cand.copy()

    return best_pts, best_score


def _diverse_starts(rng, count):
    """Generate diverse starting configurations."""
    starts = []
    # Regular polygon on unit circle (centered, scaled to unit area)
    # Unit area regular n-gon: area = (n/2) * R^2 * sin(2*pi/n) => R = sqrt(2/(n*sin(2*pi/n)))
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    R = np.sqrt(2.0 / (N * np.sin(2.0 * np.pi / N)))
    poly = np.column_stack([R * np.cos(angles), R * np.sin(angles)])
    # Center and scale to [0,1] box
    poly = (poly - poly.min(axis=0)) / (poly.max(axis=0) - poly.min(axis=0))
    starts.append(poly)

    # Random starts
    for _ in range(count - 1):
        starts.append(np.array([[rng.random(), rng.random()] for _ in range(N)]))
    return starts


def construct():
    rng = np.random.RandomState(N)
    starts = _diverse_starts(rng, count=6)

    best_pts = starts[0].copy()
    best_score = _min_ratio(best_pts)

    # Allocate time: give each start a budget, but stop early if deadline near
    # Total budget ~ 110s to be safe
    total_budget = 110.0
    per_start_budget = total_budget / len(starts)
    max_iter_per_start = 200

    for i, start in enumerate(starts):
        remaining = _remaining_time()
        if remaining < 5.0:
            break
        # Give this start a chunk of the remaining time
        alloc = min(per_start_budget, remaining - 5.0)
        if alloc < 2.0:
            break

        # time_budget closure for this optimization
        local_start = time.time()
        def time_budget():
            return alloc - (time.time() - local_start)

        pts, score = _optimize_from_start(start, max_iter_per_start, time_budget)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()

    # Final global check
    final_score = _min_ratio(best_pts)
    if final_score > best_score:
        best_score = final_score

    return best_pts


# Ensure construct returns an (N, 2) array
def construct():
    result = _construct_impl()
    result = np.asarray(result, dtype=np.float64).reshape(N, 2)
    return result


def _construct_impl():
    rng = np.random.RandomState(N)
    starts = _diverse_starts(rng, count=6)

    best_pts = starts[0].copy()
    best_score = _min_ratio(best_pts)

    total_budget = 110.0
    per_start_budget = total_budget / len(starts)
    max_iter_per_start = 200

    for i, start in enumerate(starts):
        remaining = _remaining_time()
        if remaining < 5.0:
            break
        alloc = min(per_start_budget, remaining - 5.0)
        if alloc < 2.0:
            break

        local_start = time.time()
        def time_budget():
            return alloc - (time.time() - local_start)

        pts, score = _optimize_from_start(start, max_iter_per_start, time_budget)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()

    final_score = _min_ratio(best_pts)
    if final_score > best_score:
        best_score = final_score

    return best_pts
