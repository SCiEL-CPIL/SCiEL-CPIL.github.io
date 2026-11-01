import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14

# Precompute triangle indices
_IDX = np.array([(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)])
_I, _J, _K = _IDX[:, 0], _IDX[:, 1], _IDX[:, 2]

def compute_score(points):
    """Min triangle area / convex hull area (exact, used for final reporting)."""
    n = len(points)
    if n < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0
    p_i = points[_I]; p_j = points[_J]; p_k = points[_K]
    v1 = p_j - p_i; v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    min_area = np.min(areas)
    if min_area < 1e-12:
        return 0.0
    return min_area / hull_area

def _fast_hull_area(points):
    """Convex hull area computed directly (no ConvexHull object)."""
    p = points
    n = p.shape[0]
    # Find a pivot (lowest y, then lowest x)
    pivot = np.argmin(p[:, 1] * 1e6 + p[:, 0])
    order = np.argsort(np.arctan2(p[:, 1] - p[pivot, 1], p[:, 0] - p[pivot, 0]))
    hull_idx = [pivot] + list(order)
    # Andrew's monotone chain
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower = []
    for idx in hull_idx:
        while len(lower) >= 2 and cross(p[lower[-2]], p[lower[-1]], p[idx]) <= 0:
            lower.pop()
        lower.append(idx)
    upper = []
    for idx in reversed(hull_idx):
        while len(upper) >= 2 and cross(p[upper[-2]], p[upper[-1]], p[idx]) <= 0:
            upper.pop()
        upper.append(idx)
    hull = np.array(lower[:-1] + upper[:-1])
    if len(hull) < 3:
        return 0.0
    hx = p[hull, 0]; hy = p[hull, 1]
    return 0.5 * abs(np.sum(hx * np.roll(hy, -1) - np.roll(hx, -1) * hy))

def smooth_obj(x, tau, hull_penalty, hull_area_floor):
    """
    Differentiable objective to MINIMIZE.
    Uses a log-sum-exp lower bound on triangle areas (smooth, always <= min),
    so maximizing it pushes ALL triangles up, not just the current minimum.
    """
    points = x.reshape((N, 2))
    p_i = points[_I]; p_j = points[_J]; p_k = points[_K]
    v1 = p_j - p_i; v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    areas = np.maximum(areas, 1e-10)
    # log-sum-exp lower bound:  -tau * log(sum(exp(-areas/tau)))  (<= min(areas))
    scaled = -areas / tau
    m = np.max(scaled)
    lse = -tau * (m + np.log(np.sum(np.exp(scaled - m))))
    # smooth convex hull area
    hull_area = _fast_hull_area(points)
    if hull_area < hull_area_floor:
        hull_area = hull_area_floor
    return -lse / hull_area + hull_penalty * (1.0 / hull_area)

def annealed_minimize(x0, tau_schedule, hull_penalty, hull_area_floor, deadline, maxiter_per_tau=80):
    """Run L-BFGS-B with a decreasing tau (annealing) schedule."""
    x = x0.copy()
    for tau, hp in zip(tau_schedule, hull_penalty):
        if time.time() > deadline:
            break
        try:
            res = minimize(smooth_obj, x, args=(tau, hp, hull_area_floor),
                           method='L-BFGS-B',
                           options={'maxiter': maxiter_per_tau, 'ftol': 1e-12, 'gtol': 1e-7})
            x = res.x
        except Exception:
            pass
    return x

def construct():
    start_time = time.time()
    deadline = start_time + 138.0  # safety margin before 150s kill

    rng = np.random.default_rng(12345)

    best_points = None
    best_score = -1.0

    def try_update(points):
        nonlocal best_points, best_score
        s = compute_score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
        return s

    # ---------- Build diverse structural starting points ----------
    def ring_pts(r, n, th0):
        angles = th0 + np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.column_stack((r * np.cos(angles), r * np.sin(angles)))

    starts = []

    # Regular 14-gon
    starts.append(ring_pts(1.0, 14, 0.0))

    # Two rings 7+7 (various inner radii and offsets)
    for r_in in [0.3, 0.45, 0.55, 0.65, 0.75]:
        for off in [0.0, np.pi / 7]:
            starts.append(np.vstack((ring_pts(1.0, 7, 0.0), ring_pts(r_in, 7, off))))

    # Three rings
    for (n1, n2, n3, r1, r2, r3) in [
        (6, 5, 3, 1.0, 0.6, 0.25),
        (5, 5, 4, 1.0, 0.65, 0.3),
        (7, 4, 3, 1.0, 0.55, 0.2),
        (4, 5, 5, 1.0, 0.7, 0.35),
    ]:
        o1 = 0.0
        o2 = np.pi / n1
        o3 = np.pi / n2
        starts.append(np.vstack((
            ring_pts(r1, n1, o1),
            ring_pts(r2, n2, o2),
            ring_pts(r3, n3, o3),
        )))

    # Four rings
    for (n1, n2, n3, n4, r1, r2, r3, r4) in [
        (5, 4, 3, 2, 1.0, 0.7, 0.4, 0.15),
        (4, 4, 3, 3, 1.0, 0.7, 0.4, 0.15),
        (4, 4, 4, 2, 1.0, 0.7, 0.4, 0.15),
    ]:
        o1 = 0.0
        o2 = np.pi / n1
        o3 = np.pi / n2
        o4 = np.pi / n3
        starts.append(np.vstack((
            ring_pts(r1, n1, o1),
            ring_pts(r2, n2, o2),
            ring_pts(r3, n3, o3),
            ring_pts(r4, n4, o4),
        )))

    # Random starts
    for _ in range(8):
        r = 0.3 + 0.7 * rng.random(N)
        th = rng.uniform(0, 2 * np.pi, N)
        starts.append(np.column_stack((r * np.cos(th), r * np.sin(th))))

    # ---------- Phase 1: Annealed L-BFGS-B from each start ----------
    tau_schedule = [0.05, 0.02, 0.008, 0.003, 0.001]
    hp_schedule = [0.05, 0.02, 0.01, 0.005, 0.002]
    hull_area_floor = 0.3

    for idx, s in enumerate(starts):
        if time.time() > deadline - 15:
            break
        x0 = s.flatten()
        x_opt = annealed_minimize(x0, tau_schedule, hp_schedule, hull_area_floor, deadline, maxiter_per_tau=60)
        p_opt = x_opt.reshape((N, 2))
        try_update(p_opt)

    # ---------- Phase 2: Simulated-annealing random walk on full space ----------
    if best_points is not None:
        x = best_points.flatten().copy()
        cur_score = best_score
        T = 0.15
        for step in range(2000):
            if time.time() > deadline - 12:
                break
            x_new = x + rng.normal(0, T, (N, 2))
            s_new = compute_score(x_new.reshape((N, 2)))
            if s_new > cur_score or rng.random() < np.exp((s_new - cur_score) / (T * 0.1 + 1e-12)):
                x = x_new
                cur_score = s_new
            T *= 0.997
        try_update(x.reshape((N, 2)))

    # ---------- Phase 3: Per-point jitter + local polish ----------
    if best_points is not None:
        base = best_points.copy()
        for _ in range(12):
            if time.time() > deadline - 8:
                break
            x0 = base.flatten() + rng.normal(0, 0.05, (N, 2))
            try:
                res = minimize(smooth_obj, x0, args=(0.001, 0.002, hull_area_floor),
                               method='L-BFGS-B',
                               options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-7})
                try_update(res.x.reshape((N, 2)))
            except Exception:
                pass

    # ---------- Phase 4: Final direct Nelder-Mead polish (exact score) ----------
    if best_points is not None:
        x0 = best_points.flatten()
        try:
            res = minimize(lambda x: -compute_score(x.reshape((N, 2))), x0,
                           method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-11})
            try_update(res.x.reshape((N, 2)))
        except Exception:
            pass

    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))

    return best_points
