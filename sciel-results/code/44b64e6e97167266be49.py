import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 14

TRIPLE_INDICES = np.array([
    (i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)
])

def _calculate_score(points):
    if points.shape != (N, 2):
        return 0.0
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    min_area = 0.5 * np.min(np.abs(cross))
    return min_area / hull_area

def _soft_min_score(points, temperature=0.02):
    """Smooth surrogate: soft-minimum of triangle areas / hull area."""
    if points.shape != (N, 2):
        return 0.0
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    m = np.max(areas)
    if m < 1e-12:
        return 0.0
    shifted = areas - m
    s = np.sum(np.exp(shifted / temperature))
    soft_min = m - temperature * np.log(s)
    return soft_min / hull_area

def _generate_two_ring(n_out, n_in, r_in, phi_rel):
    angles_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
    pts_out = np.column_stack((np.cos(angles_out), np.sin(angles_out)))
    if n_in > 0:
        angles_in = np.linspace(0, 2 * np.pi, n_in, endpoint=False) + phi_rel
        pts_in = np.column_stack((r_in * np.cos(angles_in), r_in * np.sin(angles_in)))
        return np.vstack((pts_out, pts_in))
    return pts_out

def _optimize_two_ring(n_out, n_in, time_limit=8.0):
    start_time = time.time()
    best_score = 0.0
    best_params = None

    def objective(params):
        r_in, phi_rel = params
        if r_in < 0.0 or r_in > 1.5:
            return 1e6
        pts = _generate_two_ring(n_out, n_in, r_in, phi_rel)
        return -_calculate_score(pts)

    r_starts = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]
    phi_starts = [0.0, np.pi / n_in if n_in > 0 else 0.0,
                  np.pi / n_out if n_out > 0 else 0.0,
                  np.pi / (2.0 * n_in) if n_in > 0 else 0.0]
    for r0 in r_starts:
        for phi0 in phi_starts:
            if time.time() - start_time > time_limit:
                return best_score, best_params
            try:
                res = minimize(objective, [r0, phi0], method='Nelder-Mead',
                               options={'xatol': 1e-7, 'fatol': 1e-7, 'maxiter': 400, 'disp': False})
                if -res.fun > best_score:
                    best_score = -res.fun
                    best_params = res.x.copy()
            except Exception:
                pass
    return best_score, best_params

def _polish_points(points, time_limit=5.0):
    start_time = time.time()
    p = points.copy()
    s = _calculate_score(p)

    def neg_soft(x):
        return -_soft_min_score(x.reshape(N, 2))

    def neg_hard(x):
        return -_calculate_score(x.reshape(N, 2))

    # L-BFGS-B on smooth surrogate
    try:
        res = minimize(neg_soft, p.flatten(), method='L-BFGS-B',
                       options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
        np2 = res.x.reshape(N, 2)
        s2 = _calculate_score(np2)
        if s2 > s:
            p, s = np2, s2
    except Exception:
        pass

    # L-BFGS-B on the true (piecewise) objective for a final nudge
    if time.time() - start_time < time_limit * 0.6:
        try:
            res = minimize(neg_hard, p.flatten(), method='L-BFGS-B',
                           options={'maxiter': 100, 'ftol': 1e-13, 'gtol': 1e-11})
            np2 = res.x.reshape(N, 2)
            s2 = _calculate_score(np2)
            if s2 > s:
                p, s = np2, s2
        except Exception:
            pass

    # Nelder-Mead fallback polish
    if time.time() - start_time < time_limit * 0.5:
        try:
            res = minimize(neg_hard, p.flatten(), method='Nelder-Mead',
                           options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 800, 'disp': False})
            np2 = res.x.reshape(N, 2)
            s2 = _calculate_score(np2)
            if s2 > s:
                p, s = np2, s2
        except Exception:
            pass

    return p, s

def _generate_single_ring(angles):
    return np.column_stack((np.cos(angles), np.sin(angles)))

def _optimize_single_ring(time_limit=8.0):
    """Pure 1D angular packing on a unit circle (single concentric ring)."""
    start_time = time.time()
    best_score = 0.0
    best_angles = None

    def score_from_angles(a):
        pts = _generate_single_ring(a)
        return _calculate_score(pts)

    def neg_score(a):
        return -score_from_angles(a)

    # Regular polygon is the natural optimum for a single ring; still polish it.
    base = np.linspace(0, 2 * np.pi, N, endpoint=False)
    s = score_from_angles(base)
    if s > best_score:
        best_score, best_angles = s, base.copy()

    # Small perturbations + Nelder-Mead to confirm / refine
    rng = np.random.default_rng(7)
    for trial in range(6):
        if time.time() - start_time > time_limit:
            break
        a0 = base + rng.uniform(-0.15, 0.15, size=N)
        try:
            res = minimize(neg_score, a0, method='Nelder-Mead',
                           options={'xatol': 1e-7, 'fatol': 1e-7, 'maxiter': 600, 'disp': False})
            s = score_from_angles(res.x)
            if s > best_score:
                best_score, best_angles = s, res.x.copy()
        except Exception:
            pass
    return best_score, best_angles

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before the 150s kill
    best_score = 0.0
    best_points = None

    def try_update(score, points):
        nonlocal best_score, best_points
        if points is not None and score > best_score:
            best_score = score
            best_points = points.copy()

    # 1. Single-ring angular-packing family (distinct modelling family)
    if time.time() < deadline - 20:
        s, ang = _optimize_single_ring(time_limit=8.0)
        if ang is not None:
            pts = _generate_single_ring(ang)
            polished, ps = _polish_points(pts, time_limit=4.0)
            try_update(ps, polished)

    # 2. Broad two-ring sweep over all splits
    splits = [(14, 0), (13, 1), (12, 2), (11, 3), (10, 4), (9, 5), (8, 6), (7, 7)]
    for n_out, n_in in splits:
        if time.time() > deadline - 15:
            break
        if n_in == 0:
            pts = _generate_two_ring(n_out, 0, 0.0, 0.0)
            polished, ps = _polish_points(pts, time_limit=3.0)
            try_update(ps, polished)
            continue
        score, params = _optimize_two_ring(n_out, n_in, time_limit=8.0)
        if params is not None:
            pts = _generate_two_ring(n_out, n_in, params[0], params[1])
            polished, ps = _polish_points(pts, time_limit=5.0)
            try_update(ps, polished)

    # 3. Final polish of the best result if time remains
    if best_points is not None and time.time() < deadline - 5:
        remaining = deadline - time.time() - 2
        if remaining > 3:
            polished, ps = _polish_points(best_points, time_limit=remaining)
            try_update(ps, polished)

    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))

    return best_points
