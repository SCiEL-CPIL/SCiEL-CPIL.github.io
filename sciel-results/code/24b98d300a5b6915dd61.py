import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 14

TRIPLE_INDICES = np.array([
    (i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)
])

# Precompute the half-determinant matrix for fast area computation
# area(i,j,k) = 0.5 * |det([p_j - p_i, p_k - p_i])|
# = 0.5 * |p_i x p_j + p_j x p_k + p_k x p_i|
# where (a,b) x (c,d) = a*d - b*c
# So we can precompute a matrix M such that cross[i,j,k] = sum over terms
# Actually: det(p_j - p_i, p_k - p_i) = det(p_j, p_k) - det(p_j, p_i) - det(p_i, p_k) + det(p_i, p_i)
# = det(p_j, p_k) - det(p_j, p_i) - det(p_i, p_k)
# Let cross(a,b) = a[0]*b[1] - a[1]*b[0]
# Then det(p_j-p_i, p_k-p_i) = cross(p_j, p_k) - cross(p_j, p_i) - cross(p_i, p_k)
# Note cross(p_i, p_k) = -cross(p_k, p_i)
# So = cross(p_j, p_k) + cross(p_i, p_j) + cross(p_k, p_i)

# We can compute all cross products between pairs: C[i,j] = cross(p_i, p_j)
# Then for triple (i,j,k): val = C[j,k] + C[i,j] + C[k,i] = C[j,k] + C[i,j] - C[i,k]

def _compute_areas(points):
    """Compute all triangle areas efficiently using precomputed cross products."""
    p = points  # (N, 2)
    # C[i,j] = p[i][0]*p[j][1] - p[i][1]*p[j][0]
    C = p[:, 0:1] * p[:, 1:2].T - p[:, 1:2] * p[:, 0:1].T
    # For triple (i,j,k): val = C[j,k] + C[i,j] - C[i,k]
    i_idx = TRIPLE_INDICES[:, 0]
    j_idx = TRIPLE_INDICES[:, 1]
    k_idx = TRIPLE_INDICES[:, 2]
    vals = C[j_idx, k_idx] + C[i_idx, j_idx] - C[i_idx, k_idx]
    return 0.5 * np.abs(vals)

def _hull_area(points):
    """Compute convex hull area using shoelace formula (faster than ConvexHull for small N)."""
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _calculate_score(points):
    if points.shape != (N, 2):
        return 0.0
    try:
        hull_area = _hull_area(points)
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
    areas = _compute_areas(points)
    min_area = np.min(areas)
    return min_area / hull_area

def _soft_min_score(points, temperature=0.015):
    """Smooth surrogate: soft-minimum of triangle areas / hull area."""
    if points.shape != (N, 2):
        return 0.0
    try:
        hull_area = _hull_area(points)
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
    areas = _compute_areas(points)
    m = np.max(areas)
    if m < 1e-12:
        return 0.0
    shifted = areas - m
    s = np.sum(np.exp(shifted / temperature))
    soft_min = m - temperature * np.log(s)
    return soft_min / hull_area

def _soft_min_score_v2(points, temperature=0.005):
    """Sharper soft-min for final polishing."""
    if points.shape != (N, 2):
        return 0.0
    try:
        hull_area = _hull_area(points)
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
    areas = _compute_areas(points)
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

def _generate_three_ring(n1, n2, n3, r2, r3, phi2, phi3):
    """Three concentric rings: outer (unit), middle (r2), inner (r3)."""
    a1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
    p1 = np.column_stack((np.cos(a1), np.sin(a1)))
    a2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + phi2
    p2 = np.column_stack((r2 * np.cos(a2), r2 * np.sin(a2)))
    a3 = np.linspace(0, 2 * np.pi, n3, endpoint=False) + phi3
    p3 = np.column_stack((r3 * np.cos(a3), r3 * np.sin(a3)))
    return np.vstack((p1, p2, p3))

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

    # More starting points for better coverage
    r_starts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    phi_starts = [0.0]
    if n_in > 0:
        phi_starts += [np.pi / n_in, np.pi / (2.0 * n_in), 2.0 * np.pi / n_in,
                       np.pi / (3.0 * n_in), np.pi / (4.0 * n_in)]
    if n_out > 0:
        phi_starts += [np.pi / n_out, np.pi / (2.0 * n_out)]

    for r0 in r_starts:
        for phi0 in phi_starts:
            if time.time() - start_time > time_limit:
                return best_score, best_params
            try:
                res = minimize(objective, [r0, phi0], method='Nelder-Mead',
                               options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 500, 'disp': False})
                if -res.fun > best_score:
                    best_score = -res.fun
                    best_params = res.x.copy()
            except Exception:
                pass
    return best_score, best_params

def _optimize_three_ring(n1, n2, n3, time_limit=8.0):
    start_time = time.time()
    best_score = 0.0
    best_params = None

    def objective(params):
        r2, r3, phi2, phi3 = params
        if r2 < 0.0 or r2 > 1.5 or r3 < 0.0 or r3 > 1.5:
            return 1e6
        pts = _generate_three_ring(n1, n2, n3, r2, r3, phi2, phi3)
        return -_calculate_score(pts)

    r_starts = [0.3, 0.5, 0.7, 0.9]
    r3_starts = [0.2, 0.4, 0.6, 0.8]
    phi_starts = [0.0, np.pi / n2 if n2 > 0 else 0.0, np.pi / n3 if n3 > 0 else 0.0]

    count = 0
    for r2 in r_starts:
        for r3 in r3_starts:
            for phi2 in phi_starts:
                for phi3 in phi_starts:
                    if time.time() - start_time > time_limit or count >= 20:
                        return best_score, best_params
                    count += 1
                    try:
                        res = minimize(objective, [r2, r3, phi2, phi3], method='Nelder-Mead',
                                       options={'xatol': 1e-7, 'fatol': 1e-7, 'maxiter': 400, 'disp': False})
                        if -res.fun > best_score:
                            best_score = -res.fun
                            best_params = res.x.copy()
                    except Exception:
                        pass
    return best_score, best_params

def _coordinate_descent_polish(points, time_limit=5.0):
    """Optimize each point individually while holding others fixed."""
    start_time = time.time()
    p = points.copy()
    s = _calculate_score(p)

    for iteration in range(3):
        if time.time() - start_time > time_limit:
            break
        for i in range(N):
            if time.time() - start_time > time_limit:
                break
            def obj_2d(x):
                p[i] = x
                return -_calculate_score(p)
            try:
                res = minimize(obj_2d, p[i], method='Nelder-Mead',
                               options={'xatol': 1e-9, 'fatol': 1e-9, 'maxiter': 300, 'disp': False})
                p[i] = res.x
            except Exception:
                pass
        s_new = _calculate_score(p)
        if s_new > s:
            s = s_new
        else:
            # Revert if worse
            p = points.copy()
            break
    return p, s

def _polish_points(points, time_limit=5.0):
    start_time = time.time()
    p = points.copy()
    s = _calculate_score(p)

    # Pass 1: L-BFGS-B on smooth surrogate (wider temperature)
    def neg_soft(x):
        return -_soft_min_score(x.reshape(N, 2))
    try:
        res = minimize(neg_soft, p.flatten(), method='L-BFGS-B',
                       options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
        np2 = res.x.reshape(N, 2)
        s2 = _calculate_score(np2)
        if s2 > s:
            p, s = np2, s2
    except Exception:
        pass

    # Pass 2: L-BFGS-B on sharper surrogate
    if time.time() - start_time < time_limit * 0.7:
        def neg_soft2(x):
            return -_soft_min_score_v2(x.reshape(N, 2))
        try:
            res = minimize(neg_soft2, p.flatten(), method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
            np2 = res.x.reshape(N, 2)
            s2 = _calculate_score(np2)
            if s2 > s:
                p, s = np2, s2
        except Exception:
            pass

    # Pass 3: L-BFGS-B on the true (piecewise) objective
    if time.time() - start_time < time_limit * 0.5:
        def neg_hard(x):
            return -_calculate_score(x.reshape(N, 2))
        try:
            res = minimize(neg_hard, p.flatten(), method='L-BFGS-B',
                           options={'maxiter': 150, 'ftol': 1e-15, 'gtol': 1e-13})
            np2 = res.x.reshape(N, 2)
            s2 = _calculate_score(np2)
            if s2 > s:
                p, s = np2, s2
        except Exception:
            pass

    # Pass 4: Coordinate descent polish
    if time.time() - start_time < time_limit * 0.3:
        remaining = time_limit - (time.time() - start_time)
        if remaining > 2.0:
            p, s = _coordinate_descent_polish(p, time_limit=remaining)

    # Pass 5: Nelder-Mead fallback on hard objective
    if time.time() - start_time < time_limit * 0.15:
        remaining = time_limit - (time.time() - start_time)
        if remaining > 1.0:
            def neg_hard(x):
                return -_calculate_score(x.reshape(N, 2))
            try:
                res = minimize(neg_hard, p.flatten(), method='Nelder-Mead',
                               options={'xatol': 1e-9, 'fatol': 1e-9, 'maxiter': 600, 'disp': False})
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

    # Regular polygon
    base = np.linspace(0, 2 * np.pi, N, endpoint=False)
    s = score_from_angles(base)
    if s > best_score:
        best_score, best_angles = s, base.copy()

    # Perturbations + Nelder-Mead
    rng = np.random.default_rng(42)
    for trial in range(8):
        if time.time() - start_time > time_limit:
            break
        a0 = base + rng.uniform(-0.2, 0.2, size=N)
        try:
            res = minimize(neg_score, a0, method='Nelder-Mead',
                           options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 800, 'disp': False})
            s = score_from_angles(res.x)
            if s > best_score:
                best_score, best_angles = s, res.x.copy()
        except Exception:
            pass
    return best_score, best_angles

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before the 150s kill
    best_score = 0.0
    best_points = None

    def try_update(score, points):
        nonlocal best_score, best_points
        if points is not None and score > best_score:
            best_score = score
            best_points = points.copy()

    # 1. Single-ring angular-packing family
    if time.time() < deadline - 30:
        s, ang = _optimize_single_ring(time_limit=6.0)
        if ang is not None:
            pts = _generate_single_ring(ang)
            polished, ps = _polish_points(pts, time_limit=5.0)
            try_update(ps, polished)

    # 2. Two-ring sweep over all splits
    splits = [(14, 0), (13, 1), (12, 2), (11, 3), (10, 4), (9, 5), (8, 6), (7, 7)]
    for n_out, n_in in splits:
        if time.time() > deadline - 20:
            break
        if n_in == 0:
            pts = _generate_two_ring(n_out, 0, 0.0, 0.0)
            polished, ps = _polish_points(pts, time_limit=3.0)
            try_update(ps, polished)
            continue
        # Allocate more time to promising splits
        tl = 7.0 if n_in >= 5 else 5.0
        score, params = _optimize_two_ring(n_out, n_in, time_limit=tl)
        if params is not None:
            pts = _generate_two_ring(n_out, n_in, params[0], params[1])
            polished, ps = _polish_points(pts, time_limit=5.0)
            try_update(ps, polished)

    # 3. Three-ring exploration (e.g., 5-4-5, 6-3-5, 5-5-4)
    three_splits = [(5, 4, 5), (6, 3, 5), (5, 5, 4), (4, 5, 5)]
    for n1, n2, n3 in three_splits:
        if time.time() > deadline - 15:
            break
        score, params = _optimize_three_ring(n1, n2, n3, time_limit=5.0)
        if params is not None:
            pts = _generate_three_ring(n1, n2, n3, params[0], params[1], params[2], params[3])
            polished, ps = _polish_points(pts, time_limit=4.0)
            try_update(ps, polished)

    # 4. Final polish of the best result if time remains
    if best_points is not None and time.time() < deadline - 5:
        remaining = deadline - time.time() - 2
        if remaining > 3:
            polished, ps = _polish_points(best_points, time_limit=remaining)
            try_update(ps, polished)

    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))

    return best_points
