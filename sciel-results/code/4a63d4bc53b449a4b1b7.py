import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize
import time

N = 14
SOFT_BETA = 80.0

TRIPLE_INDICES = np.array([
    (i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)
])

_TRIPLE_POINT_TO_IDX = [np.where((TRIPLE_INDICES == i).any(axis=1))[0] for i in range(N)]
_POINT_OTHERS = []
for p in range(N):
    rows = _TRIPLE_POINT_TO_IDX[p]
    others = []
    for r in rows:
        a, b, c = TRIPLE_INDICES[r]
        if p == a:
            others.append((r, b, c))
        elif p == b:
            others.append((r, a, c))
        else:
            others.append((r, a, b))
    _POINT_OTHERS.append(others)


def _calculate_score(points):
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

    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])

    min_area = 0.5 * np.min(np.abs(cross))
    return min_area / hull_area


def _soft_min_score(points, beta=SOFT_BETA):
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

    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])

    areas = 0.5 * np.abs(cross)
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    soft_min_val = (1.0 / beta) * (max_scaled + np.log(np.sum(np.exp(scaled - max_scaled))))
    return soft_min_val / hull_area


def _fourier_points(c):
    """c = [c1, c2, c3] -> 14 points on r(theta)=1+sum c_k cos(k*theta)."""
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    r = 1.0 + c[0] * np.cos(theta) + c[1] * np.cos(2 * theta) + c[2] * np.cos(3 * theta)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack((x, y))


def _fourier_score(c, beta=None):
    if beta is None:
        beta = SOFT_BETA
    pts = _fourier_points(c)
    return _soft_min_score(pts, beta=beta)


def _optimize_fourier(deadline):
    best_c = np.zeros(3)
    best_score = _calculate_score(_fourier_points(best_c))
    best_pts = _fourier_points(best_c)

    # Differential evolution on 3 coefficients
    if time.time() < deadline - 30:
        try:
            bounds = [(-0.3, 0.3), (-0.15, 0.15), (-0.08, 0.08)]
            res = differential_evolution(
                lambda c: _fourier_score(c, beta=SOFT_BETA),
                bounds, seed=42, maxiter=30, popsize=15,
                tol=1e-6, polish=True
            )
            c_found = res.x
            pts_found = _fourier_points(c_found)
            s = _calculate_score(pts_found)
            if s > best_score:
                best_score = s
                best_c = c_found
                best_pts = pts_found
        except Exception:
            pass

    # L-BFGS-B polish
    if time.time() < deadline - 10:
        def neg_obj(c):
            return _fourier_score(c, beta=SOFT_BETA)
        try:
            res2 = minimize(neg_obj, best_c, method='L-BFGS-B',
                            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10})
            c2 = res2.x
            pts2 = _fourier_points(c2)
            s2 = _calculate_score(pts2)
            if s2 > best_score:
                best_score = s2
                best_c = c2
                best_pts = pts2
        except Exception:
            pass

    # Multi-start with perturbations
    if time.time() < deadline - 15:
        np.random.seed(7)
        for trial in range(8):
            if time.time() > deadline - 15:
                break
            c_init = best_c + np.random.normal(0, 0.05, 3)
            c_init = np.clip(c_init, [-0.35, -0.2, -0.1], [0.35, 0.2, 0.12])

            def neg_trial(c):
                return _fourier_score(c, beta=SOFT_BETA * 2)

            try:
                res_t = minimize(neg_trial, c_init, method='Nelder-Mead',
                                 options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 200})
                c_t = res_t.x
                pts_t = _fourier_points(c_t)
                s_t = _calculate_score(pts_t)
                if s_t > best_score:
                    best_score = s_t
                    best_c = c_t
                    best_pts = pts_t
            except Exception:
                pass

    return best_pts, best_score


def _ring_points(n_out, n_in, r, theta):
    angles_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
    x_out = np.cos(angles_out)
    y_out = np.sin(angles_out)
    angles_in = np.linspace(0, 2 * np.pi, n_in, endpoint=False) + theta
    x_in = r * np.cos(angles_in)
    y_in = r * np.sin(angles_in)
    return np.vstack([
        np.column_stack((x_out, y_out)),
        np.column_stack((x_in, y_in))
    ])


def _ring_score(n_out, n_in, params, beta=None):
    if beta is None:
        beta = SOFT_BETA
    r, theta = params
    if r < 0 or r > 0.999:
        return 0.0
    pts = _ring_points(n_out, n_in, r, theta)
    return _soft_min_score(pts, beta=beta)


def _local_polish(points, deadline):
    pts = points.copy()
    score = _calculate_score(pts)
    if score <= 0:
        return pts, score

    p_i = pts[TRIPLE_INDICES[:, 0]]
    p_j = pts[TRIPLE_INDICES[:, 1]]
    p_k = pts[TRIPLE_INDICES[:, 2]]
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    areas = 0.5 * np.abs(cross)

    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
    except Exception:
        return pts, score

    if hull_area <= 1e-9:
        return pts, score

    current_min = np.min(areas)
    current_ratio = current_min / hull_area
    step_size = 0.01

    for iteration in range(150):
        if time.time() > deadline - 2.0:
            break
        improved = False
        for p in range(N):
            if time.time() > deadline - 2.0:
                break
            best_local = pts[p].copy()
            best_ratio = current_ratio
            for dx in [-step_size, 0, step_size]:
                for dy in [-step_size, 0, step_size]:
                    if dx == 0 and dy == 0:
                        continue
                    new_px, new_py = pts[p, 0] + dx, pts[p, 1] + dy
                    new_areas = areas.copy()
                    for rr, o1, o2 in _POINT_OTHERS[p]:
                        a, b, c = TRIPLE_INDICES[rr]
                        if p == a:
                            cj = pts[b]; ck = pts[c]
                            cc = (cj[0]-new_px)*(ck[1]-new_py)-(cj[1]-new_py)*(ck[0]-new_px)
                        elif p == b:
                            ci = pts[a]; ck = pts[c]
                            cc = (ci[0]-new_px)*(ck[1]-new_py)-(ci[1]-new_py)*(ck[0]-new_px)
                        else:
                            ci = pts[a]; cj = pts[b]
                            cc = (ci[0]-new_px)*(cj[1]-new_py)-(ci[1]-new_py)*(cj[0]-new_px)
                        new_areas[rr] = 0.5 * abs(cc)
                    new_ratio = np.min(new_areas) / hull_area
                    if new_ratio > best_ratio + 1e-12:
                        best_ratio = new_ratio
                        best_local[0] = new_px
                        best_local[1] = new_py
            if not np.allclose(best_local, pts[p]):
                pts[p] = best_local
                for rr, o1, o2 in _POINT_OTHERS[p]:
                    a, b, c = TRIPLE_INDICES[rr]
                    ci = pts[a]; cj = pts[b]; ck = pts[c]
                    cc = (cj[0]-ci[0])*(ck[1]-ci[1])-(cj[1]-ci[1])*(ck[0]-ci[0])
                    areas[rr] = 0.5 * abs(cc)
                current_ratio = np.min(areas) / hull_area
                improved = True
        if not improved:
            step_size *= 0.5
            if step_size < 1e-5:
                break

    return pts, _calculate_score(pts)


def _full_polish(points, deadline):
    best_pts = points.copy()
    best_score = _calculate_score(points)

    if time.time() < deadline - 8:
        def neg_soft(x):
            return _soft_min_score(x.reshape(-1, 2), beta=SOFT_BETA)
        try:
            res = minimize(neg_soft, points.flatten(), method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10})
            p2 = res.x.reshape(-1, 2)
            s2 = _calculate_score(p2)
            if s2 > best_score:
                best_pts = p2
                best_score = s2
        except Exception:
            pass

    if time.time() < deadline - 4:
        best_pts, best_score = _local_polish(best_pts, deadline)

    return best_pts, best_score


def _optimize_ring(n_out, n_in, time_limit, deadline):
    g = np.gcd(n_out, n_in)
    theta_max = 2 * np.pi / g
    bounds = [(0.0, 0.999), (0.0, theta_max)]

    best_score = 0.0
    best_points = None

    if time_limit > 15:
        popsize, maxiter = 12, 30
    else:
        popsize, maxiter = 8, 20

    try:
        res = differential_evolution(
            lambda p: _ring_score(n_out, n_in, p),
            bounds, seed=42, maxiter=maxiter, popsize=popsize,
            tol=1e-5, polish=True
        )
        r, theta = res.x
        pts = _ring_points(n_out, n_in, r, theta)
        best_score = _calculate_score(pts)
        best_points = pts
    except Exception:
        pass

    if best_points is None and time_limit > 5:
        r, theta = 0.5, 0.0
        best_points = _ring_points(n_out, n_in, r, theta)
        best_score = _calculate_score(best_points)

    if best_points is not None and time.time() < deadline - 5:
        best_points, best_score = _full_polish(best_points, deadline)

    return best_score, best_points


def construct():
    start = time.time()
    deadline = start + 135.0

    best_score = 0.0
    best_points = None

    # Fourier strategy
    try:
        f_pts, f_score = _optimize_fourier(deadline - 20)
        if f_score > best_score:
            best_score = f_score
            best_points = f_pts.copy()
    except Exception:
        pass

    # Ring strategies
    candidates = [(7, 7), (8, 6), (6, 8), (9, 5), (5, 9), (10, 4), (4, 10)]
    for n_out, n_in in candidates:
        if time.time() > deadline - 8:
            break
        t_left = deadline - 8 - time.time()
        if t_left < 4:
            break
        try:
            r_score, r_pts = _optimize_ring(n_out, n_in, t_left, deadline)
            if r_pts is not None and r_score > best_score:
                best_score = r_score
                best_points = r_pts.copy()
        except Exception:
            pass

    # Final polish
    if best_points is not None and time.time() < deadline - 2:
        best_points, best_score = _full_polish(best_points, deadline)

    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))

    return best_points
