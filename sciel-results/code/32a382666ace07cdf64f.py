import numpy as np
import scipy.spatial
import scipy.optimize
import time

N = 14

def _get_triangle_indices(n):
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                indices.append((i, j, k))
    return np.array(indices, dtype=np.int32)

_TRI_INDICES = _get_triangle_indices(N)
_I = _TRI_INDICES[:, 0]
_J = _TRI_INDICES[:, 1]
_K = _TRI_INDICES[:, 2]

def _min_triangle_area(points):
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross).min()

def _hull_area(points):
    try:
        hull = scipy.spatial.ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    try:
        if len(points) < 3:
            return 0.0
        ha = _hull_area(points)
        if ha < 1e-12:
            return 0.0
        mta = _min_triangle_area(points)
        return mta / ha
    except Exception:
        return 0.0

def _normalize(points):
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return points
        return points / np.sqrt(ha)
    except Exception:
        return points

def _neg_score(points_flat):
    points = points_flat.reshape(-1, 2)
    points = _normalize(points)
    return -_score(points)

def _soft_min_triangle_area(points, beta=100.0):
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    max_a = np.max(areas)
    log_sum = np.log(np.sum(np.exp(-beta * (areas - max_a)))) / beta
    soft_min = max_a - log_sum
    return soft_min

def _neg_soft_score(points_flat, beta=100.0):
    points = points_flat.reshape(-1, 2)
    points = _normalize(points)
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return 0.0
        smta = _soft_min_triangle_area(points, beta)
        return -(smta / ha)
    except Exception:
        return 0.0

def _regular_polygon(n, radius=1.0, rotation=0.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _generate_two_ring(r_inner, delta_inner):
    n_ring = 7
    angles_outer = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
    points_outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    
    angles_inner = np.linspace(delta_inner, delta_inner + 2 * np.pi, n_ring, endpoint=False)
    points_inner = np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
    
    return np.vstack([points_outer, points_inner])

def _neg_score_two_ring(params):
    r, delta = params
    if r < 0: r = 0
    if r > 1: r = 1
    pts = _generate_two_ring(r, delta)
    pts = _normalize(pts)
    return -_score(pts)

def _generate_seven_gon_midpoints():
    verts = _regular_polygon(7, 1.0)
    mids = (verts + np.roll(verts, -1, axis=0)) / 2.0
    return np.vstack([verts, mids])

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_score = -np.inf
    best_points = None
    
    def _update_best(pts, sc):
        nonlocal best_score, best_points
        if sc > best_score:
            best_score = sc
            best_points = _normalize(pts).copy()

    def _time_left():
        return deadline - time.time()

    # 1. Optimize 2-ring structure first (fast)
    # Grid search
    best_r = 0.5
    best_delta = 0.0
    best_sc_2ring = -np.inf
    
    r_vals = np.linspace(0.1, 0.95, 20)
    delta_vals = np.linspace(0, 2*np.pi/7, 15, endpoint=False)
    
    for r in r_vals:
        for d in delta_vals:
            sc = -_neg_score_two_ring([r, d])
            if sc > best_sc_2ring:
                best_sc_2ring = sc
                best_r = r
                best_delta = d
                
    # Refine 2-ring with Nelder-Mead
    try:
        res_2ring = scipy.optimize.minimize(
            _neg_score_two_ring,
            [best_r, best_delta],
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8}
        )
        best_r = res_2ring.x[0]
        best_delta = res_2ring.x[1]
    except Exception:
        pass
    
    pts_2ring = _normalize(_generate_two_ring(best_r, best_delta))
    sc_2ring = _score(pts_2ring)
    _update_best(pts_2ring, sc_2ring)

    # 2. Full 28D optimization from best 2-ring start
    if _time_left() > 20:
        try:
            res_full = scipy.optimize.minimize(
                _neg_score,
                pts_2ring.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            pts_full = _normalize(res_full.x.reshape(-1, 2))
            sc_full = _score(pts_full)
            _update_best(pts_full, sc_full)
        except Exception:
            pass

    # 3. Try other starts
    other_starts = [
        _normalize(_regular_polygon(14)),
        _normalize(_generate_seven_gon_midpoints()),
    ]
    
    # Add a few random starts
    rng = np.random.RandomState(42)
    for _ in range(3):
        angles = rng.uniform(0, 2*np.pi, 14)
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
        other_starts.append(_normalize(pts))

    for start in other_starts:
        if _time_left() < 15:
            break
        sc_init = _score(start)
        _update_best(start, sc_init)
        
        try:
            res = scipy.optimize.minimize(
                _neg_score,
                start.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            pts_opt = _normalize(res.x.reshape(-1, 2))
            sc_opt = _score(pts_opt)
            _update_best(pts_opt, sc_opt)
        except Exception:
            pass

    # 4. Final polish
    if best_points is not None and _time_left() > 10:
        for beta in [100, 200, 500]:
            if _time_left() < 5:
                break
            try:
                res_polish = scipy.optimize.minimize(
                    lambda x: _neg_soft_score(x, beta),
                    best_points.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                pts_polish = _normalize(res_polish.x.reshape(-1, 2))
                sc_polish = _score(pts_polish)
                _update_best(pts_polish, sc_polish)
            except Exception:
                pass

    if best_points is None:
        best_points = _normalize(_regular_polygon(14))
    else:
        best_points = _normalize(best_points)
        
    return best_points
