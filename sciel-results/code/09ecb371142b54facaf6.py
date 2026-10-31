import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings('ignore')

N = 13

_TRIPLE_IDX = np.array([
    (i, j, k)
    for i in range(N)
    for j in range(i + 1, N)
    for k in range(j + 1, N)
], dtype=np.int32)


def _triangle_areas_vectorized(points):
    pi = points[_TRIPLE_IDX[:, 0]]
    pj = points[_TRIPLE_IDX[:, 1]]
    pk = points[_TRIPLE_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)


def _hull_area(points):
    try:
        if len(points) < 3:
            return 0.0
        if np.allclose(points, points[0], atol=1e-12):
            return 0.0
        hull = ConvexHull(points)
        return float(hull.volume)
    except Exception:
        return 0.0


def _score(points):
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    min_area = float(np.min(areas))
    if min_area < 1e-15:
        return 0.0
    return min_area / hull_area


def _neg_score(points):
    s = _score(points)
    if s <= 0:
        return 1e10
    return -s


def _soft_min_value(areas, t):
    """Soft minimum of areas using log-sum-exp trick (stable)."""
    # We want to approximate min(areas) as areas go to infinity in the soft sense
    # soft_min = -log(sum(exp(-a/t))) * t
    neg_scaled = -areas / t
    max_neg = np.max(neg_scaled)
    if max_neg < -700:
        return float(np.min(areas))
    exp_vals = np.exp(neg_scaled - max_neg)
    s = np.sum(exp_vals)
    if s < 1e-300:
        return float(np.min(areas))
    return -t * (max_neg + np.log(s))


def _soft_score(points, t=0.01):
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    sm = _soft_min_value(areas, t)
    if sm < 1e-15:
        return 0.0
    return sm / hull_area


def _neg_soft_score(points, t=0.01):
    s = _soft_score(points, t)
    if s <= 0:
        return 1e10
    return -s


def _seed_regular():
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack([np.cos(ang), np.sin(ang)])


def _seed_two_ring(n_out=7, r_in=0.5, a_off=0.0):
    n_in = N - n_out
    a_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
    a_in = np.linspace(a_off, 2 * np.pi + a_off, n_in, endpoint=False)
    outer = np.column_stack([np.cos(a_out), np.sin(a_out)])
    inner = np.column_stack([r_in * np.cos(a_in), r_in * np.sin(a_in)])
    return np.vstack([outer, inner])


def _seed_three_ring(n1=5, n2=4, n3=4, r2=0.6, r3=0.3, o2=0.0, o3=0.0):
    a1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
    a2 = np.linspace(o2, 2 * np.pi + o2, n2, endpoint=False)
    a3 = np.linspace(o3, 2 * np.pi + o3, n3, endpoint=False)
    pts = np.vstack([
        np.column_stack([np.cos(a1), np.sin(a1)]),
        np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)]),
        np.column_stack([r3 * np.cos(a3), r3 * np.sin(a3)]),
    ])
    return pts


def _seed_quasi_regular():
    """Slightly perturbed regular polygon."""
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    r = 1.0 + 0.05 * np.sin(2 * ang + 0.3)
    return np.column_stack([r * np.cos(ang), r * np.sin(ang)])


def _annealed_soft_minimize(x0, deadline, t0=0.1, t_end=0.001, max_iter=200):
    """
    Anneal the soft-min temperature while optimizing with Powell/Nelder-Mead.
    This helps escape local minima associated with specific triangle tie-breaks.
    """
    best_x = x0.copy()
    best_s = _score(best_x)
    
    n_params = len(x0)
    n_steps = max(3, min(max_iter, int((deadline - time.time()) / 2.0)))
    
    for step in range(n_steps):
        remaining = deadline - time.time()
        if remaining < 5:
            break
        
        # Anneal temperature
        progress = step / max(1, n_steps - 1)
        t = t0 * (t_end / t0) ** progress
        
        # Choose method based on remaining time
        if remaining > 30:
            method = 'Powell'
            opts = {'maxiter': min(100, int(remaining / 0.5)), 'xatol': 1e-8, 'fatol': 1e-8}
        else:
            method = 'Nelder-Mead'
            opts = {'maxiter': min(200, int(remaining / 0.1)), 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
        
        def obj(x):
            pts = x.reshape(N, 2)
            return _neg_soft_score(pts, t)
        
        try:
            res = minimize(obj, best_x, method=method, options=opts)
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s = s
                best_x = cand.copy()
        except Exception:
            pass
    
    # Final polish with true objective
    if time.time() < deadline - 3:
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True},
            )
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s = s
                best_x = cand.copy()
        except Exception:
            pass
    
    return best_x, best_s


def _differential_evolution_full(deadline, maxiter=50, popsize=15):
    """Run differential evolution on the full 26-dim space."""
    bounds = [(-2.0, 2.0)] * (2 * N)
    
    def obj(x):
        pts = x.reshape(N, 2)
        return _neg_score(pts)
    
    try:
        res = differential_evolution(
            obj, bounds,
            maxiter=maxiter, popsize=popsize,
            tol=1e-8, mutation=(0.3, 1.0), recombination=0.9,
            seed=42, polish=False,
            update_interval=1,
        )
        cand = res.x.reshape(N, 2)
        s = _score(cand)
        return cand, s
    except Exception:
        return None, 0.0


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin under 150s kill

    best_points = None
    best_score = 0.0

    def _update(pts):
        nonlocal best_points, best_score
        if pts is None or len(pts) != N:
            return
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()

    # ---------- Seed candidates ----------
    _update(_seed_regular())
    _update(_seed_two_ring(7, 0.5, 0.0))
    _update(_seed_two_ring(6, 0.6, np.pi / 6))
    _update(_seed_two_ring(8, 0.4, np.pi / 8))
    _update(_seed_three_ring(5, 4, 4, 0.6, 0.3, 0.0, 0.0))
    _update(_seed_three_ring(6, 4, 3, 0.6, 0.3, 0.0, 0.0))
    _update(_seed_quasi_regular())

    # ---------- Differential Evolution (global search) ----------
    if time.time() < deadline - 60:
        try:
            pts, s = _differential_evolution_full(deadline, maxiter=40, popsize=12)
            _update(pts)
        except Exception:
            pass

    # ---------- Annealed soft-min optimization from multiple seeds ----------
    seeds = [_seed_regular(), _seed_two_ring(7, 0.5, 0.0), _seed_two_ring(6, 0.6, np.pi/6),
             _seed_three_ring(5, 4, 4, 0.6, 0.3, 0.0, 0.0), _seed_quasi_regular()]
    
    if best_points is not None:
        seeds.append(best_points.copy())
    
    for seed in seeds:
        if time.time() > deadline - 15:
            break
        try:
            pts, s = _annealed_soft_minimize(seed.flatten(), deadline, t0=0.05, t_end=0.001, max_iter=8)
            _update(pts)
        except Exception:
            pass

    # ---------- Final polish on best ----------
    if best_points is not None and time.time() < deadline - 5:
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-13, 'fatol': 1e-13, 'adaptive': True},
            )
            _update(res.x.reshape(N, 2))
        except Exception:
            pass

    # ---------- Ultra-fine final polish ----------
    if best_points is not None and time.time() < deadline - 2:
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                best_points.flatten(),
                method='Powell',
                options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-12},
            )
            _update(res.x.reshape(N, 2))
        except Exception:
            pass

    if best_points is None:
        best_points = _seed_regular()

    return best_points
