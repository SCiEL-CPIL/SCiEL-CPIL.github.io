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


def _soft_min_value(values, temperature):
    if temperature <= 1e-12:
        return float(np.min(values))
    min_val = np.min(values)
    shifted = values - min_val
    log_sum = np.log(np.sum(np.exp(-shifted / temperature)))
    return min_val - temperature * log_sum


def _soft_score(points, temperature):
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    if np.any(areas < 1e-15):
        return 0.0
    soft_min = _soft_min_value(areas, temperature)
    if soft_min < 1e-15:
        return 0.0
    return soft_min / hull_area


def _neg_soft_score(points, temperature):
    s = _soft_score(points, temp)
    if s <= 0:
        return 1e10
    return -s


def _generate_ring_config(n_outer, n_inner, r_inner, offset_inner):
    """
    Generate points for a two-ring configuration.
    Outer: n_outer points at radius 1.
    Inner: n_inner points at radius r_inner.
    """
    if n_outer + n_inner != N:
        # Fill remaining with center points or random
        n_center = N - n_outer - n_inner
        pts = []
        a_out = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
        pts.extend(np.column_stack([np.cos(a_out), np.sin(a_out)]))
        
        a_in = np.linspace(offset_inner, 2 * np.pi + offset_inner, n_inner, endpoint=False)
        pts.extend(np.column_stack([r_inner * np.cos(a_in), r_inner * np.sin(a_in)]))
        
        # Add center points
        for _ in range(n_center):
            pts.append([0.0, 0.0])
            
        return np.array(pts)
    else:
        a_out = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
        a_in = np.linspace(offset_inner, 2 * np.pi + offset_inner, n_inner, endpoint=False)
        outer = np.column_stack([np.cos(a_out), np.sin(a_out)])
        inner = np.column_stack([r_inner * np.cos(a_in), r_inner * np.sin(a_in)])
        return np.vstack([outer, inner])


def _generate_three_ring_config(n1, n2, n3, r2, r3, off2, off3):
    a1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
    a2 = np.linspace(off2, 2 * np.pi + off2, n2, endpoint=False)
    a3 = np.linspace(off3, 2 * np.pi + off3, n3, endpoint=False)
    pts = np.vstack([
        np.column_stack([np.cos(a1), np.sin(a1)]),
        np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)]),
        np.column_stack([r3 * np.cos(a3), r3 * np.sin(a3)]),
    ])
    return pts


def _optimize_local(points, deadline):
    """
    Local optimization using soft-min annealing and hard-min polishing.
    """
    best_points = points.copy()
    best_score = _score(best_points)
    
    # Soft-min annealing
    temps = [0.05, 0.01, 0.001]
    for temp in temps:
        if time.time() > deadline - 10:
            break
        
        def obj(x):
            pts = x.reshape(N, 2)
            return _neg_soft_score(pts, temp)
        
        try:
            res = minimize(
                obj,
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12},
            )
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_score:
                best_score = s
                best_points = cand.copy()
        except Exception:
            pass
        
    # Hard-min polish
    if time.time() < deadline - 5:
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                best_points.flatten(),
                method='Powell',
                options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10},
            )
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_score:
                best_score = s
                best_points = cand.copy()
        except Exception:
            pass
            
    return best_points, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    np.random.seed(42)
    
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

    # 1. Regular 13-gon
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    _update(np.column_stack([np.cos(ang), np.sin(ang)]))
    
    # 2. Center + 12-gon
    a12 = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    outer12 = np.column_stack([np.cos(a12), np.sin(a12)])
    center = np.array([[0.0, 0.0]])
    _update(np.vstack([outer12, center]))
    
    # 3. Two-ring configurations (Global Search for parameters)
    # Parameters: n_outer, r_inner, offset
    # We'll test a few specific good ones and then optimize
    
    # 4. Three-ring configurations
    # 5+4+4, 6+4+3, 7+3+3, etc.
    
    # Generate a set of diverse seeds
    seeds = []
    
    # Two rings: 6 outer, 7 inner
    for r_in in [0.3, 0.5, 0.7]:
        for off in [0.0, np.pi/6, np.pi/3]:
            seeds.append(_generate_ring_config(6, 7, r_in, off))
            
    # Two rings: 7 outer, 6 inner
    for r_in in [0.3, 0.5, 0.7]:
        for off in [0.0, np.pi/7, np.pi/14]:
            seeds.append(_generate_ring_config(7, 6, r_in, off))
            
    # Three rings: 5+4+4
    seeds.append(_generate_three_ring_config(5, 4, 4, 0.6, 0.3, 0.0, 0.0))
    seeds.append(_generate_three_ring_config(5, 4, 4, 0.5, 0.25, np.pi/5, np.pi/4))
    
    # Three rings: 6+4+3
    seeds.append(_generate_three_ring_config(6, 4, 3, 0.6, 0.3, 0.0, 0.0))
    
    # Three rings: 7+3+3
    seeds.append(_generate_three_ring_config(7, 3, 3, 0.6, 0.3, 0.0, 0.0))
    
    # Evaluate seeds
    for seed in seeds:
        _update(seed)
        
    # Optimize the best seed found so far
    if best_points is not None and time.time() < deadline - 30:
        try:
            pts, s = _optimize_local(best_points, deadline - 20)
            _update(pts)
        except Exception:
            pass
            
    # Try optimizing a few other promising seeds
    # Specifically, the Center+12 and 6+7 ring
    candidate_seeds = [
        np.vstack([np.column_stack([np.cos(np.linspace(0, 2*np.pi, 12, endpoint=False)), np.sin(np.linspace(0, 2*np.pi, 12, endpoint=False))]), np.array([[0,0]])]),
        _generate_ring_config(6, 7, 0.5, np.pi/6),
    ]
    
    for seed in candidate_seeds:
        if time.time() > deadline - 20:
            break
        try:
            pts, s = _optimize_local(seed, deadline - 10)
            _update(pts)
        except Exception:
            pass
            
    # Final Polish
    if best_points is not None and time.time() < deadline - 5:
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True},
            )
            _update(res.x.reshape(N, 2))
        except Exception:
            pass
            
    if best_points is None:
        best_points = np.column_stack([np.cos(np.linspace(0, 2*np.pi, N, endpoint=False)), np.sin(np.linspace(0, 2*np.pi, N, endpoint=False))])
        
    return best_points
