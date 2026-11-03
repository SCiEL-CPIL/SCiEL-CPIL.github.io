import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14
DEADLINE_S = 145.0  # safety margin within 150s

# Precompute all triangle index triples
_indices = np.arange(N)
IDX_I, IDX_J, IDX_K = np.meshgrid(_indices, _indices, _indices, indexing='ij')
_mask = (IDX_I < IDX_J) & (IDX_J < IDX_K)
IDX_I = IDX_I[_mask].flatten()
IDX_J = IDX_J[_mask].flatten()
IDX_K = IDX_K[_mask].flatten()
N_TRI = len(IDX_I)

def _all_areas(points):
    pi = points[IDX_I]
    pj = points[IDX_J]
    pk = points[IDX_K]
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    return 0.5 * np.abs(v1x * v2y - v1y * v2x)

def _hull_area(points):
    try:
        if np.allclose(points[:, 0], points[0, 0]) or np.allclose(points[:, 1], points[0, 1]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    ha = _hull_area(points)
    if ha < 1e-10:
        return 0.0
    return np.min(_all_areas(points)) / ha

def _time_left(start):
    return DEADLINE_S - (time.time() - start)

def _regular_polygon(n=N, radius=1.0, rot=0.0):
    angles = np.linspace(0, 2*np.pi, n, endpoint=False) + rot
    return np.column_stack((radius*np.cos(angles), radius*np.sin(angles)))

def _soft_min(areas, beta=50.0):
    # Smooth approximation of min via softmax-like
    min_a = np.min(areas)
    exp_shifted = np.exp(beta * (areas - min_a))
    return min_a - np.log(np.sum(exp_shifted)) / beta

def _objective_soft(flat, beta=50.0):
    pts = flat.reshape(N, 2)
    ha = _hull_area(pts)
    if ha < 1e-10:
        return 1e6
    areas = _all_areas(pts)
    return -_soft_min(areas, beta) / ha

def _objective_hard(flat):
    pts = flat.reshape(N, 2)
    return -_score(pts)

def _slsqp_optimize(x0, obj, start, time_budget=10.0):
    """Run SLSQP with time guard."""
    end_time = time.time() + time_budget
    def obj_guarded(flat):
        if time.time() > end_time:
            return 1e10
        return obj(flat)
    try:
        res = minimize(obj_guarded, x0, method='SLSQP',
                       options={'maxiter': 200, 'ftol': 1e-12, 'disp': False})
        if res.fun < 1e9:
            return res.x
    except Exception:
        pass
    return x0

def _nm_optimize(x0, obj, start, time_budget=8.0):
    """Run Nelder-Mead with time guard."""
    end_time = time.time() + time_budget
    def obj_guarded(flat):
        if time.time() > end_time:
            return 1e10
        return obj(flat)
    try:
        res = minimize(obj_guarded, x0, method='Nelder-Mead',
                       options={'maxiter': 300, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
        if res.fun < 1e9:
            return res.x
    except Exception:
        pass
    return x0

def _random_in_circle(n=N, seed=0):
    rng = np.random.RandomState(seed)
    r = np.sqrt(rng.uniform(0, 1, n))
    th = rng.uniform(0, 2*np.pi, n)
    return np.column_stack((r*np.cos(th), r*np.sin(th)))

def _jittered_polygon(n=N, seed=0, max_jitter=0.15):
    rng = np.random.RandomState(seed)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    # Random radii
    radii = 1.0 + rng.uniform(-max_jitter, max_jitter, n)
    # Random angle perturbations
    angles = angles + rng.uniform(-0.1, 0.1, n)
    angles = np.sort(angles)  # keep in order
    # Re-normalize angles to span 2pi
    angles = angles - angles[0]
    angles = angles / angles[-1] * 2 * np.pi
    pts = np.column_stack((radii*np.cos(angles), radii*np.sin(angles)))
    return pts

def construct():
    start_time = time.time()
    best_points = None
    best_score = -np.inf

    def update_best(pts):
        nonlocal best_points, best_score
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()

    # --- Generate diverse starting points ---
    starts = []
    # Regular polygon
    starts.append(_regular_polygon())
    # Rotated regular polygons
    for rot in [np.pi/14, np.pi/7, 2*np.pi/14]:
        starts.append(_regular_polygon(rot=rot))
    # Random points in circle
    for seed in range(15):
        starts.append(_random_in_circle(seed=seed))
    # Jittered polygons
    for seed in range(10):
        starts.append(_jittered_polygon(seed=seed, max_jitter=0.2))
    # Two-ring structures
    for r1, r2 in [(0.6, 1.0), (0.7, 1.0), (0.8, 1.0)]:
        n1 = 7
        n2 = 7
        a1 = np.linspace(0, 2*np.pi, n1, endpoint=False)
        a2 = np.linspace(0, 2*np.pi, n2, endpoint=False) + np.pi/n2
        pts1 = np.column_stack((r1*np.cos(a1), r1*np.sin(a1)))
        pts2 = np.column_stack((r2*np.cos(a2), r2*np.sin(a2)))
        starts.append(np.vstack([pts1, pts2]))

    # --- Multi-start SLSQP + Nelder-Mead refinement ---
    for i, st in enumerate(starts):
        if _time_left(start_time) < 8.0:
            break

        # Quick score
        s0 = _score(st)
        if s0 > best_score:
            best_score = s0
            best_points = st.copy()

        # Phase A: Soft SLSQP
        if _time_left(start_time) < 5.0:
            break
        x0 = st.flatten()
        x_slsqp = _slsqp_optimize(x0, _objective_soft, start_time, time_budget=6.0)
        pts_a = x_slsqp.reshape(N, 2)
        s_a = _score(pts_a)
        if s_a > best_score:
            best_score = s_a
            best_points = pts_a.copy()

        # Phase B: Hard SLSQP from best of (st, pts_a)
        if _time_left(start_time) < 5.0:
            break
        x_b0 = pts_a.flatten() if s_a > s0 else st.flatten()
        x_slsqp2 = _slsqp_optimize(x_b0, _objective_hard, start_time, time_budget=5.0)
        pts_b = x_slsqp2.reshape(N, 2)
        s_b = _score(pts_b)
        if s_b > best_score:
            best_score = s_b
            best_points = pts_b.copy()

        # Phase C: Nelder-Mead polish on best so far
        if _time_left(start_time) < 4.0:
            break
        x_c0 = pts_b.flatten() if s_b >= max(s_a, s0) else (pts_a.flatten() if s_a >= s0 else st.flatten())
        x_nm = _nm_optimize(x_c0, _objective_hard, start_time, time_budget=4.0)
        pts_c = x_nm.reshape(N, 2)
        s_c = _score(pts_c)
        if s_c > best_score:
            best_score = s_c
            best_points = pts_c.copy()

    # --- Final global polish if time allows ---
    if best_points is not None and _time_left(start_time) > 10.0:
        x0 = best_points.flatten()
        # Another SLSQP with tighter tolerance
        x_final = _slsqp_optimize(x0, _objective_hard, start_time, time_budget=_time_left(start_time) - 2.0)
        pts_final = x_final.reshape(N, 2)
        s_final = _score(pts_final)
        if s_final > best_score:
            best_points = pts_final.copy()

    if best_points is None:
        best_points = _regular_polygon()

    return best_points
