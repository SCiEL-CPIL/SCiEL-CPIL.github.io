"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

# ---------- Core geometry ----------

def _min_triangle_area_vec(points):
    """Vectorized minimum triangle area for N points."""
    n = len(points)
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    for i in range(n):
        for j in range(i + 1, n):
            if j + 1 >= n:
                continue
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = dx_ji * dy_ki - dx_ki * dy_ji
            areas = 0.5 * np.abs(cross)
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    return min_area

def _hull_area_fast(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Returns hull_area / min_triangle_area (the actual evaluator score)."""
    hull_area = _hull_area_fast(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area_vec(points)
    if min_area < 1e-10:
        return 1e10 # Large score if min area is tiny
    return hull_area / min_area

def _all_triangle_areas(points):
    """Return array of all triangle areas."""
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    areas_list = []
    for i in range(n):
        for j in range(i + 1, n):
            if j + 1 >= n:
                continue
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = dx_ji * dy_ki - dx_ki * dy_ji
            areas_list.append(0.5 * np.abs(cross))
    if not areas_list:
        return np.array([0.0])
    return np.concatenate(areas_list)

def _objective_hard(points_flat):
    """Minimize min_triangle_area / hull_area."""
    points = points_flat.reshape(N, 2)
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return 1e10
    min_area = _min_triangle_area_vec(points)
    if min_area < 1e-12:
        return 1e10
    hull_area = _hull_area_fast(points)
    if hull_area < 1e-10:
        return 1e10
    return min_area / hull_area

def _objective_soft(points_flat, beta=100.0):
    """Smooth soft-min objective: soft_min(triangle areas) / hull_area."""
    points = points_flat.reshape(N, 2)
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return 1e10
    areas = _all_triangle_areas(points)
    if len(areas) == 0 or np.min(areas) < 1e-12:
        return 1e10
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min_val = -log_sum / beta
    if soft_min_val < 1e-12:
        return 1e10
    hull_area = _hull_area_fast(points)
    if hull_area < 1e-10:
        return 1e10
    return soft_min_val / hull_area

def _normalize_unit_area(points):
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area > 1e-10:
            scale = np.sqrt(1.0 / hull_area)
            return points * scale
    except Exception:
        pass
    xs = points[:, 0]
    ys = points[:, 1]
    w = xs.max() - xs.min()
    h = ys.max() - ys.min()
    if w > 1e-10 and h > 1e-10:
        scale = 1.0 / np.sqrt(w * h)
        return points * scale
    return points

def _nm(x0, deadline, max_iter=2500, obj=_objective_hard):
    remaining = deadline - time.time()
    if remaining < 2.0:
        return None
    try:
        res = minimize(obj, x0, method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
        return res
    except Exception:
        return None

def _lbfgs(x0, deadline, max_iter=5000, obj=_objective_soft, beta=100.0):
    remaining = deadline - time.time()
    if remaining < 5.0:
        return None
    try:
        res = minimize(lambda x: obj(x, beta), x0, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10})
        return res
    except Exception:
        return None

def _make_inits():
    inits = []
    rng = np.random.default_rng(2024)
    
    # 1. Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    reg = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(reg.flatten())

    # 2. Distorted: 3 points close, others on circle
    # Create a base regular polygon
    # Move points 0, 1, 2 closer to each other
    for eps in [0.1, 0.3, 0.5]:
        pts = reg.copy()
        # Cluster 0,1,2 around (1,0)
        # Original 0: (1,0)
        # Original 1: (cos(2pi/14), sin(2pi/14))
        # Original 2: (cos(4pi/14), sin(4pi/14))
        # Move them to be close to each other on the hull
        # This is tricky to do manually. Let's just use random perturbations.
        pass

    # 3. Random perturbations of regular polygon
    for js in [0.05, 0.1, 0.2, 0.5]:
        pts = reg + rng.normal(0, js, size=(N, 2))
        inits.append(pts.flatten())

    # 4. "Dumbbell" like: 7 points on left, 7 on right, stretched
    # Actually, just stretch the regular polygon
    for sx, sy in [(2, 0.5), (0.5, 2), (3, 0.3), (0.3, 3)]:
        pts = reg.copy()
        pts[:, 0] *= sx
        pts[:, 1] *= sy
        inits.append(pts.flatten())

    # 5. Random points in a disk
    for seed in range(10):
        rs = np.random.default_rng(seed)
        ang = np.sort(rs.uniform(0, 2 * np.pi, N))
        rad = rs.uniform(0.5, 1.0, N)
        pts = np.column_stack([rad * np.cos(ang), rad * np.sin(ang)])
        inits.append(pts.flatten())

    return inits

def construct():
    start_time = time.time()
    deadline = start_time + 142.0
    
    best_score = 0.0
    best_pts = None

    def _consider(pts):
        nonlocal best_score, best_pts
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (N, 2) or np.any(~np.isfinite(pts)):
            return
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()

    inits = _make_inits()

    # Phase 1: L-BFGS on soft objective (minimizing min_area/hull_area)
    for x0 in inits:
        if time.time() > deadline - 60:
            break
        # Use increasing beta
        for beta in [10, 50, 200]:
            if time.time() > deadline - 10:
                break
            res = _lbfgs(x0, deadline, max_iter=1000, obj=_objective_soft, beta=beta)
            if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
                _consider(res.x.reshape(N, 2))
                x0 = res.x # Use result for next beta

    # Phase 2: Nelder-Mead on hard objective
    if best_pts is not None:
        for _ in range(3): # A few restarts from best
            if time.time() > deadline - 30:
                break
            res = _nm(best_pts.flatten(), deadline, max_iter=2000, obj=_objective_hard)
            if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
                _consider(res.x.reshape(N, 2))
                best_pts = res.x.reshape(N, 2) # Update best for next restart

    # Phase 3: Final polish
    if best_pts is not None and time.time() < deadline - 10:
        res = _nm(best_pts.flatten(), deadline, max_iter=3000, obj=_objective_hard)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            _consider(res.x.reshape(N, 2))

    if best_pts is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])

    final_points = _normalize_unit_area(best_pts)
    
    # Safety check
    try:
        hull = ConvexHull(final_points)
        if hull.volume < 1e-10:
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            final_points = _normalize_unit_area(np.column_stack([np.cos(angles), np.sin(angles)]))
    except Exception:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        final_points = _normalize_unit_area(np.column_stack([np.cos(angles), np.sin(angles)]))

    return final_points
# EVOLVE-BLOCK-END
