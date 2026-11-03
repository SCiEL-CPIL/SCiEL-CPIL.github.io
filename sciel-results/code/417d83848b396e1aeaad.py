# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 14

def _min_triangle_area_vec(points):
    """Vectorized minimum triangle area for N points."""
    n = len(points)
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    for i in range(n - 2):
        xi, yi = x[i], y[i]
        dx = x[i+1:] - xi
        dy = y[i+1:] - yi
        for j in range(1, n - i - 1):
            cross = dx[j-1] * dy[j:] - dx[j:] * dy[j-1]
            local_min = np.min(np.abs(cross))
            if local_min < min_area:
                min_area = local_min
    return 0.5 * min_area

def _hull_area_fast(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    hull_area = _hull_area_fast(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area_vec(points)
    return min_area / hull_area

def _all_triangle_areas(points):
    """Return array of all triangle areas (N choose 3)."""
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    areas_list = []
    for i in range(n - 2):
        xi, yi = x[i], y[i]
        dx = x[i+1:] - xi
        dy = y[i+1:] - yi
        for j in range(1, n - i - 1):
            cross = dx[j-1] * dy[j:] - dx[j:] * dy[j-1]
            areas_list.append(0.5 * np.abs(cross))
    if not areas_list:
        return np.array([0.0])
    return np.concatenate(areas_list)

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

# ---------- Cartesian Symmetric Parameterization ----------
# 7 independent points (x, y) pairs. The other 7 are their negatives.
# Total 14 parameters.

def _params_to_points_cart(params):
    """Decode 14-parameter vector [x1, y1, ..., x7, y7] to 14 points."""
    # params shape (14,)
    # Reshape to (7, 2)
    half = params.reshape(7, 2)
    # Ensure non-degenerate: if any point is too close to origin, push it out?
    # Or just let the optimizer handle it.
    return np.vstack([half, -half])

def _objective_sym_cart(params):
    """Hard objective for cartesian symmetric parameters."""
    if np.any(~np.isfinite(params)):
        return 1e10
    points = _params_to_points_cart(params)
    if np.any(~np.isfinite(points)):
        return 1e10
    min_area = _min_triangle_area_vec(points)
    if min_area < 1e-12:
        return 1e10
    hull_area = _hull_area_fast(points)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / min_area

def _objective_sym_cart_soft(params, beta=200.0):
    """Soft objective for cartesian symmetric parameters."""
    if np.any(~np.isfinite(params)):
        return 1e10
    points = _params_to_points_cart(params)
    if np.any(~np.isfinite(points)):
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
    return hull_area / soft_min_val

def _generate_cart_inits():
    """Generate initial parameter vectors for cartesian symmetric optimization."""
    inits = []
    # Regular heptagon in cartesian coords
    base_angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
    for rot in [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 1.7, 2.3, 2.8]:
        a = base_angles + rot
        r = np.ones(7)
        x = r * np.cos(a)
        y = r * np.sin(a)
        params = np.concatenate([x, y])
        inits.append(params)
        
    # Jittered regular
    rng = np.random.default_rng(42)
    for seed in range(20):
        rng = np.random.default_rng(seed * 11 + 5)
        a = base_angles + rng.normal(0, 0.1, 7)
        r = 1.0 + rng.normal(0, 0.1, 7)
        r = np.maximum(r, 0.5)
        x = r * np.cos(a)
        y = r * np.sin(a)
        params = np.concatenate([x, y])
        inits.append(params)
        
    # Random
    for seed in range(10):
        rng = np.random.default_rng(seed * 7 + 3)
        x = rng.uniform(-1.5, 1.5, 7)
        y = rng.uniform(-1.5, 1.5, 7)
        # Ensure no point is too close to 0 to avoid degeneracy in hull area calc if all collapse
        # But we have -p as well, so if p is close to 0, the pair is close.
        # The min triangle area will be small. The optimizer should push them out.
        params = np.concatenate([x, y])
        inits.append(params)
        
    return inits

def _nm(x0, deadline, max_iter=3000, obj=_objective_sym_cart):
    remaining = deadline - time.time()
    if remaining < 2.0:
        return None
    try:
        res = minimize(obj, x0, method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
        return res
    except Exception:
        return None

def _lbfgs(x0, deadline, max_iter=5000, obj=_objective_sym_cart_soft):
    remaining = deadline - time.time()
    if remaining < 5.0:
        return None
    try:
        res = minimize(obj, x0, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-11})
        return res
    except Exception:
        return None

def _de_cart(bounds, deadline, max_iter=300, popsize=20):
    """Differential evolution on cartesian symmetric parameters."""
    remaining = deadline - time.time()
    if remaining < 10.0:
        return None
    try:
        res = differential_evolution(_objective_sym_cart, bounds, maxiter=max_iter,
                                      popsize=popsize, tol=1e-11, atol=1e-13,
                                      seed=42, polish=True, init='latinhypercube')
        return res
    except Exception:
        return None

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    best_score = 0.0
    best_pts = None
    best_params = None

    def _consider_from_params(params):
        nonlocal best_score, best_pts, best_params
        points = _params_to_points_cart(params)
        if points.shape != (N, 2) or np.any(~np.isfinite(points)):
            return
        s = _score(points)
        if s > best_score:
            best_score = s
            best_pts = points.copy()
            best_params = params.copy()

    # Phase 1: Differential Evolution in Cartesian Symmetric Space
    # Bounds: x, y in [-2, 2] is safe.
    bounds = [(-2.0, 2.0)] * 14
    res = _de_cart(bounds, deadline, max_iter=200, popsize=30)
    if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
        _consider_from_params(res.x)

    # Phase 2: Nelder-Mead from diverse Cartesian starts
    inits = _generate_cart_inits()
    for x0 in inits:
        if time.time() > deadline - 40:
            break
        res = _nm(x0, deadline, max_iter=2000, obj=_objective_sym_cart)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            _consider_from_params(res.x)

    # Phase 3: L-BFGS on soft objective
    if best_params is not None and time.time() < deadline - 30:
        res = _lbfgs(best_params, deadline, max_iter=4000, obj=_objective_sym_cart_soft)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            _consider_from_params(res.x)

    # Phase 4: Powell polish
    if best_params is not None and time.time() < deadline - 20:
        try:
            res = minimize(_objective_sym_cart, best_params, method='Powell',
                           options={'maxiter': 2000, 'xtol': 1e-12, 'ftol': 1e-14})
            if np.isfinite(res.fun) and res.fun < 1e9:
                _consider_from_params(res.x)
        except Exception:
            pass

    # Phase 5: Final NM polish
    if best_params is not None and time.time() < deadline - 10:
        res = _nm(best_params, deadline, max_iter=5000, obj=_objective_sym_cart)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            _consider_from_params(res.x)

    # Fallback: regular 14-gon
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
