"""Heilbronn convex regions n=14 – centrally-symmetric DE + local refinement."""
N = 14
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

# ---------- geometry helpers ----------

def _min_triangle_area(pts):
    """Return the minimum triangle area among all C(n,3) triples."""
    n = len(pts)
    x = pts[:, 0]; y = pts[:, 1]
    best = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            if j + 1 >= n:
                continue
            dxj = x[j] - x[i]
            dyj = y[j] - y[i]
            cross = dxj * (y[j+1:] - y[i]) - (x[j+1:] - x[i]) * dyj
            areas = 0.5 * np.abs(cross)
            m = np.min(areas)
            if m < best:
                best = m
    return best

def _hull_area(pts):
    try:
        return ConvexHull(pts).volume
    except Exception:
        return 0.0

def _score(pts):
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 0.0
    ma = _min_triangle_area(pts)
    return ma / ha

# ---------- centrally-symmetric parameterization ----------
# 7 antipodal pairs: points k and k+7 are negatives of each other.
# Parameters: 7 angles theta_k in [0, pi) and 7 radii r_k in [0.05, 2.5].
# Points: p_k = r_k * (cos theta_k, sin theta_k), p_{k+7} = -p_k.

def _params_to_points(params):
    """14-dim param vector -> (14,2) point array (centrally symmetric)."""
    thetas = params[:7]
    radii = params[7:]
    pts = np.zeros((N, 2))
    for k in range(7):
        pts[k, 0] = radii[k] * np.cos(thetas[k])
        pts[k, 1] = radii[k] * np.sin(thetas[k])
        pts[k + 7] = -pts[k]
    return pts

def _objective(params):
    """Minimize hull_area / min_triangle_area (lower is better)."""
    pts = _params_to_points(params)
    ma = _min_triangle_area(pts)
    if ma < 1e-12:
        return 1e10
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 1e10
    return ha / ma

def _objective_soft(params, beta=150.0):
    """Soft-min version for smoother L-BFGS-B optimization."""
    pts = _params_to_points(params)
    n = len(pts)
    x = pts[:, 0]; y = pts[:, 1]
    areas_list = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            if j + 1 >= n:
                continue
            dxj = x[j] - x[i]
            dyj = y[j] - y[i]
            cross = dxj * (y[j+1:] - y[i]) - (x[j+1:] - x[i]) * dyj
            tri = 0.5 * np.abs(cross)
            if len(tri) > 0:
                areas_list.append(tri)
    if not areas_list:
        return 1e10
    areas = np.concatenate(areas_list)
    if np.min(areas) < 1e-12:
        return 1e10
    z = -beta * areas
    max_z = np.max(z)
    log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min = -log_sum / beta
    if soft_min < 1e-12:
        return 1e10
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 1e10
    return ha / soft_min

# ---------- bounds for DE ----------
THETA_LO = 0.0
THETA_HI = np.pi
R_LO = 0.05
R_HI = 2.5

def _make_bounds():
    bounds = []
    for _ in range(7):
        bounds.append((THETA_LO, THETA_HI))
    for _ in range(7):
        bounds.append((R_LO, R_HI))
    return bounds

# ---------- normalization ----------

def _normalize_unit_area(pts):
    try:
        ha = ConvexHull(pts).volume
        if ha > 1e-10:
            return pts * np.sqrt(1.0 / ha)
    except Exception:
        pass
    xs = pts[:, 0]; ys = pts[:, 1]
    w = xs.max() - xs.min()
    h = ys.max() - ys.min()
    if w > 1e-10 and h > 1e-10:
        return pts / np.sqrt(w * h)
    return pts

# ---------- initial guesses for DE ----------

def _regular_init():
    """Regular 14-gon as symmetric params."""
    params = np.zeros(14)
    for k in range(7):
        params[k] = k * np.pi / 7.0          # theta_k
        params[7 + k] = 1.0                   # r_k = 1
    return params

# ---------- local refinement ----------

def _refine_lbfgs(params, deadline, max_iter=3000):
    remaining = deadline - time.time()
    if remaining < 4.0:
        return params
    try:
        res = minimize(
            _objective_soft, params, method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-11, 'maxfun': 50000}
        )
        return res.x
    except Exception:
        return params

def _refine_nm(params, deadline, max_iter=5000):
    remaining = deadline - time.time()
    if remaining < 4.0:
        return params
    try:
        res = minimize(
            _objective, params, method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        return res.x
    except Exception:
        return params

def _refine_powell(params, deadline, max_iter=3000):
    remaining = deadline - time.time()
    if remaining < 4.0:
        return params
    try:
        res = minimize(
            _objective, params, method='Powell',
            options={'maxiter': max_iter, 'xtol': 1e-12, 'ftol': 1e-14}
        )
        return res.x
    except Exception:
        return params

# ---------- main ----------

def construct():
    start = time.time()
    deadline = start + 142.0   # 8 s safety margin before 150 s kill

    bounds = _make_bounds()
    best_params = _regular_init()
    best_obj = _objective(best_params)

    # --- Phase 1: Differential Evolution (global search) ---
    remaining = deadline - time.time()
    if remaining > 50.0:
        try:
            de_res = differential_evolution(
                _objective,
                bounds=bounds,
                seed=42,
                maxiter=300,
                popsize=20,
                tol=1e-10,
                mutation=(0.3, 1.0),
                recombination=0.8,
                polish=False,       # we do our own polish
                init='latinhypercube',
                updating='deferred',
                seed=42,
            )
            if de_res.fun < best_obj:
                best_obj = de_res.fun
                best_params = de_res.x.copy()
        except Exception:
            pass

    # --- Phase 2: L-BFGS-B on soft objective ---
    remaining = deadline - time.time()
    if remaining > 15.0:
        p2 = _refine_lbfgs(best_params, deadline, max_iter=2000)
        obj2 = _objective(p2)
        if obj2 < best_obj:
            best_obj = obj2
            best_params = p2.copy()

    # --- Phase 3: Nelder-Mead on hard objective ---
    remaining = deadline - time.time()
    if remaining > 12.0:
        p3 = _refine_nm(best_params, deadline, max_iter=5000)
        obj3 = _objective(p3)
        if obj3 < best_obj:
            best_obj = obj3
            best_params = p3.copy()

    # --- Phase 4: Powell on hard objective ---
    remaining = deadline - time.time()
    if remaining > 10.0:
        p4 = _refine_powell(best_params, deadline, max_iter=2000)
        obj4 = _objective(p4)
        if obj4 < best_obj:
            best_obj = obj4
            best_params = p4.copy()

    # --- Phase 5: Second L-BFGS-B pass (tighter) ---
    remaining = deadline - time.time()
    if remaining > 8.0:
        p5 = _refine_lbfgs(best_params, deadline, max_iter=3000)
        obj5 = _objective(p5)
        if obj5 < best_obj:
            best_obj = obj5
            best_params = p5.copy()

    # --- Phase 6: Final NM polish ---
    remaining = deadline - time.time()
    if remaining > 5.0:
        p6 = _refine_nm(best_params, deadline, max_iter=3000)
        obj6 = _objective(p6)
        if obj6 < best_obj:
            best_obj = obj6
            best_params = p6.copy()

    # --- Convert to points and normalize ---
    final_pts = _params_to_points(best_params)
    final_pts = _normalize_unit_area(final_pts)

    # Safety checks
    try:
        ha = ConvexHull(final_pts).volume
        if ha < 1e-10:
            raise ValueError("degenerate hull")
    except Exception:
        # Fallback: regular 14-gon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        final_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        final_pts = _normalize_unit_area(final_pts)

    if np.any(np.isnan(final_pts)) or np.any(np.isinf(final_pts)):
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        final_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        final_pts = _normalize_unit_area(final_pts)

    return final_pts
