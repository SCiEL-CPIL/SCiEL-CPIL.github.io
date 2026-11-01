import numpy as np
import scipy.spatial
import scipy.optimize
import time
from itertools import combinations

N = 14
N_TRI = N * (N - 1) * (N - 2) // 6

# Precompute triangle index arrays
def _build_tri_indices(n):
    idx = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                idx.append((i, j, k))
    a = np.array(idx, dtype=np.int32)
    return a[:, 0], a[:, 1], a[:, 2]

_I, _J, _K = _build_tri_indices(N)

def _min_triangle_area(pts):
    pi, pj, pk = pts[_I], pts[_J], pts[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross).min()

def _hull_area(pts):
    try:
        h = scipy.spatial.ConvexHull(pts)
        return h.volume
    except Exception:
        return 0.0

def _score(pts):
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 0.0
    return _min_triangle_area(pts) / ha

def _regular_polygon(n, r=1.0, rot=0.0):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False) + rot
    return np.column_stack([r * np.cos(th), r * np.sin(th)])

# ------------------------------------------------------------------
# Convex-hull area with analytic subgradient
# ------------------------------------------------------------------
def _hull_area_and_grad(pts):
    """Return (area, grad) where grad[i] is the subgradient of area w.r.t. pts[i].
    Interior points get zero gradient (area is flat in their direction)."""
    n = len(pts)
    try:
        h = scipy.spatial.ConvexHull(pts)
    except Exception:
        return 0.0, np.zeros((n, 2))
    hidx = h.simplices[:, 0]  # CCW-ordered hull vertex indices
    m = len(hidx)
    if m < 3:
        return 0.0, np.zeros((n, 2))
    # Shoelace area
    px = pts[hidx, 0]
    py = pts[hidx, 1]
    area = 0.5 * np.abs(np.sum(px * np.roll(py, -1) - np.roll(px, -1) * py))
    # dA/dx_i = 0.5 * (y_{i+1} - y_{i-1}); dA/dy_i = 0.5 * (x_{i-1} - x_{i+1})
    gx = 0.5 * (np.roll(py, -1) - np.roll(py, 1))
    gy = 0.5 * (np.roll(px, 1) - np.roll(px, -1))
    grad = np.zeros((n, 2))
    grad[hidx, 0] = gx
    grad[hidx, 1] = gy
    return area, grad

# ------------------------------------------------------------------
# Objective: maximize t  s.t.  t <= 0.5|cross| for all triangles,  area(hull) <= 1
# We minimize -t.
# ------------------------------------------------------------------
def _superlevel_obj(x):
    """x = [p_0x, p_0y, ..., p_{n-1}x, p_{n-1}y, t]"""
    t = x[-1]
    pts = x[:-1].reshape(-1, 2)
    pi, pj, pk = pts[_I], pts[_J], pts[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    # violation: max(t - areas, 0)
    viol = np.maximum(t - areas, 0.0)
    # hull area constraint: area - 1 <= 0
    ha, _ = _hull_area_and_grad(pts)
    ha_viol = max(ha - 1.0, 0.0)
    # penalty
    return -t + 1e3 * (np.sum(viol) + ha_viol)

def _superlevel_grad(x):
    t = x[-1]
    pts = x[:-1].reshape(-1, 2)
    pi, pj, pk = pts[_I], pts[_J], pts[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    signs = np.sign(cross)
    # d(areas)/d(pts)
    d_cross_di_x = -(v1[:, 1] + v2[:, 1])
    d_cross_di_y = (v1[:, 0] + v2[:, 0])
    d_cross_dj_x = v2[:, 1]
    d_cross_dj_y = -v2[:, 0]
    d_cross_dk_x = -v1[:, 1]
    d_cross_dk_y = v1[:, 0]
    w = 0.5 * signs * (t > areas)  # only active where t > area
    grad_pts = np.zeros_like(pts)
    np.add.at(grad_pts, _I, w[:, None] * np.column_stack([d_cross_di_x, d_cross_di_y]))
    np.add.at(grad_pts, _J, w[:, None] * np.column_stack([d_cross_dj_x, d_cross_dj_y]))
    np.add.at(grad_pts, _K, w[:, None] * np.column_stack([d_cross_dk_x, d_cross_dk_y]))
    ha, hgrad = _hull_area_and_grad(pts)
    ha_active = ha > 1.0
    grad = -grad_pts.copy()  # because we minimize -t + penalty
    grad[-1] = -1.0 + 1e3 * (-np.sum(w))  # d(-t)/dt = -1, plus penalty derivative
    # penalty gradient on pts: 1e3 * (grad of sum viol + grad of ha_viol)
    pen_pts = 1e3 * (grad_pts + (ha_active * hgrad))
    grad[:-1] = -1.0 * (-grad_pts)  # careful: obj = -t + 1e3*(sum viol + ha_viol)
    # d obj / d pts = 1e3 * d(sum viol)/d pts + 1e3 * d(ha_viol)/d pts
    # d(sum viol)/d pts = grad_pts (already computed as d(areas)/d pts where active, with sign)
    # Actually: d(t - areas)/d pts = -d(areas)/d pts = -grad_pts_signed
    # Let me recompute: viol_i = max(t - areas_i, 0). d viol_i / d pts = -d(areas_i)/d pts if active.
    # So d(sum viol)/d pts = -sum over active of d(areas_i)/d pts
    # d(areas_i)/d pts_i = 0.5 * sign_i * d(cross_i)/d pts_i
    # So d(sum viol)/d pts = -sum over active of 0.5*sign_i*d(cross_i)/d pts = -grad_pts (as computed)
    # Wait, grad_pts above is sum of 0.5*sign_i*(active_i)*d(cross_i)/d pts = d(sum active areas)/d pts
    # So d(sum viol)/d pts = -grad_pts
    grad[:-1] = 1e3 * (-grad_pts + (ha_active * hgrad))
    grad[-1] = -1.0 + 1e3 * (-np.sum(w) + 0.0)  # d(t)/dt=1 so d(-t)/dt=-1; d(viol)/dt = 1 per active
    # d(sum viol)/dt = count of active = sum(w * 2) ... let me just use count
    active_count = np.sum((t > areas).astype(float))
    grad[-1] = -1.0 + 1e3 * active_count
    return grad

def _superlevel_constraints(x):
    pts = x[:-1].reshape(-1, 2)
    t = x[-1]
    pi, pj, pk = pts[_I], pts[_J], pts[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    # t - areas <= 0
    return areas - t

def _optimize_superlevel(init_pts, t0, maxiter=500, time_budget=10.0):
    x0 = np.concatenate([init_pts.flatten(), [t0]])
    try:
        res = scipy.optimize.minimize(
            _superlevel_obj, x0, jac=_superlevel_grad,
            method='SLSQP',
            options={'maxiter': maxiter, 'ftol': 1e-10}
        )
        return res.x
    except Exception:
        return x0

def construct():
    t_start = time.time()
    DEADLINE = t_start + 140.0  # 10s safety margin
    rng = np.random.RandomState(7)

    best_score = 0.0
    best_pts = None

    def _update(pts):
        nonlocal best_score, best_pts
        ha = _hull_area(pts)
        if ha < 1e-12:
            return 0.0
        sc = _min_triangle_area(pts) / ha
        if sc > best_score:
            best_score = sc
            best_pts = pts.copy()
        return sc

    def _time_ok(margin=5.0):
        return time.time() < DEADLINE - margin

    # --- Phase 1: parametric families as starting points ---
    starts = []

    # Regular polygon
    reg = _regular_polygon(N, 1.0)
    starts.append(reg)

    # Two-ring: outer regular n1, inner regular n2, with rotation offset
    for n1, n2 in [(7, 7), (6, 8), (8, 6), (5, 9), (9, 5), (4, 10), (10, 4), (3, 11), (11, 3)]:
        for r_ratio in [0.3, 0.45, 0.6, 0.75]:
            for rot2 in [0.0, np.pi / n2, 2 * np.pi / n2, np.pi / n1]:
                p1 = _regular_polygon(n1, 1.0, 0.0)
                p2 = _regular_polygon(n2, r_ratio, rot2)
                starts.append(np.vstack([p1, p2]))

    # Three-ring
    for (n1, n2, n3) in [(5, 4, 5), (4, 5, 5), (5, 5, 4), (6, 4, 4), (4, 6, 4), (7, 3, 4)]:
        for r2 in [0.5, 0.65]:
            for r3 in [0.25, 0.4]:
                p1 = _regular_polygon(n1, 1.0, 0.0)
                p2 = _regular_polygon(n2, r2, np.pi / n2)
                p3 = _regular_polygon(n3, r3, 0.0)
                starts.append(np.vstack([p1, p2, p3]))

    # Random convex: generate random points, take convex hull, add interior points
    for trial in range(15):
        angles = np.sort(rng.uniform(0, 2 * np.pi, 8))
        radii = rng.uniform(0.85, 1.0, 8)
        outer = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        # Add 6 interior points
        inner_angles = rng.uniform(0, 2 * np.pi, 6)
        inner_radii = rng.uniform(0.2, 0.7, 6)
        inner = np.column_stack([inner_radii * np.cos(inner_angles), inner_radii * np.sin(inner_angles)])
        starts.append(np.vstack([outer, inner]))

    # --- Phase 2: optimize each start with superlevel SLSQP ---
    for i, init in enumerate(starts):
        if not _time_ok(8.0):
            break
        _update(init)
        # Estimate initial t
        ha = _hull_area(init)
        if ha < 1e-12:
            continue
        mta = _min_triangle_area(init)
        t0 = mta / ha  # start t at current ratio (normalized)
        # Work in normalized space: scale so hull area = 1
        pts_norm = init / np.sqrt(ha)
        t0_norm = mta / ha  # min area / hull area
        x = _optimize_superlevel(pts_norm, t0_norm, maxiter=300, time_budget=8.0)
        if x is not None:
            cand = x[:-1].reshape(-1, 2)
            _update(cand)

    # --- Phase 3: local refinement via coordinate descent on best ---
    if best_pts is not None and _time_ok(10.0):
        pts = best_pts.copy()
        for it in range(3):
            if not _time_ok(5.0):
                break
            for i in range(N):
                if not _time_ok(2.0):
                    break
                def _obj1d(p):
                    new = pts.copy()
                    new[i] = p
                    return -_score(new)
                try:
                    res = scipy.optimize.minimize_scalar(
                        lambda px: _obj1d(np.array([px, pts[i, 1]])),
                        bounds=(pts[i, 0] - 0.3, pts[i, 0] + 0.3),
                        options={'maxiter': 30}
                    )
                    pts[i, 0] = res.x
                    res = scipy.optimize.minimize_scalar(
                        lambda py: _obj1d(np.array([pts[i, 0], py])),
                        bounds=(pts[i, 1] - 0.3, pts[i, 1] + 0.3),
                        options={'maxiter': 30}
                    )
                    pts[i, 1] = res.x
                except Exception:
                    pass
            _update(pts)

    # --- Phase 4: final SLSQP polish on best ---
    if best_pts is not None and _time_ok(5.0):
        ha = _hull_area(best_pts)
        if ha > 1e-12:
            pts_norm = best_pts / np.sqrt(ha)
            mta = _min_triangle_area(pts_norm)
            t0 = mta
            x = _optimize_superlevel(pts_norm, t0, maxiter=200, time_budget=4.0)
            if x is not None:
                cand = x[:-1].reshape(-1, 2)
                _update(cand)

    if best_pts is None:
        best_pts = _regular_polygon(N, 1.0)
    return best_pts
