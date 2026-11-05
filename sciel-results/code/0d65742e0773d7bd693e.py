import math
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.spatial import ConvexHull

N = 11
D = math.sqrt(3)

def _hex_vertices(cx, cy, angle_deg, s=1.0):
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    vs = np.column_stack([s * np.cos(angles), s * np.sin(angles)])
    return vs + np.array([cx, cy])

def _point_in_polygon(pt, poly):
    n = len(poly)
    sign = 0
    for i in range(n):
        a = poly[i]
        b = poly[(i+1) % n]
        cross = (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])
        if cross > 1e-12:
            if sign < 0:
                return False
            sign = 1
        elif cross < -1e-12:
            if sign > 0:
                return False
            sign = -1
    return True

def _seg_seg_dist(p1, p2, p3, p4):
    d1 = p2 - p1
    d2 = p4 - p3
    r = p1 - p3
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)
    if a < 1e-12 and e < 1e-12:
        return np.linalg.norm(p1 - p3)
    if a < 1e-12:
        t = f / e
        t = max(0, min(1, t))
        return np.linalg.norm(p1 - (p3 + t * d2))
    c = np.dot(d1, r)
    if e < 1e-12:
        s = c / a
        s = max(0, min(1, s))
        return np.linalg.norm((p1 + s * d1) - p3)
    b = np.dot(d1, d2)
    denom = a * e - b * b
    if denom > 1e-12:
        s = (b * f - c * e) / denom
        s = max(0, min(1, s))
        t = (b * s + f) / e
        if t < 0:
            t = 0
            s = max(0, min(1, c / a))
        elif t > 1:
            t = 1
            s = max(0, min(1, (b + f) / a))
        return np.linalg.norm((p1 + s * d1) - (p3 + t * d2))
    else:
        t = f / e
        t = max(0, min(1, t))
        return np.linalg.norm(p1 - (p3 + t * d2))

def _poly_separation(p1, p2):
    for v in p1:
        if _point_in_polygon(v, p2):
            return 0.0
    for v in p2:
        if _point_in_polygon(v, p1):
            return 0.0
    min_dist = float('inf')
    n1 = len(p1)
    n2 = len(p2)
    for i in range(n1):
        a1 = p1[i]
        a2 = p1[(i+1) % n1]
        for j in range(n2):
            b1 = p2[j]
            b2 = p2[(j+1) % n2]
            d = _seg_seg_dist(a1, a2, b1, b2)
            if d < min_dist:
                min_dist = d
    return min_dist

def _outer_hex_vertices(cx, cy, side, angle_deg):
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    vs = np.column_stack([side * np.cos(angles), side * np.sin(angles)])
    return vs + np.array([cx, cy])

def _normals_for(theta_deg):
    """Unit edge normals of a regular hexagon with rotation theta_deg.
    For a regular hexagon, edge normals point at theta+30, theta+90, ... (6 evenly spaced)."""
    ang = np.deg2rad(theta_deg + 30 + np.arange(6) * 60)
    return np.column_stack([np.cos(ang), np.sin(ang)])

def _width_side(centers, base_vertices, theta_deg, hull_verts=None):
    """Side length of the smallest enclosing regular hexagon (rotation theta_deg)
    that contains all inner hexagons. Translation-invariant: depends only on widths.
    side = max_k (max_v v.n_k - min_v v.n_k) / sqrt(3)."""
    normals = _normals_for(theta_deg)
    if hull_verts is None:
        # gather all vertices
        verts = np.vstack([base_vertices + c for c in centers])
        hull_verts = _convex_hull_vertices(verts)
    proj = hull_verts @ normals.T          # (n_hull, 6)
    widths = proj.max(axis=0) - proj.min(axis=0)
    return float(widths.max() / D)

def _center_and_side(centers, base_vertices, theta_deg, hull_verts=None):
    """Return (side, outer_center) for the smallest enclosing hexagon with rotation
    theta_deg. Uses width-based side (translation-invariant) plus analytic centering
    via 1D interval intersection along the tightest normal."""
    normals = _normals_for(theta_deg)
    if hull_verts is None:
        verts = np.vstack([base_vertices + c for c in centers])
        hull_verts = _convex_hull_vertices(verts)
    proj = hull_verts @ normals.T          # (n_hull, 6)
    max_p = proj.max(axis=0)
    min_p = proj.min(axis=0)
    widths = max_p - min_p
    side = float(widths.max() / D)

    ap = D / 2.0
    L = min_p + side * ap
    R = max_p - side * ap

    # tightest normal index
    ti = int(np.argmax(widths))
    n_tight = normals[ti]
    c_tight = L[ti]
    C0 = c_tight * n_tight
    v_perp = np.array([-n_tight[1], n_tight[0]])

    t_min = -np.inf
    t_max = np.inf
    for k in range(6):
        if k == ti:
            continue
        A = np.dot(v_perp, normals[k])
        B = np.dot(C0, normals[k])
        Lk, Rk = L[k], R[k]
        if abs(A) < 1e-9:
            if B < Lk - 1e-9 or B > Rk + 1e-9:
                return side, (0.0, 0.0)
        else:
            lo = Lk - B
            hi = Rk - B
            if A > 0:
                t_low, t_high = lo / A, hi / A
            else:
                t_low, t_high = hi / A, lo / A
            t_min = max(t_min, t_low)
            t_max = min(t_max, t_high)
            if t_min > t_max + 1e-9:
                return side, (0.0, 0.0)

    if t_min == -np.inf and t_max == np.inf:
        t_val = 0.0
    elif t_min == -np.inf:
        t_val = t_max
    elif t_max == np.inf:
        t_val = t_min
    else:
        t_val = 0.5 * (t_min + t_max)

    C_opt = C0 + t_val * v_perp
    return side, (float(C_opt[0]), float(C_opt[1]))

def _all_inner_vertices(inner_params):
    verts = []
    for i in range(len(inner_params)):
        x = inner_params[i, 0]
        y = inner_params[i, 1]
        a = inner_params[i, 2]
        verts.append(_hex_vertices(x, y, a))
    return np.vstack(verts)

def _convex_hull_vertices(verts):
    if len(verts) < 3:
        return verts
    try:
        hull = ConvexHull(verts)
        return verts[hull.vertices]
    except Exception:
        return verts

def _overlap_penalty(inner_params, penalty_weight=50.0):
    n = len(inner_params)
    penalty = 0.0
    for i in range(n):
        v1 = _hex_vertices(inner_params[i, 0], inner_params[i, 1], inner_params[i, 2])
        for j in range(i + 1, n):
            v2 = _hex_vertices(inner_params[j, 0], inner_params[j, 1], inner_params[j, 2])
            sep = _poly_separation(v1, v2)
            if sep < 0.005:
                penalty += penalty_weight * (0.005 - sep) ** 2
    return penalty

def _is_valid(inner_params, side, outer_center, outer_angle_deg, min_sep=0.0):
    n = len(inner_params)
    outer_v = _outer_hex_vertices(outer_center[0], outer_center[1], side, outer_angle_deg)
    for i in range(n):
        hex_v = _hex_vertices(inner_params[i, 0], inner_params[i, 1], inner_params[i, 2])
        for v in hex_v:
            if not _point_in_polygon(v, outer_v):
                return False
    for i in range(n):
        v1 = _hex_vertices(inner_params[i, 0], inner_params[i, 1], inner_params[i, 2])
        for j in range(i + 1, n):
            v2 = _hex_vertices(inner_params[j, 0], inner_params[j, 1], inner_params[j, 2])
            if _poly_separation(v1, v2) < min_sep:
                return False
    return True

def _seed_solution():
    eps = 1e-9
    centers = [(0.0, 0.0)]
    for k in range(6):
        ang = np.deg2rad(k * 60)
        centers.append(((np.sqrt(3) + eps) * np.cos(ang), (np.sqrt(3) + eps) * np.sin(ang)))
    for k in range(4):
        ang = np.deg2rad(30 + k * 60)
        centers.append(((3.0 + 2 * eps) * np.cos(ang), (3.0 + 2 * eps) * np.sin(ang)))
    inner = [[x, y, 30.0] for (x, y) in centers]
    return inner

def _lattice_packing():
    sqrt3 = math.sqrt(3)
    v1 = np.array([sqrt3, 0.0])
    v2 = np.array([sqrt3 / 2, 1.5])
    selected = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1), (1, 1), (-1, -1), (2, 0), (0, -2)]
    centers = []
    for (i, j) in selected:
        centers.append(i * v1 + j * v2)
    inner = [[float(c[0]), float(c[1]), 30.0] for c in centers]
    return inner

def _build_base(inner_params):
    centers = np.array([[p[0], p[1]] for p in inner_params])
    base_vertices = _hex_vertices(0.0, 0.0, inner_params[0][2])
    return centers, base_vertices

def _angle_search(inner_params, initial_angle, time_limit):
    """High-precision polish of the outer hexagon rotation angle.
    Returns (side, angle, center). Uses width-based side + analytic centering."""
    centers, base_vertices = _build_base(inner_params)
    all_verts = _all_inner_vertices(np.array(inner_params))
    hull_verts = _convex_hull_vertices(all_verts)

    def side_of(a):
        return _width_side(centers, base_vertices, a, hull_verts)

    best_side = side_of(initial_angle)
    best_angle = initial_angle

    # coarse
    for a in np.arange(initial_angle - 5, initial_angle + 5, 0.5):
        if time.time() > time_limit:
            break
        s = side_of(a)
        if s < best_side:
            best_side, best_angle = s, a
    # medium
    for a in np.arange(best_angle - 0.5, best_angle + 0.5, 0.05):
        if time.time() > time_limit:
            break
        s = side_of(a)
        if s < best_side:
            best_side, best_angle = s, a
    # fine scalar
    if time.time() < time_limit:
        try:
            res = minimize_scalar(lambda a: side_of(a), bounds=(best_angle - 0.1, best_angle + 0.1),
                                  method='bounded', options={'xatol': 1e-12})
            if res.success and res.fun < best_side:
                best_side, best_angle = res.fun, res.x
        except Exception:
            pass
    # ternary around local minima
    if time.time() < time_limit - 2:
        n_samples = 200
        ang_s = np.linspace(best_angle - 1, best_angle + 1, n_samples)
        s_s = np.array([side_of(a) for a in ang_s])
        for i in range(1, n_samples - 1):
            if s_s[i] < s_s[i - 1] and s_s[i] < s_s[i + 1]:
                lo, hi = ang_s[i - 1], ang_s[i + 1]
                for _ in range(50):
                    m1 = lo + (hi - lo) / 3
                    m2 = hi - (hi - lo) / 3
                    if side_of(m1) < side_of(m2):
                        hi = m2
                    else:
                        lo = m1
                    if hi - lo < 1e-10:
                        break
                mid = 0.5 * (lo + hi)
                fm = side_of(mid)
                if fm < best_side:
                    best_side, best_angle = fm, mid

    side, center = _center_and_side(centers, base_vertices, best_angle, hull_verts)
    return side, best_angle, center

def _objective(params, outer_angle_deg, n):
    inner_params = params.reshape(n, 3)
    centers, base_vertices = _build_base(inner_params)
    side = _width_side(centers, base_vertices, outer_angle_deg)
    return side + _overlap_penalty(inner_params)

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    margin = 5.0
    n = 11

    # seed candidates
    seed1 = _seed_solution()
    seed2 = _lattice_packing()
    seed_candidates = [seed1, seed2]

    best_side = float('inf')
    best_inner = None
    best_angle = 0.0
    best_center = (0.0, 0.0)

    for seed in seed_candidates:
        if time.time() > DEADLINE - margin - 40:
            break
        side, ang, ctr = _angle_search(seed, 30.0, DEADLINE - margin - 40)
        if side < best_side:
            best_side, best_inner, best_angle, best_center = side, [list(r) for r in seed], ang, ctr

    # inner optimization for a few outer angles (Nelder-Mead, smooth objective)
    init_params = np.zeros(3 * n)
    for i, (x, y, a) in enumerate(best_inner):
        init_params[i * 3] = x
        init_params[i * 3 + 1] = y
        init_params[i * 3 + 2] = a

    for oa in [best_angle, 0.0, 15.0, 30.0, 45.0]:
        if time.time() > DEADLINE - margin - 20:
            break
        try:
            res = minimize(_objective, init_params.copy(), args=(oa, n),
                           method='Nelder-Mead',
                           options={'maxiter': 400, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True})
            if time.time() > DEADLINE - margin:
                break
            cand = res.x.reshape(n, 3)
            side, ctr = _center_and_side(*_build_base(cand), oa)
            if _is_valid(cand, side, ctr, oa, min_sep=1e-8) and side < best_side:
                best_side, best_inner, best_angle, best_center = side, cand.tolist(), oa, ctr
                init_params = res.x.copy()
        except Exception:
            pass

    # final angle polish
    if time.time() < DEADLINE - margin:
        side, ang, ctr = _angle_search(best_inner, best_angle, DEADLINE - margin)
        if side < best_side:
            best_side, best_angle, best_center = side, ang, ctr

    # final validation
    inner_arr = np.array(best_inner)
    side, ctr = _center_and_side(*_build_base(inner_arr), best_angle)
    if not _is_valid(inner_arr, side, ctr, best_angle, min_sep=1e-8):
        # fallback
        side, ang, ctr = _angle_search(seed1, 30.0, DEADLINE - margin)
        best_inner, best_side, best_angle, best_center = [list(r) for r in seed1], side, ang, ctr

    inner_rows = [[float(r[0]), float(r[1]), float(r[2])] for r in best_inner]
    return inner_rows, (float(best_center[0]), float(best_center[1])), float(best_side), float(best_angle)
