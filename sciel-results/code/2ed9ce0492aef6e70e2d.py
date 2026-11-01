"""
Position-Based Dynamics (PBD) based packing of 11 unit hexagons in a regular hexagon.
Uses iterative projection-based collision resolution for inner-inner and inner-container
constraints, combined with a smooth surrogate for the container size objective.
"""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)
EPS = 1e-10

# EVOLVE-BLOCK-START

def hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon as a numpy array (6,2)."""
    thetas = angle_rad + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + side * np.cos(thetas), cy + side * np.sin(thetas)])

def compute_required_side_fast(inner, outer_center, outer_angle_deg):
    """Compute minimum outer side length to contain all inner hexagons."""
    if not inner:
        return 0.0
    ox, oy = outer_center
    normals = np.array([[math.cos(math.radians(30 + i * 60)),
                         math.sin(math.radians(30 + i * 60))] for i in range(6)])
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    R = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
    max_proj = -np.inf
    for (ix, iy, iang) in inner:
        verts = hex_vertices(ix, iy, 1.0, math.radians(iang))
        local = (verts - np.array([ox, oy])) @ R.T
        projs = local @ normals.T
        cur = projs.max()
        if cur > max_proj:
            max_proj = cur
    if max_proj < 0:
        max_proj = 0.0
    return 2.0 * max_proj / SQRT3

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=50.0):
    """Smooth (log-sum-exp) surrogate for the max projection."""
    if not inner:
        return 0.0
    ox, oy = outer_center
    normals = np.array([[math.cos(math.radians(30 + i * 60)),
                         math.sin(math.radians(30 + i * 60))] for i in range(6)])
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    R = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
    all_projs = []
    for (ix, iy, iang) in inner:
        verts = hex_vertices(ix, iy, 1.0, math.radians(iang))
        local = (verts - np.array([ox, oy])) @ R.T
        projs = local @ normals.T
        all_projs.append(projs)
    P = np.concatenate([p.ravel() for p in all_projs])
    if P.size == 0:
        return 0.0
    m = P.max()
    if m < -1e6:
        return 0.0
    smooth_max = m + math.log(np.sum(np.exp(beta * (P - m)))) / beta
    if smooth_max < 0:
        smooth_max = 0.0
    return 2.0 * smooth_max / SQRT3

def sat_overlap(v1, v2, eps=1e-9):
    """SAT overlap check for two convex polygons (lists of (x,y))."""
    axes = []
    for poly in (v1, v2):
        for i in range(len(poly)):
            p1 = poly[i]; p2 = poly[(i + 1) % len(poly)]
            e = (p2[0] - p1[0], p2[1] - p1[1])
            n = (-e[1], e[0])
            ln = math.hypot(n[0], n[1])
            if ln > 1e-12:
                axes.append((n[0] / ln, n[1] / ln))
    for ax in axes:
        min1 = min(p[0] * ax[0] + p[1] * ax[1] for p in v1)
        max1 = max(p[0] * ax[0] + p[1] * ax[1] for p in v1)
        min2 = min(p[0] * ax[0] + p[1] * ax[1] for p in v2)
        max2 = max(p[0] * ax[0] + p[1] * ax[1] for p in v2)
        if max1 < min2 + eps or max2 < min1 + eps:
            return False
    return True

def check_disjoint(inner, eps=1e-6):
    """Pairwise SAT check that all inner hexagons are disjoint."""
    n = len(inner)
    polys = []
    for (x, y, a) in inner:
        polys.append(hex_vertices(x, y, 1.0, math.radians(a)))
    for i in range(n):
        for j in range(i + 1, n):
            if sat_overlap(polys[i], polys[j], eps):
                return False
    return True

def overlap_penalty(inner, min_dist=SQRT3 - 0.001):
    """Quadratic penalty when center distances fall below min_dist."""
    pen = 0.0
    n = len(inner)
    for i in range(n):
        xi, yi, _ = inner[i]
        for j in range(i + 1, n):
            xj, yj, _ = inner[j]
            d = math.hypot(xi - xj, yi - yj)
            if d < min_dist:
                pen += (min_dist - d) ** 2
    return pen

def build_lattice_config(axial_coords, s=1.0 + 1e-8):
    """Build inner hexagons from axial coordinates with spacing s."""
    centers = [(1.5 * s * q, SQRT3 * s * (r + q / 2.0)) for q, r in axial_coords]
    return [[x, y, 0.0] for x, y in centers]

def pbd_solve(inner, outer_center, outer_angle_deg, outer_side, iterations=200, beta=50.0, deadline=None):
    """
    Position-Based Dynamics solver: iteratively project hexagons to satisfy
    non-overlap (inner-inner) and containment (inner-container) constraints.
    Returns (inner, outer_center, outer_angle_deg, outer_side).
    """
    n = len(inner)
    # Work with mutable arrays
    pos = np.array([[inner[i][0], inner[i][1]] for i in range(n)], dtype=float)
    ang = np.array([inner[i][2] for i in range(n)], dtype=float)
    oc = np.array(outer_center, dtype=float)
    oa = outer_angle_deg
    side = outer_side

    # Precompute hexagon vertex offsets (unit hexagon at angle 0)
    base_thetas = np.arange(6) * np.pi / 3.0
    base_vx = np.cos(base_thetas)
    base_vy = np.sin(base_thetas)
    base_verts = np.column_stack([base_vx, base_vy])

    # Container normals
    cnormals = np.array([[math.cos(math.radians(30 + i * 60)),
                          math.sin(math.radians(30 + i * 60))] for i in range(6)])

    def get_verts(i):
        a = math.radians(ang[i])
        ca, sa = math.cos(a), math.sin(a)
        R = np.array([[ca, sa], [-sa, ca]])
        return pos[i] + base_verts @ R.T

    for it in range(iterations):
        if deadline is not None and time.time() > deadline - 10:
            break

        # --- Inner-inner collision resolution ---
        for i in range(n):
            vi = get_verts(i)
            for j in range(i + 1, n):
                vj = get_verts(j)
                # SAT check
                axes = []
                for poly in (vi, vj):
                    for k in range(6):
                        p1 = poly[k]; p2 = poly[(k + 1) % 6]
                        e = p2 - p1
                        nn = np.array([-e[1], e[0]])
                        ln = np.linalg.norm(nn)
                        if ln > 1e-12:
                            axes.append(nn / ln)
                min_dist = 0.0
                min_axis = None
                for ax in axes:
                    proj_i = vi @ ax
                    proj_j = vj @ ax
                    gap = min(proj_j.max() - proj_i.min(), proj_i.max() - proj_j.min())
                    if gap < 0:
                        if gap < min_dist:
                            min_dist = gap
                            min_axis = ax
                if min_axis is not None:
                    # Push apart along min_axis
                    center_i = pos[i]
                    center_j = pos[j]
                    diff = center_j - center_i
                    if diff @ min_axis < 0:
                        min_axis = -min_axis
                    push = -min_dist / 2.0
                    pos[i] -= push * min_axis
                    pos[j] += push * min_axis

        # --- Inner-container collision resolution ---
        # Compute current required side
        max_proj = -np.inf
        for i in range(n):
            vi = get_verts(i)
            local = (vi - oc) @ np.array([[math.cos(math.radians(oa)), math.sin(math.radians(oa))],
                                           [-math.sin(math.radians(oa)), math.cos(math.radians(oa))]]).T
            projs = local @ cnormals.T
            cur = projs.max()
            if cur > max_proj:
                max_proj = cur
        if max_proj < 0:
            max_proj = 0.0
        current_side = 2.0 * max_proj / SQRT3

        if current_side > side:
            # Scale container or push inner hexagons inward
            # We'll push the offending hexagons toward the center
            excess = current_side - side
            for i in range(n):
                vi = get_verts(i)
                local = (vi - oc) @ np.array([[math.cos(math.radians(oa)), math.sin(math.radians(oa))],
                                               [-math.sin(math.radians(oa)), math.cos(math.radians(oa))]]).T
                projs = local @ cnormals.T
                # For each normal, if projection exceeds allowed, push back
                allowed = side * SQRT3 / 2.0
                for k in range(6):
                    if projs[k] > allowed:
                        # Push hexagon center toward container center along this normal
                        norm_vec = cnormals[k]
                        # Rotate back to world frame
                        ca_o = math.cos(math.radians(oa))
                        sa_o = math.sin(math.radians(oa))
                        world_norm = np.array([ca_o * norm_vec[0] - sa_o * norm_vec[1],
                                                sa_o * norm_vec[0] + ca_o * norm_vec[1]])
                        push = (projs[k] - allowed) * 0.5
                        pos[i] -= push * world_norm

    # Final side computation
    max_proj = -np.inf
    for i in range(n):
        vi = get_verts(i)
        local = (vi - oc) @ np.array([[math.cos(math.radians(oa)), math.sin(math.radians(oa))],
                                       [-math.sin(math.radians(oa)), math.cos(math.radians(oa))]]).T
        projs = local @ cnormals.T
        cur = projs.max()
        if cur > max_proj:
            max_proj = cur
    if max_proj < 0:
        max_proj = 0.0
    final_side = 2.0 * max_proj / SQRT3

    result_inner = [[float(pos[i][0]), float(pos[i][1]), float(ang[i])] for i in range(n)]
    return result_inner, (float(oc[0]), float(oc[1])), float(oa), float(final_side)

def rigid_composite_optimize(axial_coords, deadline):
    """
    Treat the 11 hexagons as a rigid composite (6 DOF).
    """
    local_vertices = []
    for q, r in axial_coords:
        cx = 1.5 * q
        cy = SQRT3 * (r + q / 2.0)
        thetas = np.arange(6) * np.pi / 3.0
        vx = cx + np.cos(thetas)
        vy = cy + np.sin(thetas)
        local_vertices.append(np.column_stack([vx, vy]))

    all_local = np.vstack(local_vertices)
    normals = np.array([[math.cos(math.radians(30 + i * 60)),
                         math.sin(math.radians(30 + i * 60))] for i in range(6)])

    def obj_rigid_smooth(params, beta=80.0):
        tx, ty, ca_deg, ocx, ocy, oa_deg = params
        ca = math.radians(ca_deg)
        oa = math.radians(oa_deg)
        cos_c, sin_c = math.cos(ca), math.sin(ca)
        Rc = np.array([[cos_c, sin_c], [-sin_c, cos_c]])
        pts = (all_local @ Rc.T) + np.array([tx, ty])
        cos_o, sin_o = math.cos(oa), math.sin(oa)
        Ro = np.array([[cos_o, sin_o], [-sin_o, cos_o]])
        local_pts = (pts - np.array([ocx, ocy])) @ Ro.T
        projs = local_pts @ normals.T
        P = projs.ravel()
        m = P.max()
        if m < -1e6:
            smooth_max = 0.0
        else:
            smooth_max = m + math.log(np.sum(np.exp(beta * (P - m)))) / beta
            if smooth_max < 0:
                smooth_max = 0.0
        return 2.0 * smooth_max / SQRT3

    def obj_rigid_exact(params):
        tx, ty, ca_deg, ocx, ocy, oa_deg = params
        ca = math.radians(ca_deg)
        oa = math.radians(oa_deg)
        cos_c, sin_c = math.cos(ca), math.sin(ca)
        Rc = np.array([[cos_c, sin_c], [-sin_c, cos_c]])
        pts = (all_local @ Rc.T) + np.array([tx, ty])
        cos_o, sin_o = math.cos(oa), math.sin(oa)
        Ro = np.array([[cos_o, sin_o], [-sin_o, cos_o]])
        local_pts = (pts - np.array([ocx, ocy])) @ Ro.T
        projs = local_pts @ normals.T
        max_proj = projs.max()
        if max_proj < 0:
            max_proj = 0.0
        return 2.0 * max_proj / SQRT3

    cx0 = all_local[:, 0].mean()
    cy0 = all_local[:, 1].mean()

    best_val = float('inf')
    best_params = None

    for ca0 in range(0, 60, 10):
        for oa0 in range(0, 60, 15):
            if time.time() > deadline - 20:
                break
            p0 = np.array([0.0, 0.0, float(ca0), cx0, cy0, float(oa0)])
            try:
                r1 = minimize(obj_rigid_smooth, p0, method='Nelder-Mead',
                              options={'maxiter': 400, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 600})
                r2 = minimize(obj_rigid_exact, r1.x, method='Nelder-Mead',
                              options={'maxiter': 600, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': 1000})
                if r2.fun < best_val:
                    best_val = r2.fun
                    best_params = r2.x.copy()
            except Exception:
                pass

    if best_params is None:
        return None

    tx, ty, ca_deg, ocx, ocy, oa_deg = best_params
    ca = math.radians(ca_deg)
    cos_c, sin_c = math.cos(ca), math.sin(ca)
    Rc = np.array([[cos_c, sin_c], [-sin_c, cos_c]])

    inner = []
    for q, r in axial_coords:
        cx = 1.5 * q
        cy = SQRT3 * (r + q / 2.0)
        pt = np.array([cx, cy]) @ Rc.T + np.array([tx, ty])
        inner.append([float(pt[0]), float(pt[1]), float(ca_deg)])

    return inner, (float(ocx), float(ocy)), float(oa_deg), float(best_val)

def joint_optimize_pbd(init_inner, init_outer_center, init_outer_angle, deadline):
    """
    Joint optimization using PBD-like iterative refinement with smooth surrogate.
    """
    def obj_joint_smooth(params, beta=50.0):
        il = []
        for i in range(N):
            il.append([params[i * 3], params[i * 3 + 1], params[i * 3 + 2]])
        ocx, ocy, oa = params[N * 3], params[N * 3 + 1], params[N * 3 + 2]
        side = compute_required_side_smooth(il, (ocx, ocy), oa, beta=beta)
        pen = overlap_penalty(il)
        return side + 300.0 * pen

    def obj_joint_exact(params):
        il = []
        for i in range(N):
            il.append([params[i * 3], params[i * 3 + 1], params[i * 3 + 2]])
        ocx, ocy, oa = params[N * 3], params[N * 3 + 1], params[N * 3 + 2]
        side = compute_required_side_fast(il, (ocx, ocy), oa)
        pen = overlap_penalty(il)
        return side + 2000.0 * pen

    p0 = np.zeros(N * 3 + 3)
    for i, (x, y, a) in enumerate(init_inner):
        p0[i * 3] = x
        p0[i * 3 + 1] = y
        p0[i * 3 + 2] = a
    p0[N * 3] = init_outer_center[0]
    p0[N * 3 + 1] = init_outer_center[1]
    p0[N * 3 + 2] = init_outer_angle

    current = p0.copy()
    current_val = obj_joint_smooth(current)
    best_params = current.copy()
    best_val_j = current_val

    # Simulated annealing with PBD-inspired projection steps
    T = 0.3
    T_min = 0.0005
    cool = 0.9997
    max_iter = 8000
    np.random.seed(42)
    step_pos = 0.2
    step_ang = 10.0
    step_oc = 0.15
    step_oa = 4.0

    for it in range(max_iter):
        if time.time() > deadline - 20:
            break
        nb = current.copy()
        r = np.random.random()
        if r < 0.55:
            idx = np.random.randint(0, N)
            dim = np.random.randint(0, 2)
            nb[idx * 3 + dim] += np.random.normal(0, step_pos)
        elif r < 0.75:
            idx = np.random.randint(0, N)
            nb[idx * 3 + 2] += np.random.normal(0, step_ang)
        elif r < 0.87:
            dim = np.random.randint(0, 2)
            nb[N * 3 + dim] += np.random.normal(0, step_oc)
        else:
            nb[N * 3 + 2] += np.random.normal(0, step_oa)

        nb_val = obj_joint_smooth(nb)
        if nb_val < current_val or np.random.random() < math.exp(-(nb_val - current_val) / max(T, 1e-10)):
            current = nb
            current_val = nb_val
            if current_val < best_val_j:
                best_params = current.copy()
                best_val_j = current_val
        T *= cool
        if T < T_min:
            break

    best_inner_j = []
    for i in range(N):
        best_inner_j.append([float(best_params[i * 3]), float(best_params[i * 3 + 1]), float(best_params[i * 3 + 2])])
    best_oc_j = (float(best_params[N * 3]), float(best_params[N * 3 + 1]))
    best_oa_j = float(best_params[N * 3 + 2])

    return best_inner_j, best_oc_j, best_oa_j, best_params

def final_polish(params, deadline):
    """High-precision Nelder-Mead polish."""
    def obj_polish(p):
        il = []
        for i in range(N):
            il.append([p[i * 3], p[i * 3 + 1], p[i * 3 + 2]])
        ocx, ocy, oa = p[N * 3], p[N * 3 + 1], p[N * 3 + 2]
        side = compute_required_side_fast(il, (ocx, ocy), oa)
        pen = overlap_penalty(il)
        return side + 5000.0 * pen

    best_res = params.copy()
    best_val = obj_polish(params)

    np.random.seed(789)
    for restart in range(4):
        if time.time() > deadline - 5:
            break
        p_try = params.copy()
        p_try[np.random.randint(0, len(params), size=4)] += np.random.normal(0, 0.005, size=4)
        try:
            res = minimize(obj_polish, p_try, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16, 'maxfev': 8000})
            if res.fun < best_val:
                best_val = res.fun
                best_res = res.x.copy()
        except Exception:
            pass

    try:
        res = minimize(obj_polish, best_res, method='Nelder-Mead',
                       options={'maxiter': 6000, 'xatol': 1e-15, 'fatol': 1e-17, 'maxfev': 10000})
        if res.fun < best_val:
            best_res = res.x.copy()
    except Exception:
        pass

    return best_res

def construct():
    t0 = time.time()
    deadline = t0 + 142.0

    # Candidate configurations
    axial1 = [(0, 0),
              (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
              (2, -1), (1, 1), (-2, 1), (-1, -1)]

    axial2 = [(-1, -1), (0, -1), (1, -1), (2, -1),
              (-1, 0), (0, 0), (1, 0),
              (-1, 1), (0, 1), (1, 1), (2, 1)]

    axial3 = [(-1, -1), (0, -1), (1, -1),
              (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0),
              (-1, 1), (0, 1), (1, 1)]

    axial4 = [(-2, -1), (-1, -1), (0, -1), (1, -1), (2, -1),
              (-1, 0), (0, 0), (1, 0),
              (-1, 1), (0, 1), (1, 1)]

    axial5 = [(0, 0),
              (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
              (2, 0), (0, 2), (-2, 1), (-2, -1)]

    axial6 = [(-1, -1), (0, -1),
              (-3, 0), (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0), (3, 0),
              (-1, 1), (0, 1)]

    axial7 = [(0, 0),
              (-1, 0), (0, -1), (1, -1),
              (-2, 1), (-1, 1), (0, 1), (1, 0), (2, -1),
              (-1, -1), (1, -2)]

    # Additional compact configurations
    axial8 = [(0, 0),
              (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
              (2, -1), (1, 1), (0, 2), (-1, 2)]

    axial9 = [(-1, 0), (0, 0), (1, 0),
              (-1, 1), (0, 1), (1, 1),
              (-2, 2), (-1, 2), (0, 2), (1, 2), (2, 2)]

    axial10 = [(0, 0),
               (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
               (2, 0), (1, 1), (0, 2), (-1, 2)]

    configs = [axial1, axial2, axial3, axial4, axial5, axial6, axial7, axial8, axial9, axial10]

    best_side = float('inf')
    best_inner = None
    best_outer_center = None
    best_outer_angle = 0.0
    best_params = None

    # Phase 0: Rigid composite optimization for top configs
    for ci, axial in enumerate(configs[:5]):
        if time.time() > deadline - 45:
            break
        result = rigid_composite_optimize(axial, deadline)
        if result is not None:
            rc_inner, rc_oc, rc_oa, rc_side = result
            if check_disjoint(rc_inner, eps=1e-5) and rc_side < best_side:
                best_side = rc_side
                best_inner = rc_inner
                best_outer_center = rc_oc
                best_outer_angle = rc_oa
                p0 = np.zeros(N * 3 + 3)
                for i, (x, y, a) in enumerate(rc_inner):
                    p0[i * 3] = x
                    p0[i * 3 + 1] = y
                    p0[i * 3 + 2] = a
                p0[N * 3] = rc_oc[0]
                p0[N * 3 + 1] = rc_oc[1]
                p0[N * 3 + 2] = rc_oa
                best_params = p0.copy()

    # Phase 1: Lattice init + orientation optimization
    for ci, axial in enumerate(configs):
        if time.time() > deadline - 45:
            break
        s0 = 1.0 + 1e-8
        init_inner = build_lattice_config(axial, s0)
        cx0 = sum(c[0] for c in init_inner) / len(init_inner)
        cy0 = sum(c[1] for c in init_inner) / len(init_inner)

        # Try multiple outer angles
        for oa0 in [0.0, 15.0, 30.0, 45.0]:
            if time.time() > deadline - 40:
                break
            try:
                def obj_orient(p):
                    return compute_required_side_smooth(init_inner, (p[0], p[1]), p[2], beta=60.0)

                p0 = np.array([cx0, cy0, oa0])
                r = minimize(obj_orient, p0, method='Nelder-Mead',
                             options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 400})
                side_val = compute_required_side_fast(init_inner, (r.x[0], r.x[1]), r.x[2])
                if side_val < best_side:
                    best_side = side_val
                    best_inner = init_inner
                    best_outer_center = (float(r.x[0]), float(r.x[1]))
                    best_outer_angle = float(r.x[2])
                    p0p = np.zeros(N * 3 + 3)
                    for i, (x, y, a) in enumerate(init_inner):
                        p0p[i * 3] = x
                        p0p[i * 3 + 1] = y
                        p0p[i * 3 + 2] = a
                    p0p[N * 3] = r.x[0]
                    p0p[N * 3 + 1] = r.x[1]
                    p0p[N * 3 + 2] = r.x[2]
                    best_params = p0p.copy()
            except Exception:
                pass

    if best_inner is None:
        axial = configs[0]
        s0 = 1.0 + 1e-8
        best_inner = build_lattice_config(axial, s0)
        cx0 = sum(c[0] for c in best_inner) / len(best_inner)
        cy0 = sum(c[1] for c in best_inner) / len(best_inner)
        best_outer_center = (cx0, cy0)
        best_outer_angle = 0.0
        best_side = compute_required_side_fast(best_inner, best_outer_center, best_outer_angle)
        p0 = np.zeros(N * 3 + 3)
        for i, (x, y, a) in enumerate(best_inner):
            p0[i * 3] = x
            p0[i * 3 + 1] = y
            p0[i * 3 + 2] = a
        p0[N * 3] = cx0
        p0[N * 3 + 1] = cy0
        p0[N * 3 + 2] = 0.0
        best_params = p0.copy()

    # Phase 2: PBD-style joint optimization
    if time.time() < deadline - 25:
        joint_inner, joint_oc, joint_oa, joint_params = joint_optimize_pbd(
            best_inner, best_outer_center, best_outer_angle, deadline)

        if check_disjoint(joint_inner, eps=1e-5):
            joint_side = compute_required_side_fast(joint_inner, joint_oc, joint_oa)
            if joint_side < best_side:
                best_inner = joint_inner
                best_outer_center = joint_oc
                best_outer_angle = joint_oa
                best_side = joint_side
                best_params = joint_params

    # Phase 3: PBD refinement on best result
    if time.time() < deadline - 15:
        pbd_inner, pbd_oc, pbd_oa, pbd_side = pbd_solve(
            best_inner, best_outer_center, best_outer_angle, best_side,
            iterations=100, deadline=deadline)
        if check_disjoint(pbd_inner, eps=1e-5) and pbd_side < best_side:
            best_inner = pbd_inner
            best_outer_center = pbd_oc
            best_outer_angle = pbd_oa
            best_side = pbd_side

    # Phase 4: Final high-precision polish
    if time.time() < deadline - 5 and best_params is not None:
        polished = final_polish(best_params, deadline)
        polish_inner = []
        for i in range(N):
            polish_inner.append([float(polished[i * 3]), float(polished[i * 3 + 1]), float(polished[i * 3 + 2])])
        polish_oc = (float(polished[N * 3]), float(polished[N * 3 + 1]))
        polish_oa = float(polished[N * 3 + 2])

        if check_disjoint(polish_inner, eps=1e-5):
            polish_side = compute_required_side_fast(polish_inner, polish_oc, polish_oa)
            if polish_side < best_side:
                best_inner = polish_inner
                best_outer_center = polish_oc
                best_outer_angle = polish_oa
                best_side = polish_side

    # Final verification
    if not check_disjoint(best_inner, eps=1e-5):
        s0 = 1.0 + 1e-8
        best_inner = build_lattice_config(configs[0], s0)
        cx0 = sum(c[0] for c in best_inner) / len(best_inner)
        cy0 = sum(c[1] for c in best_inner) / len(best_inner)
        best_outer_center = (cx0, cy0)
        best_outer_angle = 0.0
        best_side = compute_required_side_fast(best_inner, best_outer_center, best_outer_angle)

    return best_inner, best_outer_center, best_side, best_outer_angle

# EVOLVE-BLOCK-END
