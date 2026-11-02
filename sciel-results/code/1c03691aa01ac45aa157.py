"""Combined hexagon packing: contact-graph seeds + rigid composite + smooth surrogate + L-BFGS-B polish."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

def hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon as a numpy array (6,2)."""
    thetas = angle_rad + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + side * np.cos(thetas), cy + side * np.sin(thetas)])

def compute_required_side_fast(inner, outer_center, outer_angle_deg):
    """
    Compute minimum outer side length to contain all inner hexagons.
    inner: list of (x, y, angle_degrees)
    """
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

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=80.0):
    """
    Smooth (log-sum-exp) surrogate for the max projection.
    """
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

def overlap_penalty(inner, min_dist=SQRT3 - 0.002):
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

def try_orientation_optimization(inner, outer_center, outer_angle_deg, deadline):
    """Optimize outer hexagon orientation and center for fixed inner positions."""
    def obj_exact(p):
        return compute_required_side_fast(inner, (p[0], p[1]), p[2])
    
    best_val = float('inf')
    best_p = None
    # Coarse angle sweep first to find good region
    for ia in range(0, 60, 5):
        if time.time() > deadline - 15:
            break
        p0 = np.array([outer_center[0], outer_center[1], float(ia)])
        try:
            r1 = minimize(obj_exact, p0, method='Nelder-Mead',
                          options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 200})
            if r1.fun < best_val:
                best_val = r1.fun
                best_p = r1.x.copy()
        except Exception:
            pass
    # Fine sweep around best angle found
    if best_p is not None and time.time() < deadline - 8:
        for ia in np.arange(best_p[2] - 5, best_p[2] + 5, 1.0):
            if time.time() > deadline - 8:
                break
            p0 = np.array([best_p[0], best_p[1], float(ia)])
            try:
                r1 = minimize(obj_exact, p0, method='Nelder-Mead',
                              options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 400})
                if r1.fun < best_val:
                    best_val = r1.fun
                    best_p = r1.x.copy()
            except Exception:
                pass
    
    if best_p is not None:
        return (float(best_p[0]), float(best_p[1])), float(best_p[2]), best_val
    else:
        return outer_center, outer_angle_deg, compute_required_side_fast(inner, outer_center, outer_angle_deg)

def rigid_composite_objective(base_centers, cx0, cy0, p):
    """
    Objective for rigid composite: 5 DOF (comp_angle, dx, dy, outer_angle, outer_center_x_offset)
    We fix outer center to centroid + offset for simplicity.
    p = [comp_angle, dx, dy, outer_angle, ox_off, oy_off]
    """
    comp_a, dx, dy, oa, ox_off, oy_off = p[0], p[1], p[2], p[3], p[4], p[5]
    ca = math.cos(math.radians(comp_a))
    sa = math.sin(math.radians(comp_a))
    inner = []
    for (bx, by, _) in base_centers:
        rx = bx - cx0
        ry = by - cy0
        nx = cx0 + dx + ca * rx - sa * ry
        ny = cy0 + dy + sa * rx + ca * ry
        # Rotate each hexagon by comp_angle
        na = comp_a
        inner.append([nx, ny, na])
    
    ocx = cx0 + dx + ox_off
    ocy = cy0 + dy + oy_off
    
    side = compute_required_side_fast(inner, (ocx, ocy), oa)
    # Penalty for overlaps in rigid composite
    pen = overlap_penalty(inner)
    return side + 1000.0 * pen

def rigid_composite_smooth(base_centers, cx0, cy0, p, beta=60.0):
    """Smooth version of rigid composite objective."""
    comp_a, dx, dy, oa, ox_off, oy_off = p[0], p[1], p[2], p[3], p[4], p[5]
    ca = math.cos(math.radians(comp_a))
    sa = math.sin(math.radians(comp_a))
    inner = []
    for (bx, by, _) in base_centers:
        rx = bx - cx0
        ry = by - cy0
        nx = cx0 + dx + ca * rx - sa * ry
        ny = cy0 + dy + sa * rx + ca * ry
        na = comp_a
        inner.append([nx, ny, na])
    
    ocx = cx0 + dx + ox_off
    ocy = cy0 + dy + oy_off
    
    side = compute_required_side_smooth(inner, (ocx, ocy), oa, beta=beta)
    pen = overlap_penalty(inner)
    return side + 1000.0 * pen

def optimize_rigid_composite(axial, deadline):
    """
    Optimize rigid composite: 6 DOF (comp_angle, dx, dy, outer_angle, ox_off, oy_off)
    Returns (inner, outer_center, outer_angle, side)
    """
    s0 = 1.0 + 1e-8
    base = build_lattice_config(axial, s0)
    cx0 = sum(c[0] for c in base) / len(base)
    cy0 = sum(c[1] for c in base) / len(base)
    
    best_p = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_val = rigid_composite_objective(base, cx0, cy0, best_p)
    
    # Coarse sweep of comp_angle and outer_angle
    for ca in range(0, 60, 10):
        for oa in range(0, 60, 10):
            if time.time() > deadline - 30:
                break
            p0 = np.array([float(ca), 0.0, 0.0, float(oa), 0.0, 0.0])
            try:
                r = minimize(rigid_composite_objective, p0, args=(base, cx0, cy0),
                             method='Nelder-Mead',
                             options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8, 'maxfev': 300})
                if r.fun < best_val:
                    best_val = r.fun
                    best_p = r.x.copy()
            except Exception:
                pass
    
    # Fine polish with smooth objective
    if time.time() < deadline - 10:
        try:
            r = minimize(lambda p: rigid_composite_smooth(base, cx0, cy0, p, beta=50.0),
                         best_p, method='Nelder-Mead',
                         options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 4000, 'adaptive': True})
            if r.fun < best_val:
                best_val = r.fun
                best_p = r.x.copy()
        except Exception:
            pass
    
    # Extract result
    comp_a, dx, dy, oa, ox_off, oy_off = best_p
    ca = math.cos(math.radians(comp_a))
    sa = math.sin(math.radians(comp_a))
    inner = []
    for (bx, by, _) in base:
        rx = bx - cx0
        ry = by - cy0
        nx = cx0 + dx + ca * rx - sa * ry
        ny = cy0 + dy + sa * rx + ca * ry
        na = comp_a
        inner.append([nx, ny, na])
    
    ocx = cx0 + dx + ox_off
    ocy = cy0 + dy + oy_off
    
    # Check disjointness
    if not check_disjoint(inner, eps=1e-5):
        # Repair
        for repair_iter in range(30):
            polys = [hex_vertices(x, y, 1.0, math.radians(a)) for x, y, a in inner]
            separated = True
            for i in range(N):
                for j in range(i + 1, N):
                    if sat_overlap(polys[i], polys[j], eps=-1e-7):
                        separated = False
                        dx_rep = inner[j][0] - inner[i][0]
                        dy_rep = inner[j][1] - inner[i][1]
                        d = math.hypot(dx_rep, dy_rep)
                        if d > 1e-6:
                            push = 0.02
                            inner[i][0] -= push * dx_rep / d
                            inner[i][1] -= push * dy_rep / d
                            inner[j][0] += push * dx_rep / d
                            inner[j][1] += push * dy_rep / d
                        else:
                            inner[i][0] -= 0.02
                            inner[j][0] += 0.02
                        polys = [hex_vertices(x, y, 1.0, math.radians(a)) for x, y, a in inner]
                        break
                if not separated:
                    break
            if separated:
                break
    
    side = compute_required_side_fast(inner, (ocx, ocy), oa)
    return inner, (ocx, ocy), oa, side

def joint_optimize(init_inner, init_outer_center, init_outer_angle, deadline):
    """Joint optimization of inner positions/angles and outer hexagon using contact-graph awareness."""
    def obj_joint_smooth(params, beta=50.0):
        il = []
        for i in range(N):
            il.append([params[i * 3], params[i * 3 + 1], params[i * 3 + 2]])
        ocx, ocy, oa = params[N * 3], params[N * 3 + 1], params[N * 3 + 2]
        side = compute_required_side_smooth(il, (ocx, ocy), oa, beta=beta)
        pen = overlap_penalty(il)
        return side + 2000.0 * pen

    def obj_joint_exact(params):
        il = []
        for i in range(N):
            il.append([params[i * 3], params[i * 3 + 1], params[i * 3 + 2]])
        ocx, ocy, oa = params[N * 3], params[N * 3 + 1], params[N * 3 + 2]
        side = compute_required_side_fast(il, (ocx, ocy), oa)
        pen = overlap_penalty(il)
        return side + 5000.0 * pen

    # Build initial params
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

    # Multi-phase simulated annealing with adaptive step sizes
    np.random.seed(123)
    phases = [
        {'T': 0.8, 'T_min': 0.01, 'cool': 0.9998, 'pos_step': 0.3, 'ang_step': 12.0, 'oc_step': 0.2, 'oa_step': 5.0},
        {'T': 0.3, 'T_min': 0.003, 'cool': 0.99985, 'pos_step': 0.15, 'ang_step': 6.0, 'oc_step': 0.1, 'oa_step': 3.0},
        {'T': 0.1, 'T_min': 0.001, 'cool': 0.9999, 'pos_step': 0.06, 'ang_step': 3.0, 'oc_step': 0.05, 'oa_step': 1.5},
    ]
    
    for phase in phases:
        if time.time() > deadline - 25:
            break
        T = phase['T']
        T_min = phase['T_min']
        cool = phase['cool']
        max_iter = 5000
        for it in range(max_iter):
            if time.time() > deadline - 25:
                break
            nb = current.copy()
            r = np.random.random()
            if r < 0.55:
                # perturb one inner hexagon position
                idx = np.random.randint(0, N)
                dim = np.random.randint(0, 2)
                nb[idx * 3 + dim] += np.random.normal(0, phase['pos_step'])
            elif r < 0.75:
                # perturb one inner hexagon angle
                idx = np.random.randint(0, N)
                nb[idx * 3 + 2] += np.random.normal(0, phase['ang_step'])
            elif r < 0.87:
                # perturb outer center
                dim = np.random.randint(0, 2)
                nb[N * 3 + dim] += np.random.normal(0, phase['oc_step'])
            else:
                # perturb outer angle
                nb[N * 3 + 2] += np.random.normal(0, phase['oa_step'])

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

    # Extract best
    best_inner_j = []
    for i in range(N):
        best_inner_j.append([float(best_params[i * 3]), float(best_params[i * 3 + 1]), float(best_params[i * 3 + 2])])
    best_oc_j = (float(best_params[N * 3]), float(best_params[N * 3 + 1]))
    best_oa_j = float(best_params[N * 3 + 2])

    return best_inner_j, best_oc_j, best_oa_j, best_params

def final_polish(params, deadline):
    """High-precision polish: smooth NM -> exact NM -> L-BFGS-B."""
    def obj_smooth(p):
        il = []
        for i in range(N):
            il.append([p[i * 3], p[i * 3 + 1], p[i * 3 + 2]])
        ocx, ocy, oa = p[N * 3], p[N * 3 + 1], p[N * 3 + 2]
        side = compute_required_side_smooth(il, (ocx, ocy), oa, beta=60.0)
        pen = overlap_penalty(il)
        return side + 1000.0 * pen

    def obj_exact(p):
        il = []
        for i in range(N):
            il.append([p[i * 3], p[i * 3 + 1], p[i * 3 + 2]])
        ocx, ocy, oa = p[N * 3], p[N * 3 + 1], p[N * 3 + 2]
        side = compute_required_side_fast(il, (ocx, ocy), oa)
        pen = overlap_penalty(il)
        return side + 5000.0 * pen

    best_p = params.copy()
    best_v = obj_exact(best_p)
    
    # Stage 1: Nelder-Mead on smooth objective
    if time.time() < deadline - 8:
        try:
            res = minimize(obj_smooth, best_p, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 4000, 'adaptive': True})
            if res.fun < best_v:
                best_v = res.fun
                best_p = res.x.copy()
        except Exception:
            pass
    
    # Stage 2: Nelder-Mead on exact objective
    if time.time() < deadline - 4:
        try:
            res = minimize(obj_exact, best_p, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': 6000, 'adaptive': True})
            if res.fun < best_v:
                best_v = res.fun
                best_p = res.x.copy()
        except Exception:
            pass
    
    # Stage 3: L-BFGS-B with tight bounds
    if time.time() < deadline - 2:
        il_tmp = []
        for i in range(N):
            il_tmp.append([best_p[i * 3], best_p[i * 3 + 1], best_p[i * 3 + 2]])
        cx = sum(p[0] for p in il_tmp) / N
        cy = sum(p[1] for p in il_tmp) / N
        bounds = []
        for i in range(N):
            bounds.append((best_p[i * 3] - 0.3, best_p[i * 3] + 0.3))
            bounds.append((best_p[i * 3 + 1] - 0.3, best_p[i * 3 + 1] + 0.3))
            bounds.append((-180.0, 180.0))
        bounds.append((best_p[N * 3] - 1.0, best_p[N * 3] + 1.0))
        bounds.append((best_p[N * 3 + 1] - 1.0, best_p[N * 3 + 1] + 1.0))
        bounds.append((best_p[N * 3 + 2] - 15.0, best_p[N * 3 + 2] + 15.0))
        try:
            res = minimize(obj_exact, best_p, method='L-BFGS-B',
                           bounds=bounds, options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 8000})
            if res.fun < best_v:
                best_v = res.fun
                best_p = res.x.copy()
        except Exception:
            pass
    
    return best_p

def construct():
    t0 = time.time()
    deadline = t0 + 148.0  # 2s safety margin

    # --- Define multiple candidate configurations ---
    # Config 1: Standard hexagonal flower (1+6+4)
    axial1 = [(0, 0),
              (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
              (2, -1), (1, 1), (-2, 1), (-1, -1)]
    
    # Config 2: Row-based 4-3-4
    axial2 = [(-1, -1), (0, -1), (1, -1), (2, -1),
              (-1, 0), (0, 0), (1, 0),
              (-1, 1), (0, 1), (1, 1), (2, 1)]
    
    # Config 3: Compact 3-5-3
    axial3 = [(-1, -1), (0, -1), (1, -1),
              (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0),
              (-1, 1), (0, 1), (1, 1)]
    
    # Config 4: Another variant - 5-3-3
    axial4 = [(-2, -1), (-1, -1), (0, -1), (1, -1), (2, -1),
              (-1, 0), (0, 0), (1, 0),
              (-1, 1), (0, 1), (1, 1)]
    
    # Config 5: Centered with different outer placement
    axial5 = [(0, 0),
              (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
              (2, 0), (0, 2), (-2, 1), (-2, -1)]
    
    # Config 6: 2-4-3-2 diamond
    axial6 = [(0, -2), (1, -2),
              (-1, -1), (0, -1), (1, -1), (2, -1),
              (-1, 0), (0, 0), (1, 0),
              (0, 1), (1, 1)]
    
    # Config 7: 3-4-4
    axial7 = [(-1, -1), (0, -1), (1, -1),
              (-2, 0), (-1, 0), (0, 0), (1, 0),
              (-1, 1), (0, 1), (1, 1), (2, 1)]
    
    # Config 8: 4-4-3 staggered
    axial8 = [(-1, -1), (0, -1), (1, -1), (2, -1),
              (-2, 0), (-1, 0), (0, 0), (1, 0),
              (0, 1), (1, 1), (2, 1)]
    
    # Config 9: 3-4-4 shifted
    axial9 = [(-2, -1), (-1, -1), (0, -1),
              (-1, 0), (0, 0), (1, 0), (2, 0),
              (0, 1), (1, 1), (2, 1), (3, 1)]
    
    configs = [axial1, axial2, axial3, axial4, axial5, axial6, axial7, axial8, axial9]
    
    best_side = float('inf')
    best_inner = None
    best_outer_center = None
    best_outer_angle = 0.0
    best_params = None
    
    # Phase 1a: Rigid composite optimization for each config (fast, 6 DOF)
    for ci, axial in enumerate(configs):
        if time.time() > deadline - 60:
            break
        try:
            r_inner, r_oc, r_oa, r_side = optimize_rigid_composite(axial, deadline)
            if r_side < best_side:
                best_side = r_side
                best_inner = r_inner
                best_outer_center = r_oc
                best_outer_angle = r_oa
                p0 = np.zeros(N * 3 + 3)
                for i, (x, y, a) in enumerate(best_inner):
                    p0[i * 3] = x
                    p0[i * 3 + 1] = y
                    p0[i * 3 + 2] = a
                p0[N * 3] = r_oc[0]
                p0[N * 3 + 1] = r_oc[1]
                p0[N * 3 + 2] = r_oa
                best_params = p0.copy()
        except Exception:
            pass
    
    # Phase 1b: Also try orientation optimization from lattice configs
    for ci, axial in enumerate(configs):
        if time.time() > deadline - 40:
            break
        s0 = 1.0 + 1e-8
        init_inner = build_lattice_config(axial, s0)
        cx0 = sum(c[0] for c in init_inner) / len(init_inner)
        cy0 = sum(c[1] for c in init_inner) / len(init_inner)
        
        # Fast orientation search: coarse angle sweep + quick NM polish
        oc, oa, side_val = try_orientation_optimization(init_inner, (cx0, cy0), 0.0, deadline)
        
        if side_val < best_side:
            best_side = side_val
            best_inner = init_inner
            best_outer_center = oc
            best_outer_angle = oa
            p0 = np.zeros(N * 3 + 3)
            for i, (x, y, a) in enumerate(init_inner):
                p0[i * 3] = x
                p0[i * 3 + 1] = y
                p0[i * 3 + 2] = a
            p0[N * 3] = oc[0]
            p0[N * 3 + 1] = oc[1]
            p0[N * 3 + 2] = oa
            best_params = p0.copy()
    
    if best_inner is None:
        # Fallback
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
    
    # Phase 2: Joint optimization from best config
    if time.time() < deadline - 40:
        joint_inner, joint_oc, joint_oa, joint_params = joint_optimize(
            best_inner, best_outer_center, best_outer_angle, deadline)
        
        # Try to repair overlaps via local separation
        if not check_disjoint(joint_inner, eps=1e-5):
            for repair_iter in range(50):
                if time.time() > deadline - 35:
                    break
                polys = [hex_vertices(x, y, 1.0, math.radians(a)) for x, y, a in joint_inner]
                separated = True
                for i in range(N):
                    for j in range(i + 1, N):
                        if sat_overlap(polys[i], polys[j], eps=-1e-7):
                            separated = False
                            dx = joint_inner[j][0] - joint_inner[i][0]
                            dy = joint_inner[j][1] - joint_inner[i][1]
                            d = math.hypot(dx, dy)
                            if d > 1e-6:
                                push = 0.02
                                joint_inner[i][0] -= push * dx / d
                                joint_inner[i][1] -= push * dy / d
                                joint_inner[j][0] += push * dx / d
                                joint_inner[j][1] += push * dy / d
                            else:
                                joint_inner[i][0] -= 0.02
                                joint_inner[j][0] += 0.02
                            polys = [hex_vertices(x, y, 1.0, math.radians(a)) for x, y, a in joint_inner]
                            break
                    if not separated:
                        break
                if separated:
                    break
        
        if check_disjoint(joint_inner, eps=1e-5):
            joint_side = compute_required_side_fast(joint_inner, joint_oc, joint_oa)
            if joint_side < best_side:
                best_inner = joint_inner
                best_outer_center = joint_oc
                best_outer_angle = joint_oa
                best_side = joint_side
                best_params = joint_params
        else:
            test_inner = []
            for i in range(N):
                test_inner.append([float(joint_params[i * 3]), float(joint_params[i * 3 + 1]), float(joint_params[i * 3 + 2])])
            if check_disjoint(test_inner, eps=1e-4):
                test_side = compute_required_side_fast(test_inner, (float(joint_params[N*3]), float(joint_params[N*3+1])), float(joint_params[N*3+2]))
                if test_side < best_side:
                    best_inner = test_inner
                    best_outer_center = (float(joint_params[N*3]), float(joint_params[N*3+1]))
                    best_outer_angle = float(joint_params[N*3+2])
                    best_side = test_side
    
    # Phase 2b: Also try joint optimization from second-best config if time permits
    if time.time() < deadline - 30 and best_params is not None:
        np.random.seed(789)
        p2 = best_params.copy()
        for i in range(N):
            p2[i*3] += np.random.normal(0, 0.08)
            p2[i*3+1] += np.random.normal(0, 0.08)
            p2[i*3+2] += np.random.normal(0, 8.0)
        
        init_inner2 = []
        for i in range(N):
            init_inner2.append([float(p2[i*3]), float(p2[i*3+1]), float(p2[i*3+2])])
        init_oc2 = (float(p2[N*3]), float(p2[N*3+1]))
        init_oa2 = float(p2[N*3+2])
        
        joint_inner2, joint_oc2, joint_oa2, joint_params2 = joint_optimize(
            init_inner2, init_oc2, init_oa2, deadline)
        
        # Repair overlaps
        if not check_disjoint(joint_inner2, eps=1e-5):
            for repair_iter in range(50):
                if time.time() > deadline - 25:
                    break
                polys = [hex_vertices(x, y, 1.0, math.radians(a)) for x, y, a in joint_inner2]
                separated = True
                for i in range(N):
                    for j in range(i + 1, N):
                        if sat_overlap(polys[i], polys[j], eps=-1e-7):
                            separated = False
                            dx = joint_inner2[j][0] - joint_inner2[i][0]
                            dy = joint_inner2[j][1] - joint_inner2[i][1]
                            d = math.hypot(dx, dy)
                            if d > 1e-6:
                                push = 0.02
                                joint_inner2[i][0] -= push * dx / d
                                joint_inner2[i][1] -= push * dy / d
                                joint_inner2[j][0] += push * dx / d
                                joint_inner2[j][1] += push * dy / d
                            else:
                                joint_inner2[i][0] -= 0.02
                                joint_inner2[j][0] += 0.02
                            polys = [hex_vertices(x, y, 1.0, math.radians(a)) for x, y, a in joint_inner2]
                            break
                    if not separated:
                        break
                if separated:
                    break
        
        if check_disjoint(joint_inner2, eps=1e-5):
            joint_side2 = compute_required_side_fast(joint_inner2, joint_oc2, joint_oa2)
            if joint_side2 < best_side:
                best_inner = joint_inner2
                best_outer_center = joint_oc2
                best_outer_angle = joint_oa2
                best_side = joint_side2
                best_params = joint_params2
    
    # Phase 3: Final high-precision polish with L-BFGS-B
    if time.time() < deadline - 5 and best_params is not None:
        polished = final_polish(best_params, deadline)
        polish_inner = []
        for i in range(N):
            polish_inner.append([float(polished[i * 3]), float(polished[i * 3 + 1]), float(polished[i * 3 + 2])])
        polish_oc = (float(polished[N * 3]), float(polished[N * 3 + 1]))
        polish_oa = float(polished[N * 3 + 2])
        
        # Repair if needed
        if not check_disjoint(polish_inner, eps=1e-5):
            for repair_iter in range(30):
                if time.time() > deadline - 3:
                    break
                polys = [hex_vertices(x, y, 1.0, math.radians(a)) for x, y, a in polish_inner]
                separated = True
                for i in range(N):
                    for j in range(i + 1, N):
                        if sat_overlap(polys[i], polys[j], eps=-1e-7):
                            separated = False
                            dx = polish_inner[j][0] - polish_inner[i][0]
                            dy = polish_inner[j][1] - polish_inner[i][1]
                            d = math.hypot(dx, dy)
                            if d > 1e-6:
                                push = 0.01
                                polish_inner[i][0] -= push * dx / d
                                polish_inner[i][1] -= push * dy / d
                                polish_inner[j][0] += push * dx / d
                                polish_inner[j][1] += push * dy / d
                            else:
                                polish_inner[i][0] -= 0.01
                                polish_inner[j][0] += 0.01
                            polys = [hex_vertices(x, y, 1.0, math.radians(a)) for x, y, a in polish_inner]
                            break
                    if not separated:
                        break
                if separated:
                    break
        
        if check_disjoint(polish_inner, eps=1e-5):
            polish_side = compute_required_side_fast(polish_inner, polish_oc, polish_oa)
            if polish_side < best_side:
                best_inner = polish_inner
                best_outer_center = polish_oc
                best_outer_angle = polish_oa
                best_side = polish_side
    
    # Final verification with repair
    if not check_disjoint(best_inner, eps=1e-5):
        for repair_iter in range(50):
            polys = [hex_vertices(x, y, 1.0, math.radians(a)) for x, y, a in best_inner]
            separated = True
            for i in range(N):
                for j in range(i + 1, N):
                    if sat_overlap(polys[i], polys[j], eps=-1e-7):
                        separated = False
                        dx = best_inner[j][0] - best_inner[i][0]
                        dy = best_inner[j][1] - best_inner[i][1]
                        d = math.hypot(dx, dy)
                        if d > 1e-6:
                            push = 0.01
                            best_inner[i][0] -= push * dx / d
                            best_inner[i][1] -= push * dy / d
                            best_inner[j][0] += push * dx / d
                            best_inner[j][1] += push * dy / d
                        else:
                            best_inner[i][0] -= 0.01
                            best_inner[j][0] += 0.01
                        polys = [hex_vertices(x, y, 1.0, math.radians(a)) for x, y, a in best_inner]
                        break
                if not separated:
                    break
            if separated:
                break
        
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
