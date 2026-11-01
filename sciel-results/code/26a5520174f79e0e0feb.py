"""Combined program: contact-graph packing with rigid-body composite optimization and
smooth surrogate joint refinement. Merges multi-topology lattice configs from the current
program with the rigid-composite 6-DOF optimization from the inspiration, plus improved
smooth surrogates and tighter final polish."""
import math
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar

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
    Smooth (log-sum-exp) surrogate for the max projection over all vertices and all normals.
    Uses soft-max over absolute projections to handle both positive and negative directions.
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
    # Coarse angle sweep
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
    # Fine sweep around best angle
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

def optimize_rigid_composite(axial_coords, deadline):
    """
    Treat 11 hexagons as a rigid composite body. Optimize 6 DOF:
    composite rotation, translation (dx, dy), outer center (ox, oy), outer angle.
    This is a lower-dimensional problem that can find good global orientations.
    """
    s0 = 1.0 + 1e-8
    centers = [(1.5 * s0 * q, SQRT3 * s0 * (r + q / 2.0)) for q, r in axial_coords]
    base_positions = [[x, y, 0.0] for x, y in centers]
    cx0 = sum(c[0] for c in centers) / len(centers)
    cy0 = sum(c[1] for c in centers) / len(centers)
    
    def obj_rigid_smooth(p):
        comp_angle = p[0]
        dx, dy = p[1], p[2]
        ox, oy = p[3], p[4]
        outer_angle = p[5]
        ca = math.cos(math.radians(comp_angle))
        sa = math.sin(math.radians(comp_angle))
        inner = []
        for i in range(N):
            bx, by, ba = base_positions[i]
            rx = bx - cx0
            ry = by - cy0
            nx = cx0 + ca * rx - sa * ry + dx
            ny = cy0 + sa * rx + ca * ry + dy
            na = ba + comp_angle
            inner.append([nx, ny, na])
        return compute_required_side_smooth(inner, (ox, oy), outer_angle, beta=50.0)
    
    def obj_rigid_exact(p):
        comp_angle = p[0]
        dx, dy = p[1], p[2]
        ox, oy = p[3], p[4]
        outer_angle = p[5]
        ca = math.cos(math.radians(comp_angle))
        sa = math.sin(math.radians(comp_angle))
        inner = []
        for i in range(N):
            bx, by, ba = base_positions[i]
            rx = bx - cx0
            ry = by - cy0
            nx = cx0 + ca * rx - sa * ry + dx
            ny = cy0 + sa * rx + ca * ry + dy
            na = ba + comp_angle
            inner.append([nx, ny, na])
        return compute_required_side_fast(inner, (ox, oy), outer_angle)
    
    best_val = float('inf')
    best_p = None
    
    # Multi-start: try different initial composite angles and outer angles
    np.random.seed(42)
    starts = []
    for comp_a in range(0, 60, 10):
        for outer_a in range(0, 60, 10):
            starts.append([float(comp_a), 0.0, 0.0, cx0, cy0, float(outer_a)])
    # Add some random starts
    for _ in range(6):
        starts.append([float(np.random.uniform(0, 60)), float(np.random.uniform(-1, 1)),
                       float(np.random.uniform(-1, 1)), cx0, cy0, float(np.random.uniform(0, 60))])
    
    for p0 in starts:
        if time.time() > deadline - 20:
            break
        try:
            r1 = minimize(obj_rigid_smooth, p0, method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 500})
            if r1.fun < best_val:
                best_val = r1.fun
                best_p = r1.x.copy()
        except Exception:
            pass
    
    # Polish the best with exact objective
    if best_p is not None and time.time() < deadline - 10:
        try:
            r2 = minimize(obj_rigid_exact, best_p, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 800})
            if r2.fun < best_val:
                best_val = r2.fun
                best_p = r2.x.copy()
        except Exception:
            pass
    
    if best_p is None:
        return None
    
    # Extract inner positions
    comp_angle = best_p[0]
    dx, dy = best_p[1], best_p[2]
    ox, oy = best_p[3], best_p[4]
    outer_angle = best_p[5]
    ca = math.cos(math.radians(comp_angle))
    sa = math.sin(math.radians(comp_angle))
    inner = []
    for i in range(N):
        bx, by, ba = base_positions[i]
        rx = bx - cx0
        ry = by - cy0
        nx = cx0 + ca * rx - sa * ry + dx
        ny = cy0 + sa * rx + ca * ry + dy
        na = ba + comp_angle
        inner.append([float(nx), float(ny), float(na)])
    
    return inner, (float(ox), float(oy)), float(outer_angle), best_val

def joint_optimize(init_inner, init_outer_center, init_outer_angle, deadline):
    """Joint optimization of inner positions/angles and outer hexagon."""
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
                idx = np.random.randint(0, N)
                dim = np.random.randint(0, 2)
                nb[idx * 3 + dim] += np.random.normal(0, phase['pos_step'])
            elif r < 0.75:
                idx = np.random.randint(0, N)
                nb[idx * 3 + 2] += np.random.normal(0, phase['ang_step'])
            elif r < 0.87:
                dim = np.random.randint(0, 2)
                nb[N * 3 + dim] += np.random.normal(0, phase['oc_step'])
            else:
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
    
    best_inner_j = []
    for i in range(N):
        best_inner_j.append([float(best_params[i * 3]), float(best_params[i * 3 + 1]), float(best_params[i * 3 + 2])])
    best_oc_j = (float(best_params[N * 3]), float(best_params[N * 3 + 1]))
    best_oa_j = float(best_params[N * 3 + 2])
    
    return best_inner_j, best_oc_j, best_oa_j, best_params

def final_polish(params, deadline):
    """High-precision Nelder-Mead polish with multi-start."""
    def obj_polish(p):
        il = []
        for i in range(N):
            il.append([p[i * 3], p[i * 3 + 1], p[i * 3 + 2]])
        ocx, ocy, oa = p[N * 3], p[N * 3 + 1], p[N * 3 + 2]
        side = compute_required_side_fast(il, (ocx, ocy), oa)
        pen = overlap_penalty(il)
        return side + 5000.0 * pen
    
    best_p = params.copy()
    best_v = obj_polish(best_p)
    
    np.random.seed(456)
    for trial in range(6):
        if time.time() > deadline - 4:
            break
        p_start = params.copy()
        p_start += np.random.normal(0, 0.005, size=p_start.shape)
        try:
            res = minimize(obj_polish, p_start, method='Nelder-Mead',
                           options={'maxiter': 4000, 'xatol': 1e-14, 'fatol': 1e-16, 'maxfev': 8000})
            if res.fun < best_v:
                best_v = res.fun
                best_p = res.x.copy()
        except Exception:
            pass
    
    if time.time() < deadline - 2:
        try:
            res = minimize(obj_polish, best_p, method='Nelder-Mead',
                           options={'maxiter': 8000, 'xatol': 1e-15, 'fatol': 1e-17, 'maxfev': 12000})
            if res.fun < best_v:
                best_p = res.x.copy()
        except Exception:
            pass
    
    return best_p

def repair_overlaps(inner, deadline, max_iters=60, push=0.02):
    """Iteratively push overlapping hexagons apart."""
    for _ in range(max_iters):
        if time.time() > deadline - 3:
            break
        polys = [hex_vertices(x, y, 1.0, math.radians(a)) for x, y, a in inner]
        separated = True
        for i in range(N):
            for j in range(i + 1, N):
                if sat_overlap(polys[i], polys[j], eps=-1e-7):
                    separated = False
                    dx = inner[j][0] - inner[i][0]
                    dy = inner[j][1] - inner[i][1]
                    d = math.hypot(dx, dy)
                    if d > 1e-6:
                        inner[i][0] -= push * dx / d
                        inner[i][1] -= push * dy / d
                        inner[j][0] += push * dx / d
                        inner[j][1] += push * dy / d
                    else:
                        inner[i][0] -= push
                        inner[j][0] += push
                    polys = [hex_vertices(x, y, 1.0, math.radians(a)) for x, y, a in inner]
                    break
            if not separated:
                break
        if separated:
            break
    return inner

def construct():
    t0 = time.time()
    deadline = t0 + 148.0  # 2s safety margin

    # Candidate configurations (axial coordinates)
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
    
    axial6 = [(0, -2), (1, -2),
              (-1, -1), (0, -1), (1, -1), (2, -1),
              (-1, 0), (0, 0), (1, 0),
              (0, 1), (1, 1)]
    
    axial7 = [(-1, -1), (0, -1), (1, -1),
              (-2, 0), (-1, 0), (0, 0), (1, 0),
              (-1, 1), (0, 1), (1, 1), (2, 1)]
    
    configs = [axial1, axial2, axial3, axial4, axial5, axial6, axial7]
    
    best_side = float('inf')
    best_inner = None
    best_outer_center = None
    best_outer_angle = 0.0
    best_params = None
    
    # Phase 1a: Rigid composite optimization for each config
    for ci, axial in enumerate(configs):
        if time.time() > deadline - 70:
            break
        result = optimize_rigid_composite(axial, deadline)
        if result is not None:
            r_inner, r_oc, r_oa, r_val = result
            if r_val < best_side:
                best_side = r_val
                best_inner = r_inner
                best_outer_center = r_oc
                best_outer_angle = r_oa
                p0 = np.zeros(N * 3 + 3)
                for i, (x, y, a) in enumerate(r_inner):
                    p0[i * 3] = x
                    p0[i * 3 + 1] = y
                    p0[i * 3 + 2] = a
                p0[N * 3] = r_oc[0]
                p0[N * 3 + 1] = r_oc[1]
                p0[N * 3 + 2] = r_oa
                best_params = p0.copy()
    
    # Phase 1b: If rigid composite didn't find anything, fall back to lattice + orientation
    if best_inner is None:
        for ci, axial in enumerate(configs):
            if time.time() > deadline - 60:
                break
            s0 = 1.0 + 1e-8
            init_inner = build_lattice_config(axial, s0)
            cx0 = sum(c[0] for c in init_inner) / len(init_inner)
            cy0 = sum(c[1] for c in init_inner) / len(init_inner)
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
        s0 = 1.0 + 1e-8
        best_inner = build_lattice_config(configs[0], s0)
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
        
        joint_inner = repair_overlaps(joint_inner, deadline)
        
        if check_disjoint(joint_inner, eps=1e-5):
            joint_side = compute_required_side_fast(joint_inner, joint_oc, joint_oa)
            if joint_side < best_side:
                best_inner = joint_inner
                best_outer_center = joint_oc
                best_outer_angle = joint_oa
                best_side = joint_side
                best_params = joint_params
    
    # Phase 2b: Joint optimization from perturbed best
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
        
        joint_inner2 = repair_overlaps(joint_inner2, deadline)
        
        if check_disjoint(joint_inner2, eps=1e-5):
            joint_side2 = compute_required_side_fast(joint_inner2, joint_oc2, joint_oa2)
            if joint_side2 < best_side:
                best_inner = joint_inner2
                best_outer_center = joint_oc2
                best_outer_angle = joint_oa2
                best_side = joint_side2
                best_params = joint_params2
    
    # Phase 3: Final high-precision polish
    if time.time() < deadline - 5 and best_params is not None:
        polished = final_polish(best_params, deadline)
        polish_inner = []
        for i in range(N):
            polish_inner.append([float(polished[i * 3]), float(polished[i * 3 + 1]), float(polished[i * 3 + 2])])
        polish_oc = (float(polished[N * 3]), float(polished[N * 3 + 1]))
        polish_oa = float(polished[N * 3 + 2])
        
        polish_inner = repair_overlaps(polish_inner, deadline, max_iters=30, push=0.01)
        
        if check_disjoint(polish_inner, eps=1e-5):
            polish_side = compute_required_side_fast(polish_inner, polish_oc, polish_oa)
            if polish_side < best_side:
                best_inner = polish_inner
                best_outer_center = polish_oc
                best_outer_angle = polish_oa
                best_side = polish_side
    
    # Final verification
    best_inner = repair_overlaps(best_inner, deadline, max_iters=50, push=0.01)
    
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
