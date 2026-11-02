"""
Optimized hexagon packing: minimize outer hexagon side length for N unit hexagons.
Combines rigid-composite optimization with smooth support-function objective,
differential evolution for global search, multi-start Nelder-Mead polish,
semi-rigid refinement, and high-precision final optimization with L-BFGS-B.
"""
import math
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

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

def rigid_composite_objective(p, base_positions, cx0, cy0, beta=60.0):
    """
    Objective for rigid composite: p = [comp_angle_deg, dx, dy, ox, oy, outer_angle_deg]
    Returns smooth required side length.
    """
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
    
    return compute_required_side_smooth(inner, (ox, oy), outer_angle, beta=beta)

def rigid_composite_objective_exact(p, base_positions, cx0, cy0):
    """Exact (nonsmooth) objective for rigid composite."""
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

def optimize_rigid_composite(axial_coords, deadline):
    """
    Optimize rigid composite: global rotation, translation, outer center, outer angle.
    Uses differential evolution for global search, then Nelder-Mead polish.
    """
    s0 = 1.0 + 1e-8
    centers = [(1.5 * s0 * q, SQRT3 * s0 * (r + q / 2.0)) for q, r in axial_coords]
    base_positions = [[x, y, 0.0] for x, y in centers]
    
    cx0 = sum(c[0] for c in centers) / len(centers)
    cy0 = sum(c[1] for c in centers) / len(centers)
    
    # Bounds for differential evolution
    bounds = [(-30, 90), (-3, 3), (-3, 3), (-4, 4), (-4, 4), (-30, 90)]
    
    def obj_de(p):
        return rigid_composite_objective(p, base_positions, cx0, cy0, beta=50.0)
    
    best_val = float('inf')
    best_p = None
    
    # Try differential evolution for global search
    try:
        if time.time() < deadline - 40:
            de_result = differential_evolution(
                obj_de, bounds, seed=42, maxiter=50, tol=1e-8,
                popsize=15, mutation=(0.3, 0.9), recombination=0.7,
                polish=False, init='sobol'
            )
            if de_result.fun < best_val:
                best_val = de_result.fun
                best_p = de_result.x.copy()
    except Exception:
        pass
    
    # Multi-start Nelder-Mead from various initial points
    np.random.seed(42)
    n_starts = 8
    for si in range(n_starts):
        if time.time() > deadline - 30:
            break
        if si == 0 and best_p is not None:
            p0 = best_p.copy()
        else:
            p0 = np.array([
                np.random.uniform(-15, 75),
                np.random.uniform(-1.5, 1.5),
                np.random.uniform(-1.5, 1.5),
                np.random.uniform(-2, 2),
                np.random.uniform(-2, 2),
                np.random.uniform(-15, 75)
            ])
        
        try:
            r1 = minimize(rigid_composite_objective, p0, args=(base_positions, cx0, cy0, 50.0),
                          method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-9, 'fatol': 1e-11, 'maxfev': 500, 'adaptive': True})
            r2 = minimize(rigid_composite_objective_exact, r1.x, args=(base_positions, cx0, cy0),
                          method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': 1000, 'adaptive': True})
            if r2.fun < best_val:
                best_val = r2.fun
                best_p = r2.x.copy()
        except Exception:
            pass
    
    if best_p is None:
        inner = [[x, y, 0.0] for x, y in centers]
        return inner, (cx0, cy0), 0.0, compute_required_side_fast(inner, (cx0, cy0), 0.0)
    
    comp_angle = float(best_p[0])
    dx, dy = float(best_p[1]), float(best_p[2])
    ox, oy = float(best_p[3]), float(best_p[4])
    outer_angle = float(best_p[5])
    
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
    
    return inner, (ox, oy), outer_angle, best_val

def semi_rigid_optimize(inner, outer_center, outer_angle, deadline):
    """
    Semi-rigid: allow small individual angle perturbations while maintaining
    approximate lattice structure. Use SA with smooth objective.
    """
    def obj_semi(params, beta=70.0):
        il = []
        for i in range(N):
            il.append([params[i * 3], params[i * 3 + 1], params[i * 3 + 2]])
        ocx, ocy, oa = params[N * 3], params[N * 3 + 1], params[N * 3 + 2]
        side = compute_required_side_smooth(il, (ocx, ocy), oa, beta=beta)
        angle_pen = 0.0
        for i in range(N):
            da = params[i * 3 + 2] - inner[i][2]
            da = ((da + 180) % 360) - 180
            angle_pen += (da / 10.0) ** 2
        pos_pen = 0.0
        for i in range(N):
            dx = params[i * 3] - inner[i][0]
            dy = params[i * 3 + 1] - inner[i][1]
            pos_pen += (dx ** 2 + dy ** 2) / 100.0
        return side + 10.0 * angle_pen + 5.0 * pos_pen
    
    def obj_semi_exact(params):
        il = []
        for i in range(N):
            il.append([params[i * 3], params[i * 3 + 1], params[i * 3 + 2]])
        ocx, ocy, oa = params[N * 3], params[N * 3 + 1], params[N * 3 + 2]
        side = compute_required_side_fast(il, (ocx, ocy), oa)
        pen = overlap_penalty(il)
        return side + 2000.0 * pen
    
    p0 = np.zeros(N * 3 + 3)
    for i, (x, y, a) in enumerate(inner):
        p0[i * 3] = x
        p0[i * 3 + 1] = y
        p0[i * 3 + 2] = a
    p0[N * 3] = outer_center[0]
    p0[N * 3 + 1] = outer_center[1]
    p0[N * 3 + 2] = outer_angle
    
    current = p0.copy()
    current_val = obj_semi(current)
    best_params = current.copy()
    best_val = current_val
    
    T = 0.5
    T_min = 0.0005
    cool = 0.99985
    max_iter = 8000
    np.random.seed(789)
    
    for it in range(max_iter):
        if time.time() > deadline - 20:
            break
        nb = current.copy()
        r = np.random.random()
        if r < 0.50:
            idx = np.random.randint(0, N)
            dim = np.random.randint(0, 2)
            nb[idx * 3 + dim] += np.random.normal(0, 0.08)
        elif r < 0.70:
            idx = np.random.randint(0, N)
            nb[idx * 3 + 2] += np.random.normal(0, 3.0)
        elif r < 0.85:
            dim = np.random.randint(0, 2)
            nb[N * 3 + dim] += np.random.normal(0, 0.06)
        else:
            nb[N * 3 + 2] += np.random.normal(0, 2.0)
        
        nb_val = obj_semi(nb)
        if nb_val < current_val or np.random.random() < math.exp(-(nb_val - current_val) / max(T, 1e-10)):
            current = nb
            current_val = nb_val
            if current_val < best_val:
                best_params = current.copy()
                best_val = current_val
        T *= cool
        if T < T_min:
            break
    
    try:
        res = minimize(obj_semi_exact, best_params, method='Nelder-Mead',
                       options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': 5000, 'adaptive': True})
        if res.fun < best_val:
            best_params = res.x
            best_val = res.fun
    except Exception:
        pass
    
    result_inner = []
    for i in range(N):
        result_inner.append([float(best_params[i * 3]), float(best_params[i * 3 + 1]), float(best_params[i * 3 + 2])])
    result_oc = (float(best_params[N * 3]), float(best_params[N * 3 + 1]))
    result_oa = float(best_params[N * 3 + 2])
    
    return result_inner, result_oc, result_oa

def full_joint_optimize(init_inner, init_oc, init_oa, deadline, seed=42):
    """Full joint optimization with simulated annealing using smooth surrogate."""
    def obj_joint_smooth(params, beta=60.0):
        il = []
        for i in range(N):
            il.append([params[i * 3], params[i * 3 + 1], params[i * 3 + 2]])
        ocx, ocy, oa = params[N * 3], params[N * 3 + 1], params[N * 3 + 2]
        side = compute_required_side_smooth(il, (ocx, ocy), oa, beta=beta)
        pen = overlap_penalty(il)
        return side + 500.0 * pen
    
    def obj_joint_exact(params):
        il = []
        for i in range(N):
            il.append([params[i * 3], params[i * 3 + 1], params[i * 3 + 2]])
        ocx, ocy, oa = params[N * 3], params[N * 3 + 1], params[N * 3 + 2]
        side = compute_required_side_fast(il, (ocx, ocy), oa)
        pen = overlap_penalty(il)
        return side + 3000.0 * pen
    
    p0 = np.zeros(N * 3 + 3)
    for i, (x, y, a) in enumerate(init_inner):
        p0[i * 3] = x
        p0[i * 3 + 1] = y
        p0[i * 3 + 2] = a
    p0[N * 3] = init_oc[0]
    p0[N * 3 + 1] = init_oc[1]
    p0[N * 3 + 2] = init_oa
    
    current = p0.copy()
    current_val = obj_joint_smooth(current)
    best_params = current.copy()
    best_val_j = current_val
    
    T = 1.0
    T_min = 0.001
    cool = 0.9998
    max_iter = 10000
    np.random.seed(seed)
    for it in range(max_iter):
        if time.time() > deadline - 25:
            break
        nb = current.copy()
        r = np.random.random()
        if r < 0.60:
            idx = np.random.randint(0, N)
            dim = np.random.randint(0, 2)
            nb[idx * 3 + dim] += np.random.normal(0, 0.15)
        elif r < 0.75:
            idx = np.random.randint(0, N)
            nb[idx * 3 + 2] += np.random.normal(0, 6.0)
        elif r < 0.88:
            dim = np.random.randint(0, 2)
            nb[N * 3 + dim] += np.random.normal(0, 0.10)
        else:
            nb[N * 3 + 2] += np.random.normal(0, 3.0)
        
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
    
    try:
        res = minimize(obj_joint_exact, best_params, method='Nelder-Mead',
                       options={'maxiter': 4000, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': 6000, 'adaptive': True})
        if res.fun < best_val_j:
            best_params = res.x
            best_val_j = res.fun
    except Exception:
        pass
    
    best_inner_j = []
    for i in range(N):
        best_inner_j.append([float(best_params[i * 3]), float(best_params[i * 3 + 1]), float(best_params[i * 3 + 2])])
    best_oc_j = (float(best_params[N * 3]), float(best_params[N * 3 + 1]))
    best_oa_j = float(best_params[N * 3 + 2])
    
    return best_inner_j, best_oc_j, best_oa_j

def final_polish(params, deadline):
    """High-precision multi-stage polish: NM smooth -> NM exact -> L-BFGS-B."""
    def obj_smooth(p, beta=100.0):
        il = []
        for i in range(N):
            il.append([p[i * 3], p[i * 3 + 1], p[i * 3 + 2]])
        ocx, ocy, oa = p[N * 3], p[N * 3 + 1], p[N * 3 + 2]
        side = compute_required_side_smooth(il, (ocx, ocy), oa, beta=beta)
        pen = overlap_penalty(il)
        return side + 5000.0 * pen
    
    def obj_exact(p):
        il = []
        for i in range(N):
            il.append([p[i * 3], p[i * 3 + 1], p[i * 3 + 2]])
        ocx, ocy, oa = p[N * 3], p[N * 3 + 1], p[N * 3 + 2]
        side = compute_required_side_fast(il, (ocx, ocy), oa)
        pen = overlap_penalty(il)
        return side + 5000.0 * pen
    
    best_p = params.copy()
    best_v = obj_exact(params)
    
    # Stage 1: NM on smooth with high beta
    if time.time() < deadline - 10:
        try:
            r = minimize(obj_smooth, best_p, method='Nelder-Mead',
                         options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': 7000, 'adaptive': True})
            if r.fun < best_v:
                best_v = r.fun
                best_p = r.x.copy()
        except Exception:
            pass
    
    # Stage 2: NM on exact
    if time.time() < deadline - 5:
        try:
            r = minimize(obj_exact, best_p, method='Nelder-Mead',
                         options={'maxiter': 8000, 'xatol': 1e-13, 'fatol': 1e-15, 'maxfev': 12000, 'adaptive': True})
            if r.fun < best_v:
                best_v = r.fun
                best_p = r.x.copy()
        except Exception:
            pass
    
    # Stage 3: L-BFGS-B with bounds
    if time.time() < deadline - 2:
        cx = sum(best_p[i * 3] for i in range(N)) / N
        cy = sum(best_p[i * 3 + 1] for i in range(N)) / N
        bounds = []
        for i in range(N):
            bounds.append((best_p[i * 3] - 0.5, best_p[i * 3] + 0.5))
            bounds.append((best_p[i * 3 + 1] - 0.5, best_p[i * 3 + 1] + 0.5))
            bounds.append((-180.0, 180.0))
        bounds.append((cx - 2.0, cx + 2.0))
        bounds.append((cy - 2.0, cy + 2.0))
        bounds.append((-30.0, 30.0))
        try:
            r = minimize(obj_exact, best_p, method='L-BFGS-B',
                         bounds=bounds, options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-13, 'maxfun': 8000})
            if r.fun < best_v:
                best_v = r.fun
                best_p = r.x.copy()
        except Exception:
            pass
    
    return best_p

def construct():
    t0 = time.time()
    deadline = t0 + 140.0

    # --- Define multiple candidate polyhex topologies ---
    axial1 = [(0, 0),
              (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
              (2, -1), (1, 1), (-2, 1), (-1, -1)]
    
    axial2 = [(-1, -1), (0, -1), (1, -1),
              (-1, 0), (0, 0), (1, 0),
              (-1, 1), (0, 1), (1, 1),
              (2, -1), (2, 1)]
    
    axial3 = [(-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0),
              (-1, -1), (0, -1), (1, -1),
              (-1, 1), (0, 1), (1, 1)]
    
    axial4 = [(0, 0),
              (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
              (2, 0), (0, 2), (-2, 1), (-2, -1)]
    
    axial5 = [(0, 0), (1, 0), (0, 1), (1, 1),
              (-1, 0), (-1, 1), (-2, 1),
              (2, -1), (2, 0), (1, -1), (0, -1)]
    
    axial6 = [(0, 0),
              (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
              (2, -1), (2, 0), (-2, 1), (-2, 0)]
    
    configs = [axial1, axial2, axial3, axial4, axial5, axial6]
    
    best_side = float('inf')
    best_inner = None
    best_outer_center = None
    best_outer_angle = 0.0
    best_params = None
    
    # Phase 1: Rigid composite optimization for each topology
    for ci, axial in enumerate(configs):
        if time.time() > deadline - 50:
            break
        inner, oc, oa, side_val = optimize_rigid_composite(axial, deadline)
        
        if check_disjoint(inner, eps=1e-5) and side_val < best_side:
            best_side = side_val
            best_inner = inner
            best_outer_center = oc
            best_outer_angle = oa
            p0 = np.zeros(N * 3 + 3)
            for i, (x, y, a) in enumerate(inner):
                p0[i * 3] = x
                p0[i * 3 + 1] = y
                p0[i * 3 + 2] = a
            p0[N * 3] = oc[0]
            p0[N * 3 + 1] = oc[1]
            p0[N * 3 + 2] = oa
            best_params = p0.copy()
    
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
    
    # Phase 2: Semi-rigid refinement
    if time.time() < deadline - 35:
        semi_inner, semi_oc, semi_oa = semi_rigid_optimize(
            best_inner, best_outer_center, best_outer_angle, deadline)
        
        if check_disjoint(semi_inner, eps=1e-5):
            semi_side = compute_required_side_fast(semi_inner, semi_oc, semi_oa)
            if semi_side < best_side:
                best_inner = semi_inner
                best_outer_center = semi_oc
                best_outer_angle = semi_oa
                best_side = semi_side
                p0 = np.zeros(N * 3 + 3)
                for i, (x, y, a) in enumerate(best_inner):
                    p0[i * 3] = x
                    p0[i * 3 + 1] = y
                    p0[i * 3 + 2] = a
                p0[N * 3] = semi_oc[0]
                p0[N * 3 + 1] = semi_oc[1]
                p0[N * 3 + 2] = semi_oa
                best_params = p0.copy()
    
    # Phase 3: Full joint optimization (SA)
    if time.time() < deadline - 30:
        joint_inner, joint_oc, joint_oa = full_joint_optimize(
            best_inner, best_outer_center, best_outer_angle, deadline, seed=123)
        
        if check_disjoint(joint_inner, eps=1e-5):
            joint_side = compute_required_side_fast(joint_inner, joint_oc, joint_oa)
            if joint_side < best_side:
                best_inner = joint_inner
                best_outer_center = joint_oc
                best_outer_angle = joint_oa
                best_side = joint_side
                p0 = np.zeros(N * 3 + 3)
                for i, (x, y, a) in enumerate(best_inner):
                    p0[i * 3] = x
                    p0[i * 3 + 1] = y
                    p0[i * 3 + 2] = a
                p0[N * 3] = joint_oc[0]
                p0[N * 3 + 1] = joint_oc[1]
                p0[N * 3 + 2] = joint_oa
                best_params = p0.copy()
    
    # Phase 3b: Second SA pass
    if time.time() < deadline - 20:
        joint_inner2, joint_oc2, joint_oa2 = full_joint_optimize(
            best_inner, best_outer_center, best_outer_angle, deadline, seed=456)
        
        if check_disjoint(joint_inner2, eps=1e-5):
            joint_side2 = compute_required_side_fast(joint_inner2, joint_oc2, joint_oa2)
            if joint_side2 < best_side:
                best_inner = joint_inner2
                best_outer_center = joint_oc2
                best_outer_angle = joint_oa2
                best_side = joint_side2
                p0 = np.zeros(N * 3 + 3)
                for i, (x, y, a) in enumerate(best_inner):
                    p0[i * 3] = x
                    p0[i * 3 + 1] = y
                    p0[i * 3 + 2] = a
                p0[N * 3] = joint_oc2[0]
                p0[N * 3 + 1] = joint_oc2[1]
                p0[N * 3 + 2] = joint_oa2
                best_params = p0.copy()
    
    # Phase 4: Final high-precision multi-stage polish
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
