"""Rigid-body composite packing with LP-exact solver and multi-phase optimization."""
import math
import time
import numpy as np
from scipy.optimize import minimize, linprog

N = 11
SQRT3 = math.sqrt(3)
APOTHEM = SQRT3 / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY = 5.0

# EVOLVE-BLOCK-START

def hex_vertices(cx, cy, side, angle_rad):
    thetas = angle_rad + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + side * np.cos(thetas), cy + side * np.sin(thetas)])

def compute_required_side_fast(inner, outer_center, outer_angle_deg):
    """Exact required side length via vertex projections."""
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

def compute_required_side_lp(pts, beta_deg):
    """
    LP-based exact computation: find minimum D such that all point projections
    fit within [-D, D] along 6 normals, plus apothem.
    pts: (n,2) array of all vertex positions in world coords
    beta_deg: rotation of outer hexagon in degrees
    Returns: required side length, center (cx,cy), or (inf, None, None) if infeasible
    """
    if time.time() - START_TIME > TIME_LIMIT - SAFETY:
        return float('inf'), None, None

    normals = []
    for k in range(6):
        ang = math.radians(beta_deg) + math.pi / 6.0 + k * math.pi / 3.0
        normals.append((math.cos(ang), math.sin(ang)))

    c_obj = np.array([0.0, 0.0, 1.0])
    A_ub = []
    b_ub = []

    for nx, ny in normals:
        for px, py in pts:
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-(px * nx + py * ny))
            A_ub.append([nx, ny, -1.0])
            b_ub.append(px * nx + py * ny)

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(None, None), (None, None), (0.0, None)]

    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status == 0:
            D = res.x[2]
            cx = res.x[0]
            cy = res.x[1]
            S = (D + APOTHEM) * 2.0 / SQRT3
            return S, (cx, cy), beta_deg
    except Exception:
        pass
    return float('inf'), None, None

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=50.0):
    """Smooth log-sum-exp surrogate."""
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
    spread = P.max() - P.min()
    eff_beta = min(beta, max(10.0, 50.0 / max(spread, 0.1)))
    smooth_max = m + math.log(np.sum(np.exp(eff_beta * (P - m)))) / eff_beta
    if smooth_max < 0:
        smooth_max = 0.0
    return 2.0 * smooth_max / SQRT3

def sat_overlap(v1, v2, eps=1e-9):
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

def axial_to_cartesian(axial_list, s=1.0 + 1e-8):
    pts = []
    for q, r in axial_list:
        x = SQRT3 * s * (q + r / 2.0)
        y = 1.5 * s * r
        pts.append((x, y))
    return pts

def build_lattice_config(axial_coords, s=1.0 + 1e-8):
    centers = axial_to_cartesian(axial_coords, s)
    return [[x, y, 0.0] for x, y in centers]

def rigid_composite_objective(p, base_positions, cx0, cy0, beta=50.0):
    """p = [comp_angle, dx, dy, ox, oy, outer_angle]"""
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

def rigid_composite_exact(p, base_positions, cx0, cy0):
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

def optimize_composite_rigid(axial_coords, deadline):
    """Rigid body optimization: 6 DOF."""
    s0 = 1.0 + 1e-8
    centers = axial_to_cartesian(axial_coords, s0)
    base_positions = [[x, y, 0.0] for x, y in centers]
    cx0 = sum(c[0] for c in centers) / len(centers)
    cy0 = sum(c[1] for c in centers) / len(centers)

    best_val = float('inf')
    best_p = None

    for comp_a in range(0, 60, 5):
        for outer_a in range(0, 60, 10):
            if time.time() > deadline - 15:
                break
            p0 = np.array([float(comp_a), 0.0, 0.0, cx0, cy0, float(outer_a)])
            try:
                r1 = minimize(lambda p: rigid_composite_objective(p, base_positions, cx0, cy0, beta=50.0),
                              p0, method='Nelder-Mead',
                              options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-9, 'maxfev': 300})
                r2 = minimize(lambda p: rigid_composite_exact(p, base_positions, cx0, cy0),
                              r1.x, method='Nelder-Mead',
                              options={'maxiter': 400, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 600})
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

def semi_rigid_optimize(axial_coords, deadline):
    """Semi-rigid: individual angles, fixed positions. 14 DOF."""
    s0 = 1.0 + 1e-8
    centers = axial_to_cartesian(axial_coords, s0)
    base_positions = [[x, y, 0.0] for x, y in centers]
    cx0 = sum(c[0] for c in centers) / len(centers)
    cy0 = sum(c[1] for c in centers) / len(centers)

    def obj_semi(p):
        angles = p[:N]
        ox, oy, outer_angle = p[N], p[N+1], p[N+2]
        inner = []
        for i in range(N):
            bx, by, _ = base_positions[i]
            inner.append([bx, by, angles[i]])
        return compute_required_side_smooth(inner, (ox, oy), outer_angle, beta=40.0)

    def obj_semi_exact(p):
        angles = p[:N]
        ox, oy, outer_angle = p[N], p[N+1], p[N+2]
        inner = []
        for i in range(N):
            bx, by, _ = base_positions[i]
            inner.append([bx, by, angles[i]])
        return compute_required_side_fast(inner, (ox, oy), outer_angle)

    best_val = float('inf')
    best_p = None

    for outer_a in range(0, 60, 15):
        if time.time() > deadline - 12:
            break
        p0 = np.zeros(N + 3)
        p0[N] = cx0
        p0[N+1] = cy0
        p0[N+2] = float(outer_a)
        np.random.seed(hash(outer_a) % 10000)
        p0[:N] = np.random.uniform(-15, 15, N)

        try:
            r1 = minimize(obj_semi, p0, method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8, 'maxfev': 500})
            r2 = minimize(obj_semi_exact, r1.x, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 800})
            if r2.fun < best_val:
                best_val = r2.fun
                best_p = r2.x.copy()
        except Exception:
            pass

    if best_p is None:
        inner = [[x, y, 0.0] for x, y in centers]
        return inner, (cx0, cy0), 0.0, compute_required_side_fast(inner, (cx0, cy0), 0.0)

    angles = best_p[:N]
    ox, oy, outer_angle = float(best_p[N]), float(best_p[N+1]), float(best_p[N+2])
    inner = []
    for i in range(N):
        bx, by, _ = base_positions[i]
        inner.append([bx, by, float(angles[i])])

    return inner, (ox, oy), outer_angle, best_val

def full_joint_optimize(init_inner, init_oc, init_oa, deadline):
    """Full 36-DOF optimization with SA + DE."""
    def obj_joint_smooth(params, beta=50.0):
        il = []
        for i in range(N):
            il.append([params[i*3], params[i*3+1], params[i*3+2]])
        ocx, ocy, oa = params[N*3], params[N*3+1], params[N*3+2]
        side = compute_required_side_smooth(il, (ocx, ocy), oa, beta=beta)
        pen = overlap_penalty(il)
        return side + 500.0 * pen

    def obj_joint_exact(params):
        il = []
        for i in range(N):
            il.append([params[i*3], params[i*3+1], params[i*3+2]])
        ocx, ocy, oa = params[N*3], params[N*3+1], params[N*3+2]
        side = compute_required_side_fast(il, (ocx, ocy), oa)
        pen = overlap_penalty(il)
        return side + 3000.0 * pen

    p0 = np.zeros(N * 3 + 3)
    for i, (x, y, a) in enumerate(init_inner):
        p0[i*3] = x
        p0[i*3+1] = y
        p0[i*3+2] = a
    p0[N*3] = init_oc[0]
    p0[N*3+1] = init_oc[1]
    p0[N*3+2] = init_oa

    current = p0.copy()
    current_val = obj_joint_smooth(current)
    best_params = current.copy()
    best_val_j = current_val

    # Phase 1: Simulated Annealing
    T = 1.0
    T_min = 0.002
    cool = 0.9997
    max_iter = 6000
    np.random.seed(123)
    for it in range(max_iter):
        if time.time() > deadline - 30:
            break
        nb = current.copy()
        r = np.random.random()
        if r < 0.65:
            idx = np.random.randint(0, N)
            dim = np.random.randint(0, 2)
            nb[idx*3+dim] += np.random.normal(0, 0.25)
        elif r < 0.80:
            idx = np.random.randint(0, N)
            nb[idx*3+2] += np.random.normal(0, 10.0)
        elif r < 0.90:
            dim = np.random.randint(0, 2)
            nb[N*3+dim] += np.random.normal(0, 0.15)
        else:
            nb[N*3+2] += np.random.normal(0, 4.0)

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

    # Phase 2: DE
    if time.time() < deadline - 20:
        de_best = best_params.copy()
        de_best_val = best_val_j
        np.random.seed(456)
        pop_size = 20
        population = [best_params.copy()]
        for _ in range(pop_size - 1):
            p = best_params.copy()
            for i in range(N):
                p[i*3] += np.random.normal(0, 0.5)
                p[i*3+1] += np.random.normal(0, 0.5)
                p[i*3+2] += np.random.normal(0, 15.0)
            p[N*3] += np.random.normal(0, 0.3)
            p[N*3+1] += np.random.normal(0, 0.3)
            p[N*3+2] += np.random.normal(0, 5.0)
            population.append(p)

        F = 0.8
        CR = 0.7
        max_gen = 50
        for gen in range(max_gen):
            if time.time() > deadline - 20:
                break
            new_pop = []
            for i in range(pop_size):
                a, b, c = i, i, i
                while a == i:
                    a = np.random.randint(0, pop_size)
                while b == i or b == a:
                    b = np.random.randint(0, pop_size)
                while c == i or c == a or c == b:
                    c = np.random.randint(0, pop_size)
                trial = population[a].copy()
                for j in range(len(trial)):
                    if np.random.random() < CR or j == np.random.randint(0, len(trial)):
                        trial[j] = population[a][j] + F * (population[b][j] - population[c][j])
                trial_val = obj_joint_smooth(trial)
                if trial_val <= obj_joint_smooth(population[i]):
                    new_pop.append(trial)
                else:
                    new_pop.append(population[i])
            population = new_pop
            for p in population:
                v = obj_joint_smooth(p)
                if v < de_best_val:
                    de_best_val = v
                    de_best = p.copy()

        if de_best_val < best_val_j:
            best_params = de_best
            best_val_j = de_best_val

    best_inner_j = []
    for i in range(N):
        best_inner_j.append([float(best_params[i*3]), float(best_params[i*3+1]), float(best_params[i*3+2])])
    best_oc_j = (float(best_params[N*3]), float(best_params[N*3+1]))
    best_oa_j = float(best_params[N*3+2])

    return best_inner_j, best_oc_j, best_oa_j, best_params

def final_polish(params, deadline):
    """High-precision multi-stage polish."""
    def obj_smooth(p, beta=30.0):
        il = []
        for i in range(N):
            il.append([p[i*3], p[i*3+1], p[i*3+2]])
        ocx, ocy, oa = p[N*3], p[N*3+1], p[N*3+2]
        side = compute_required_side_smooth(il, (ocx, ocy), oa, beta=beta)
        pen = overlap_penalty(il)
        return side + 2000.0 * pen

    def obj_exact(p):
        il = []
        for i in range(N):
            il.append([p[i*3], p[i*3+1], p[i*3+2]])
        ocx, ocy, oa = p[N*3], p[N*3+1], p[N*3+2]
        side = compute_required_side_fast(il, (ocx, ocy), oa)
        pen = overlap_penalty(il)
        return side + 5000.0 * pen

    current = params.copy()

    try:
        r1 = minimize(obj_smooth, current, method='Nelder-Mead',
                      options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-11, 'maxfev': 3000})
        current = r1.x
    except Exception:
        pass

    if time.time() < deadline - 5:
        try:
            r2 = minimize(obj_exact, current, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-11, 'fatol': 1e-13, 'maxfev': 3000})
            current = r2.x
        except Exception:
            pass

    if time.time() < deadline - 2:
        cur_val = obj_exact(current)
        step = 0.05
        for _ in range(50):
            if time.time() > deadline - 2:
                break
            improved = False
            for j in range(len(current)):
                for sign in [1, -1]:
                    trial = current.copy()
                    trial[j] += sign * step
                    trial_val = obj_exact(trial)
                    if trial_val < cur_val - 1e-12:
                        current = trial
                        cur_val = trial_val
                        improved = True
                        break
            if not improved:
                step *= 0.5
                if step < 1e-8:
                    break

    return current

def lp_polish(inner, outer_center, outer_angle_deg, deadline):
    """Use LP solver to find optimal center for given inner hexagon positions and outer angle."""
    all_pts = []
    for (ix, iy, iang) in inner:
        verts = hex_vertices(ix, iy, 1.0, math.radians(iang))
        all_pts.append(verts)
    pts = np.vstack(all_pts)

    # Try multiple outer angles around the given one
    best_side = float('inf')
    best_oc = outer_center
    best_oa = outer_angle_deg

    for da in range(-15, 16, 3):
        oa_try = outer_angle_deg + da
        S, oc, _ = compute_required_side_lp(pts, oa_try)
        if S < best_side:
            best_side = S
            best_oc = oc
            best_oa = oa_try

    # Fine-tune the best angle
    for da in np.linspace(-3, 3, 13):
        oa_try = best_oa + da
        S, oc, _ = compute_required_side_lp(pts, oa_try)
        if S < best_side:
            best_side = S
            best_oc = oc
            best_oa = oa_try

    return best_side, best_oc, best_oa

def construct():
    t0 = time.time()
    deadline = t0 + TIME_LIMIT - SAFETY

    # Diverse lattice configurations
    axial1 = [(0,0), (1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1), (2,-1),(1,1),(-2,1),(-1,-1)]
    axial2 = [(-1,-1),(0,-1),(1,-1),(2,-1), (-1,0),(0,0),(1,0), (-1,1),(0,1),(1,1),(2,1)]
    axial3 = [(-1,-1),(0,-1),(1,-1), (-2,0),(-1,0),(0,0),(1,0),(2,0), (-1,1),(0,1),(1,1)]
    axial4 = [(-2,-1),(-1,-1),(0,-1),(1,-1),(2,-1), (-1,0),(0,0),(1,0), (-1,1),(0,1),(1,1)]
    axial5 = [(0,0), (1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1), (2,0),(0,2),(-2,1),(-2,-1)]
    axial6 = [(0,0),(1,0),(2,0), (0,1),(1,1),(2,1), (0,2),(1,2),(2,2), (-1,1),(3,1)]
    axial7 = [(0,0),(1,0),(2,0),(3,0), (0,1),(1,1),(2,1),(3,1), (0,2),(1,2),(2,2)]
    axial8 = [(0,0),(1,0),(2,0), (0,1),(1,1),(2,1), (0,2),(1,2),(2,2), (1,3),(2,3)]
    axial9 = [(0,0),(1,0),(2,0),(3,0),(4,0), (0,1),(1,1),(2,1),(3,1),(4,1),(5,1)]
    axial10 = [(0,0), (1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1), (2,1),(1,2),(-1,2),(0,3)]

    configs = [axial1, axial2, axial3, axial4, axial5, axial6, axial7, axial8, axial9, axial10]

    best_side = float('inf')
    best_inner = None
    best_outer_center = None
    best_outer_angle = 0.0
    best_params = None

    # Phase 1: Rigid composite for each config
    for ci, axial in enumerate(configs):
        if time.time() > deadline - 80:
            break
        inner, oc, oa, side_val = optimize_composite_rigid(axial, deadline)
        if check_disjoint(inner, eps=1e-5) and side_val < best_side:
            best_side = side_val
            best_inner = inner
            best_outer_center = oc
            best_outer_angle = oa
            p0 = np.zeros(N * 3 + 3)
            for i, (x, y, a) in enumerate(inner):
                p0[i*3] = x; p0[i*3+1] = y; p0[i*3+2] = a
            p0[N*3] = oc[0]; p0[N*3+1] = oc[1]; p0[N*3+2] = oa
            best_params = p0.copy()

    # Phase 1b: Semi-rigid for top configs
    for ci, axial in enumerate(configs[:4]):
        if time.time() > deadline - 55:
            break
        inner, oc, oa, side_val = semi_rigid_optimize(axial, deadline)
        if check_disjoint(inner, eps=1e-5) and side_val < best_side:
            best_side = side_val
            best_inner = inner
            best_outer_center = oc
            best_outer_angle = oa
            p0 = np.zeros(N * 3 + 3)
            for i, (x, y, a) in enumerate(inner):
                p0[i*3] = x; p0[i*3+1] = y; p0[i*3+2] = a
            p0[N*3] = oc[0]; p0[N*3+1] = oc[1]; p0[N*3+2] = oa
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
            p0[i*3] = x; p0[i*3+1] = y; p0[i*3+2] = a
        p0[N*3] = cx0; p0[N*3+1] = cy0; p0[N*3+2] = 0.0
        best_params = p0.copy()

    # Phase 2: LP polish on current best
    if time.time() < deadline - 25:
        lp_side, lp_oc, lp_oa = lp_polish(best_inner, best_outer_center, best_outer_angle, deadline)
        if lp_side < best_side:
            best_side = lp_side
            best_outer_center = lp_oc
            best_outer_angle = lp_oa
            p0 = np.zeros(N * 3 + 3)
            for i, (x, y, a) in enumerate(best_inner):
                p0[i*3] = x; p0[i*3+1] = y; p0[i*3+2] = a
            p0[N*3] = lp_oc[0]; p0[N*3+1] = lp_oc[1]; p0[N*3+2] = lp_oa
            best_params = p0.copy()

    # Phase 3: Full joint optimization
    if time.time() < deadline - 30:
        joint_inner, joint_oc, joint_oa, joint_params = full_joint_optimize(
            best_inner, best_outer_center, best_outer_angle, deadline)
        if check_disjoint(joint_inner, eps=1e-5):
            joint_side = compute_required_side_fast(joint_inner, joint_oc, joint_oa)
            if joint_side < best_side:
                best_inner = joint_inner
                best_outer_center = joint_oc
                best_outer_angle = joint_oa
                best_side = joint_side
                best_params = joint_params

    # Phase 3b: LP polish on joint result
    if time.time() < deadline - 15:
        lp_side, lp_oc, lp_oa = lp_polish(best_inner, best_outer_center, best_outer_angle, deadline)
        if lp_side < best_side:
            best_side = lp_side
            best_outer_center = lp_oc
            best_outer_angle = lp_oa
            p0 = np.zeros(N * 3 + 3)
            for i, (x, y, a) in enumerate(best_inner):
                p0[i*3] = x; p0[i*3+1] = y; p0[i*3+2] = a
            p0[N*3] = lp_oc[0]; p0[N*3+1] = lp_oc[1]; p0[N*3+2] = lp_oa
            best_params = p0.copy()

    # Phase 4: Final high-precision polish
    if time.time() < deadline - 8 and best_params is not None:
        polished = final_polish(best_params, deadline)
        polish_inner = []
        for i in range(N):
            polish_inner.append([float(polished[i*3]), float(polished[i*3+1]), float(polished[i*3+2])])
        polish_oc = (float(polished[N*3]), float(polished[N*3+1]))
        polish_oa = float(polished[N*3+2])
        if check_disjoint(polish_inner, eps=1e-5):
            polish_side = compute_required_side_fast(polish_inner, polish_oc, polish_oa)
            if polish_side < best_side:
                best_inner = polish_inner
                best_outer_center = polish_oc
                best_outer_angle = polish_oa
                best_side = polish_side

    # Phase 5: Final LP polish
    if time.time() < deadline - 3:
        lp_side, lp_oc, lp_oa = lp_polish(best_inner, best_outer_center, best_outer_angle, deadline)
        if lp_side < best_side:
            best_side = lp_side
            best_outer_center = lp_oc
            best_outer_angle = lp_oa

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
