"""Combined: support-function objective with proper containment + multi-phase joint optimization."""
import math
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

def hex_verts(cx, cy, side, angle_rad):
    """6 vertices of a regular hexagon."""
    thetas = angle_rad + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + side * np.cos(thetas), cy + side * np.sin(thetas)])

def container_normals(outer_angle_deg):
    """6 outward normals of the container hexagon (rotated by outer_angle)."""
    base_angles = np.radians(30 + np.arange(6) * 60)
    a = math.radians(outer_angle_deg)
    cs, sn = math.cos(a), math.sin(a)
    normals = np.zeros((6, 2))
    for i in range(6):
        t = base_angles[i]
        nx, ny = math.cos(t), math.sin(t)
        normals[i, 0] = nx * cs - ny * sn
        normals[i, 1] = nx * sn + ny * cs
    return normals

def support_function(inner, outer_center, outer_angle_deg):
    """
    Compute minimum outer side length to contain all inner hexagons.
    The container hexagon has flat-top orientation with normals at 30,90,150,210,270,330 degrees.
    For containment: for each normal direction, the max projection of inner vertices
    (relative to container center) must be <= the apothem of the container.
    Apothem = side * sqrt(3)/2, so side = 2*max_proj/sqrt(3).
    We need to check all 6 normals (which cover both sides due to symmetry).
    """
    if not inner:
        return 0.0
    ox, oy = outer_center
    normals = container_normals(outer_angle_deg)
    
    max_proj = 0.0
    for (ix, iy, iang) in inner:
        verts = hex_verts(ix, iy, 1.0, math.radians(iang))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            for (nx, ny) in normals:
                p = dx * nx + dy * ny
                if p > max_proj:
                    max_proj = p
    return 2.0 * max_proj / SQRT3

def support_function_smooth(inner, outer_center, outer_angle_deg, beta=80.0):
    """Smooth log-sum-exp surrogate for support function."""
    if not inner:
        return 0.0
    ox, oy = outer_center
    normals = container_normals(outer_angle_deg)
    
    projs = []
    for (ix, iy, iang) in inner:
        verts = hex_verts(ix, iy, 1.0, math.radians(iang))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            for (nx, ny) in normals:
                projs.append(dx * nx + dy * ny)
    
    if not projs:
        return 0.0
    P = np.array(projs)
    m = P.max()
    if m < -1e6:
        return 0.0
    
    # Adaptive beta
    spread = m - P.mean() if len(P) > 1 else 0.0
    if spread < 0.01:
        eff_beta = min(beta * 3.0, 500.0)
    elif spread < 0.05:
        eff_beta = min(beta * 2.0, 400.0)
    else:
        eff_beta = beta
    
    if m * eff_beta > 400:
        eff_beta = 400.0 / max(m, 1e-10)
    
    smooth_max = m + math.log(np.sum(np.exp(eff_beta * (P - m)))) / eff_beta
    if smooth_max < 0:
        smooth_max = 0.0
    return 2.0 * smooth_max / SQRT3

def sat_overlap(v1, v2, eps=1e-9):
    """SAT overlap for two convex polygons (numpy arrays shape (6,2))."""
    axes = []
    for poly in (v1, v2):
        for i in range(len(poly)):
            p1 = poly[i]; p2 = poly[(i + 1) % len(poly)]
            e = p2 - p1
            n = np.array([-e[1], e[0]])
            ln = np.linalg.norm(n)
            if ln > 1e-12:
                axes.append(n / ln)
    for ax in axes:
        min1 = np.min(v1 @ ax)
        max1 = np.max(v1 @ ax)
        min2 = np.min(v2 @ ax)
        max2 = np.max(v2 @ ax)
        if max1 < min2 + eps or max2 < min1 + eps:
            return False
    return True

def check_disjoint(inner, eps=1e-6):
    """Pairwise SAT disjointness check."""
    n = len(inner)
    polys = [hex_verts(x, y, 1.0, math.radians(a)) for (x, y, a) in inner]
    for i in range(n):
        for j in range(i + 1, n):
            if sat_overlap(polys[i], polys[j], eps):
                return False
    return True

def overlap_penalty(inner, min_dist=SQRT3 - 0.002):
    """Quadratic penalty for center-distance violations + soft repulsion."""
    pen = 0.0
    n = len(inner)
    for i in range(n):
        xi, yi, _ = inner[i]
        for j in range(i + 1, n):
            xj, yj, _ = inner[j]
            d = math.hypot(xi - xj, yi - yj)
            if d < min_dist:
                pen += (min_dist - d) ** 2
            elif d < SQRT3 * 1.4:
                pen += 0.003 * (SQRT3 * 1.4 - d) ** 2
    return pen

def build_axial(axial_coords, s=1.0 + 1e-8):
    """Build inner hexagons from axial coords with spacing s."""
    centers = [(1.5 * s * q, SQRT3 * s * (r + q / 2.0)) for q, r in axial_coords]
    return [[x, y, 0.0] for x, y in centers]

def centroid(inner):
    """Centroid of inner hexagon centers."""
    cx = sum(p[0] for p in inner) / len(inner)
    cy = sum(p[1] for p in inner) / len(inner)
    return cx, cy

def objective_joint_smooth(params, beta=80.0):
    """Joint objective: support function + overlap penalty (smooth)."""
    il = []
    for i in range(N):
        il.append([params[i * 3], params[i * 3 + 1], params[i * 3 + 2]])
    ocx, ocy, oa = params[N * 3], params[N * 3 + 1], params[N * 3 + 2]
    side = support_function_smooth(il, (ocx, ocy), oa, beta=beta)
    pen = overlap_penalty(il)
    return side + 5000.0 * pen

def objective_joint_exact(params):
    """Joint objective: support function + overlap penalty (exact)."""
    il = []
    for i in range(N):
        il.append([params[i * 3], params[i * 3 + 1], params[i * 3 + 2]])
    ocx, ocy, oa = params[N * 3], params[N * 3 + 1], params[N * 3 + 2]
    side = support_function(il, (ocx, ocy), oa)
    pen = overlap_penalty(il)
    return side + 20000.0 * pen

def objective_joint_lbfgs(params, beta=120.0):
    """Higher-beta smooth objective for L-BFGS-B."""
    il = []
    for i in range(N):
        il.append([params[i * 3], params[i * 3 + 1], params[i * 3 + 2]])
    ocx, ocy, oa = params[N * 3], params[N * 3 + 1], params[N * 3 + 2]
    side = support_function_smooth(il, (ocx, ocy), oa, beta=beta)
    pen = overlap_penalty(il)
    return side + 8000.0 * pen

def optimize_rigid_composite(axial_coords, deadline):
    """
    Rigid composite: all hexagons move together as one body.
    6 DOF: composite angle, dx, dy, outer cx, cy, outer angle.
    """
    s0 = 1.0 + 1e-8
    centers = [(1.5 * s0 * q, SQRT3 * s0 * (r + q / 2.0)) for q, r in axial_coords]
    local_pos = [[x, y, 0.0] for x, y in centers]
    cx0, cy0 = centroid(local_pos)
    
    def obj_rigid(p):
        comp_angle = p[0]
        dx, dy = p[1], p[2]
        ox, oy = p[3], p[4]
        outer_angle = p[5]
        
        ca = math.cos(math.radians(comp_angle))
        sa = math.sin(math.radians(comp_angle))
        inner = []
        for i in range(N):
            bx, by, ba = local_pos[i]
            rx = bx - cx0
            ry = by - cy0
            nx = cx0 + ca * rx - sa * ry + dx
            ny = cy0 + sa * rx + ca * ry + dy
            na = ba + comp_angle
            inner.append([nx, ny, na])
        
        return support_function_smooth(inner, (ox, oy), outer_angle, beta=60.0)
    
    def obj_rigid_exact(p):
        comp_angle = p[0]
        dx, dy = p[1], p[2]
        ox, oy = p[3], p[4]
        outer_angle = p[5]
        
        ca = math.cos(math.radians(comp_angle))
        sa = math.sin(math.radians(comp_angle))
        inner = []
        for i in range(N):
            bx, by, ba = local_pos[i]
            rx = bx - cx0
            ry = by - cy0
            nx = cx0 + ca * rx - sa * ry + dx
            ny = cy0 + sa * rx + ca * ry + dy
            na = ba + comp_angle
            inner.append([nx, ny, na])
        
        return support_function(inner, (ox, oy), outer_angle)
    
    best_val = float('inf')
    best_p = None
    
    # Multi-start over composite and outer angles
    for comp_a in range(0, 60, 5):
        for outer_a in range(0, 60, 10):
            if time.time() > deadline - 30:
                break
            p0 = np.array([float(comp_a), 0.0, 0.0, cx0, cy0, float(outer_a)])
            try:
                r1 = minimize(obj_rigid, p0, method='Nelder-Mead',
                              options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-9, 'maxfev': 400})
                r2 = minimize(obj_rigid_exact, r1.x, method='Nelder-Mead',
                              options={'maxiter': 400, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 600})
                if r2.fun < best_val:
                    best_val = r2.fun
                    best_p = r2.x.copy()
            except Exception:
                pass
    
    if best_p is None:
        inner = build_axial(axial_coords)
        cx0, cy0 = centroid(inner)
        return inner, (cx0, cy0), 0.0, support_function(inner, (cx0, cy0), 0.0)
    
    comp_angle = float(best_p[0])
    dx, dy = float(best_p[1]), float(best_p[2])
    ox, oy = float(best_p[3]), float(best_p[4])
    outer_angle = float(best_p[5])
    
    ca = math.cos(math.radians(comp_angle))
    sa = math.sin(math.radians(comp_angle))
    inner = []
    for i in range(N):
        bx, by, ba = local_pos[i]
        rx = bx - cx0
        ry = by - cy0
        nx = cx0 + ca * rx - sa * ry + dx
        ny = cy0 + sa * rx + ca * ry + dy
        na = ba + comp_angle
        inner.append([nx, ny, na])
    
    return inner, (ox, oy), outer_angle, best_val

def full_joint_optimize(init_inner, init_oc, init_oa, deadline):
    """Full joint optimization: SA + DE + polish."""
    p0 = np.zeros(N * 3 + 3)
    for i, (x, y, a) in enumerate(init_inner):
        p0[i * 3] = x
        p0[i * 3 + 1] = y
        p0[i * 3 + 2] = a
    p0[N * 3] = init_oc[0]
    p0[N * 3 + 1] = init_oc[1]
    p0[N * 3 + 2] = init_oa

    current = p0.copy()
    current_val = objective_joint_smooth(current)
    best_params = current.copy()
    best_val_j = current_val

    # Phase 1: Simulated Annealing
    T = 0.8
    T_min = 0.0005
    cool = 0.9995
    max_iter = 6000
    np.random.seed(42)
    for it in range(max_iter):
        if time.time() > deadline - 45:
            break
        nb = current.copy()
        r = np.random.random()
        scale = max(0.05, T)
        if r < 0.55:
            idx = np.random.randint(0, N)
            dim = np.random.randint(0, 2)
            nb[idx * 3 + dim] += np.random.normal(0, 0.25 * scale)
        elif r < 0.75:
            idx = np.random.randint(0, N)
            nb[idx * 3 + 2] += np.random.normal(0, 10.0 * scale)
        elif r < 0.88:
            dim = np.random.randint(0, 2)
            nb[N * 3 + dim] += np.random.normal(0, 0.15 * scale)
        else:
            nb[N * 3 + 2] += np.random.normal(0, 4.0 * scale)

        nb_val = objective_joint_smooth(nb)
        if nb_val < current_val or np.random.random() < math.exp(-(nb_val - current_val) / max(T, 1e-10)):
            current = nb
            current_val = nb_val
            if current_val < best_val_j:
                best_params = current.copy()
                best_val_j = current_val
        T *= cool
        if T < T_min:
            break

    # Phase 2: Differential Evolution
    if time.time() < deadline - 30:
        de_best = best_params.copy()
        de_best_val = best_val_j
        
        np.random.seed(789)
        pop_size = 12
        population = [best_params.copy()]
        for _ in range(pop_size - 1):
            p = best_params.copy()
            for i in range(N):
                p[i * 3] += np.random.normal(0, 0.5)
                p[i * 3 + 1] += np.random.normal(0, 0.5)
                p[i * 3 + 2] += np.random.normal(0, 15.0)
            p[N * 3] += np.random.normal(0, 0.3)
            p[N * 3 + 1] += np.random.normal(0, 0.3)
            p[N * 3 + 2] += np.random.normal(0, 5.0)
            population.append(p)
        
        pop_vals = [objective_joint_smooth(p) for p in population]
        
        F = 0.6
        CR = 0.7
        max_gen = 30
        for gen in range(max_gen):
            if time.time() > deadline - 30:
                break
            new_pop = population[:]
            new_vals = pop_vals[:]
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
                
                trial_val = objective_joint_smooth(trial)
                if trial_val <= new_vals[i]:
                    new_pop[i] = trial
                    new_vals[i] = trial_val
            
            population = new_pop
            pop_vals = new_vals
            for v, p in zip(pop_vals, population):
                if v < de_best_val:
                    de_best_val = v
                    de_best = p.copy()
        
        if de_best_val < best_val_j:
            best_params = de_best
            best_val_j = de_best_val

    # Phase 3: Polish
    current = best_params.copy()
    
    # Smooth NM
    if time.time() < deadline - 15:
        try:
            r1 = minimize(objective_joint_smooth, current, method='Nelder-Mead',
                          options={'maxiter': 1500, 'xatol': 1e-9, 'fatol': 1e-11, 'maxfev': 2500})
            current = r1.x
        except Exception:
            pass
    
    # L-BFGS-B
    if time.time() < deadline - 10:
        try:
            r2 = minimize(objective_joint_lbfgs, current, method='L-BFGS-B',
                          options={'maxiter': 400, 'ftol': 1e-13, 'gtol': 1e-9, 'maxfun': 1500})
            if r2.fun < objective_joint_lbfgs(current):
                current = r2.x
        except Exception:
            pass
    
    # Exact NM
    if time.time() < deadline - 6:
        try:
            r3 = minimize(objective_joint_exact, current, method='Nelder-Mead',
                          options={'maxiter': 1500, 'xatol': 1e-11, 'fatol': 1e-13, 'maxfev': 2500})
            current = r3.x
        except Exception:
            pass
    
    # Fine coordinate descent
    if time.time() < deadline - 3:
        cur_val = objective_joint_exact(current)
        step = 0.004
        for _ in range(50):
            if time.time() > deadline - 3:
                break
            improved = False
            for j in range(len(current)):
                for sign in [1, -1]:
                    trial = current.copy()
                    trial[j] += sign * step
                    trial_val = objective_joint_exact(trial)
                    if trial_val < cur_val - 1e-13:
                        current = trial
                        cur_val = trial_val
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                step *= 0.5
                if step < 1e-11:
                    break
    
    best_inner_j = []
    for i in range(N):
        best_inner_j.append([float(current[i * 3]), float(current[i * 3 + 1]), float(current[i * 3 + 2])])
    best_oc_j = (float(current[N * 3]), float(current[N * 3 + 1]))
    best_oa_j = float(current[N * 3 + 2])

    return best_inner_j, best_oc_j, best_oa_j, current

def construct():
    t0 = time.time()
    deadline = t0 + 145.0

    # Multiple candidate configurations
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
    
    axial6 = [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1),
              (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)]
    
    axial7 = [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1),
              (2, -1), (2, 0), (1, 1), (0, 2), (-1, 2)]
    
    axial8 = [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1),
              (1, -1), (2, -1), (2, 0), (0, 2), (-2, 1)]
    
    axial9 = [(-1, -1), (0, -1), (1, -1),
              (-1, 0), (0, 0), (1, 0),
              (-1, 1), (0, 1), (1, 1),
              (2, 0), (-2, 0)]
    
    axial10 = [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1),
               (1, -1), (2, 0), (0, 2), (-2, 0), (2, -1)]
    
    # Additional diverse topologies
    axial11 = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1),
               (-1, 1), (1, -1), (-1, -1), (2, 0), (-2, 0)]
    
    axial12 = [(0, 0), (1, 0), (2, 0), (3, 0),
               (-1, 1), (0, 1), (1, 1), (2, 1),
               (-1, -1), (0, -1), (1, -1)]
    
    configs = [axial1, axial2, axial3, axial4, axial5, axial6, axial7, axial8, axial9, axial10, axial11, axial12]
    
    best_side = float('inf')
    best_inner = None
    best_outer_center = None
    best_outer_angle = 0.0
    best_params = None
    
    # Phase 1: Rigid composite optimization for each config
    for ci, axial in enumerate(configs):
        if time.time() > deadline - 60:
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
        best_inner = build_axial(axial)
        cx0, cy0 = centroid(best_inner)
        best_outer_center = (cx0, cy0)
        best_outer_angle = 0.0
        best_side = support_function(best_inner, best_outer_center, best_outer_angle)
        p0 = np.zeros(N * 3 + 3)
        for i, (x, y, a) in enumerate(best_inner):
            p0[i * 3] = x
            p0[i * 3 + 1] = y
            p0[i * 3 + 2] = a
        p0[N * 3] = cx0
        p0[N * 3 + 1] = cy0
        p0[N * 3 + 2] = 0.0
        best_params = p0.copy()
    
    # Phase 2: Full joint optimization from best config
    if time.time() < deadline - 50:
        joint_inner, joint_oc, joint_oa, joint_params = full_joint_optimize(
            best_inner, best_outer_center, best_outer_angle, deadline)
        
        if check_disjoint(joint_inner, eps=1e-5):
            joint_side = support_function(joint_inner, joint_oc, joint_oa)
            if joint_side < best_side:
                best_inner = joint_inner
                best_outer_center = joint_oc
                best_outer_angle = joint_oa
                best_side = joint_side
                best_params = joint_params
    
    # Phase 2b: Try joint optimization from other good configs
    if best_params is not None and time.time() < deadline - 35:
        # Try from second-best rigid config
        for ci, axial in enumerate(configs):
            if time.time() > deadline - 35:
                break
            if ci == 0:
                continue
            inner2, oc2, oa2, side2 = optimize_rigid_composite(axial, deadline)
            if check_disjoint(inner2, eps=1e-5) and side2 < best_side * 1.02:
                j2_inner, j2_oc, j2_oa, j2_params = full_joint_optimize(
                    inner2, oc2, oa2, deadline)
                if check_disjoint(j2_inner, eps=1e-5):
                    j2_side = support_function(j2_inner, j2_oc, j2_oa)
                    if j2_side < best_side:
                        best_inner = j2_inner
                        best_outer_center = j2_oc
                        best_outer_angle = j2_oa
                        best_side = j2_side
                        best_params = j2_params
                break
    
    # Phase 2c: Perturbed restart from best
    if best_params is not None and time.time() < deadline - 25:
        np.random.seed(1234)
        perturbed = best_params.copy()
        for i in range(N):
            perturbed[i * 3] += np.random.normal(0, 0.2)
            perturbed[i * 3 + 1] += np.random.normal(0, 0.2)
            perturbed[i * 3 + 2] += np.random.normal(0, 6.0)
        perturbed[N * 3] += np.random.normal(0, 0.15)
        perturbed[N * 3 + 1] += np.random.normal(0, 0.15)
        perturbed[N * 3 + 2] += np.random.normal(0, 2.0)
        
        pert_inner = []
        for i in range(N):
            pert_inner.append([float(perturbed[i * 3]), float(perturbed[i * 3 + 1]), float(perturbed[i * 3 + 2])])
        pert_oc = (float(perturbed[N * 3]), float(perturbed[N * 3 + 1]))
        pert_oa = float(perturbed[N * 3 + 2])
        
        if check_disjoint(pert_inner, eps=1e-4):
            pert_side = support_function(pert_inner, pert_oc, pert_oa)
            if pert_side < best_side * 1.05:
                j3_inner, j3_oc, j3_oa, j3_params = full_joint_optimize(
                    pert_inner, pert_oc, pert_oa, deadline)
                if check_disjoint(j3_inner, eps=1e-5):
                    j3_side = support_function(j3_inner, j3_oc, j3_oa)
                    if j3_side < best_side:
                        best_inner = j3_inner
                        best_outer_center = j3_oc
                        best_outer_angle = j3_oa
                        best_side = j3_side
                        best_params = j3_params
    
    # Phase 3: Final high-precision polish
    if time.time() < deadline - 15 and best_params is not None:
        current = best_params.copy()
        
        # Smooth NM
        if time.time() < deadline - 12:
            try:
                r1 = minimize(objective_joint_smooth, current, method='Nelder-Mead',
                              options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 3000})
                current = r1.x
            except Exception:
                pass
        
        # L-BFGS-B
        if time.time() < deadline - 8:
            try:
                r2 = minimize(objective_joint_lbfgs, current, method='L-BFGS-B',
                              options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 2000})
                if r2.fun < objective_joint_lbfgs(current):
                    current = r2.x
            except Exception:
                pass
        
        # Exact NM
        if time.time() < deadline - 5:
            try:
                r3 = minimize(objective_joint_exact, current, method='Nelder-Mead',
                              options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': 3000})
                current = r3.x
            except Exception:
                pass
        
        # Fine coordinate descent
        if time.time() < deadline - 3:
            cur_val = objective_joint_exact(current)
            step = 0.003
            for _ in range(80):
                if time.time() > deadline - 3:
                    break
                improved = False
                for j in range(len(current)):
                    for sign in [1, -1]:
                        trial = current.copy()
                        trial[j] += sign * step
                        trial_val = objective_joint_exact(trial)
                        if trial_val < cur_val - 1e-13:
                            current = trial
                            cur_val = trial_val
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    step *= 0.5
                    if step < 1e-11:
                        break
        
        # Very fine outer parameter search
        if time.time() < deadline - 2:
            cur_val = objective_joint_exact(current)
            fine_step = 0.002
            for _ in range(40):
                if time.time() > deadline - 2:
                    break
                improved = False
                for j in [N * 3, N * 3 + 1, N * 3 + 2]:
                    for sign in [1, -1]:
                        trial = current.copy()
                        trial[j] += sign * fine_step
                        trial_val = objective_joint_exact(trial)
                        if trial_val < cur_val - 1e-14:
                            current = trial
                            cur_val = trial_val
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    fine_step *= 0.5
                    if fine_step < 1e-12:
                        break
        
        polish_inner = []
        for i in range(N):
            polish_inner.append([float(current[i * 3]), float(current[i * 3 + 1]), float(current[i * 3 + 2])])
        polish_oc = (float(current[N * 3]), float(current[N * 3 + 1]))
        polish_oa = float(current[N * 3 + 2])
        
        if check_disjoint(polish_inner, eps=1e-5):
            polish_side = support_function(polish_inner, polish_oc, polish_oa)
            if polish_side < best_side:
                best_inner = polish_inner
                best_outer_center = polish_oc
                best_outer_angle = polish_oa
                best_side = polish_side
    
    # Phase 4: Bisection-guided refinement of outer angle
    if time.time() < deadline - 5:
        cur_inner = best_inner
        cur_oc = best_outer_center
        cur_oa = best_outer_angle
        cur_side = best_side
        
        for da in [1, -1, 2, -2, 0.5, -0.5, 0.25, -0.25]:
            if time.time() > deadline - 3:
                break
            new_oa = cur_oa + da
            new_side = support_function(cur_inner, cur_oc, new_oa)
            if new_side < cur_side - 1e-9:
                cur_oa = new_oa
                cur_side = new_side
        
        for dx in [0.01, -0.01, 0.02, -0.02, 0.005, -0.005]:
            for dy in [0.01, -0.01, 0.02, -0.02, 0.005, -0.005]:
                if time.time() > deadline - 3:
                    break
                new_oc = (cur_oc[0] + dx, cur_oc[1] + dy)
                new_side = support_function(cur_inner, new_oc, cur_oa)
                if new_side < cur_side - 1e-9:
                    cur_oc = new_oc
                    cur_side = new_side
        
        if cur_side < best_side:
            best_outer_center = cur_oc
            best_outer_angle = cur_oa
            best_side = cur_side
    
    # Final verification
    if not check_disjoint(best_inner, eps=1e-5):
        best_inner = build_axial(configs[0])
        cx0, cy0 = centroid(best_inner)
        best_outer_center = (cx0, cy0)
        best_outer_angle = 0.0
        best_side = support_function(best_inner, best_outer_center, best_outer_angle)
    
    return best_inner, best_outer_center, best_side, best_outer_angle

# EVOLVE-BLOCK-END
