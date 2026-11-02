"""Combined hexagon packing: rigid-body topology search + smooth-max support + L-BFGS-B polish."""
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

def hex_vertices_fast(cx, cy, angle_deg):
    """Fast vertex computation for unit hexagons using precomputed cos/sin."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    c = np.array([1.0, 0.5, -0.5, -1.0, -0.5, 0.5])
    s = np.array([0.0, SQRT3/2, SQRT3/2, 0.0, -SQRT3/2, -SQRT3/2])
    x = cx + ca * c - sa * s
    y = cy + sa * c + ca * s
    return np.column_stack([x, y])

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
    
    all_verts = []
    for (ix, iy, iang) in inner:
        verts = hex_vertices_fast(ix, iy, iang)
        all_verts.append(verts)
    
    if not all_verts:
        return 0.0
    
    V = np.vstack(all_verts)
    local = (V - np.array([ox, oy])) @ R.T
    projs = local @ normals.T
    max_proj = projs.max()
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
    
    all_verts = []
    for (ix, iy, iang) in inner:
        verts = hex_vertices_fast(ix, iy, iang)
        all_verts.append(verts)
    
    if not all_verts:
        return 0.0
    
    V = np.vstack(all_verts)
    local = (V - np.array([ox, oy])) @ R.T
    P = (local @ normals.T).ravel()
    
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

def overlap_penalty(inner, min_dist=SQRT3 - 0.005):
    """
    Quadratic penalty when inner hexagons overlap.
    """
    pen = 0.0
    n = len(inner)
    for i in range(n):
        xi, yi, ai = inner[i]
        for j in range(i + 1, n):
            xj, yj, aj = inner[j]
            dx = xj - xi
            dy = yj - yi
            d = math.hypot(dx, dy)
            angle_diff = abs(ai - aj) % 60.0
            angle_diff = min(angle_diff, 60.0 - angle_diff)
            if angle_diff < 1.0:
                threshold = min_dist
            else:
                threshold = min_dist * (1.0 - 0.02 * angle_diff / 30.0)
            if d < threshold:
                pen += (threshold - d) ** 2
    return pen

def build_lattice_config(axial_coords, s=1.0 + 1e-8):
    """Build inner hexagons from axial coordinates with spacing s."""
    centers = [(1.5 * s * q, SQRT3 * s * (r + q / 2.0)) for q, r in axial_coords]
    return [[x, y, 0.0] for x, y in centers]

def rigid_transform(inner_base, global_angle_deg, dx, dy):
    """Apply rigid rotation about centroid + translation to a base configuration."""
    cx0 = sum(p[0] for p in inner_base) / len(inner_base)
    cy0 = sum(p[1] for p in inner_base) / len(inner_base)
    a = math.radians(global_angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    result = []
    for (bx, by, ba) in inner_base:
        rx = bx - cx0
        ry = by - cy0
        nx = cx0 + ca * rx - sa * ry + dx
        ny = cy0 + sa * rx + ca * ry + dy
        na = ba + global_angle_deg
        result.append([nx, ny, na])
    return result

def rigid_objective_smooth(params, inner_base, beta=60.0):
    """
    Objective for rigid-body optimization.
    params: [global_angle_deg, dx, dy, outer_angle_deg]
    The outer center is fixed at the centroid of the transformed configuration.
    """
    ga, dx, dy, oa = params
    transformed = rigid_transform(inner_base, ga, dx, dy)
    cx = sum(p[0] for p in transformed) / len(transformed)
    cy = sum(p[1] for p in transformed) / len(transformed)
    side = compute_required_side_smooth(transformed, (cx, cy), oa, beta=beta)
    return side

def rigid_objective_exact(params, inner_base):
    """Exact objective for rigid-body optimization."""
    ga, dx, dy, oa = params
    transformed = rigid_transform(inner_base, ga, dx, dy)
    cx = sum(p[0] for p in transformed) / len(transformed)
    cy = sum(p[1] for p in transformed) / len(transformed)
    side = compute_required_side_fast(transformed, (cx, cy), oa)
    return side

def optimize_rigid(inner_base, deadline, n_starts=20):
    """
    Global 1D+ search over rigid body orientation and translation.
    Uses multi-start Nelder-Mead on the 4-DOF rigid parameterization.
    """
    best_val = float('inf')
    best_params = None
    best_inner = None
    best_oc = None
    best_oa = None
    
    cx0 = sum(p[0] for p in inner_base) / len(inner_base)
    cy0 = sum(p[1] for p in inner_base) / len(inner_base)
    
    np.random.seed(42)
    
    for start_i in range(n_starts):
        if time.time() > deadline - 15:
            break
        
        # Random starting point
        p0 = np.array([
            np.random.uniform(-30, 30),  # global angle
            np.random.uniform(-0.5, 0.5),  # dx
            np.random.uniform(-0.5, 0.5),  # dy
            np.random.uniform(-30, 30),  # outer angle
        ])
        
        try:
            # Coarse: Nelder-Mead on smooth
            r1 = minimize(rigid_objective_smooth, p0, args=(inner_base,),
                          method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8, 'maxfev': 300, 'adaptive': True})
            
            if r1.fun < best_val and time.time() < deadline - 10:
                # Fine: Nelder-Mead on exact
                r2 = minimize(rigid_objective_exact, r1.x, args=(inner_base,),
                              method='Nelder-Mead',
                              options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 800, 'adaptive': True})
                if r2.fun < best_val:
                    best_val = r2.fun
                    best_params = r2.x.copy()
                    transformed = rigid_transform(inner_base, best_params[0], best_params[1], best_params[2])
                    cx = sum(p[0] for p in transformed) / len(transformed)
                    cy = sum(p[1] for p in transformed) / len(transformed)
                    best_inner = transformed
                    best_oc = (cx, cy)
                    best_oa = float(best_params[3])
        except Exception:
            pass
    
    if best_inner is not None:
        return best_inner, best_oc, best_oa, best_val
    return None, None, None, float('inf')

def try_orientation_optimization(inner, outer_center, outer_angle_deg, deadline):
    """Optimize outer hexagon orientation and center for fixed inner positions."""
    def obj_exact(p):
        return compute_required_side_fast(inner, (p[0], p[1]), p[2])
    
    best_val = float('inf')
    best_p = None
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

def joint_optimize(init_inner, init_outer_center, init_outer_angle, deadline):
    """Joint optimization of inner positions/angles and outer hexagon."""
    def obj_joint_smooth(params, beta=60.0):
        il = []
        for i in range(N):
            il.append([params[i * 3], params[i * 3 + 1], params[i * 3 + 2]])
        ocx, ocy, oa = params[N * 3], params[N * 3 + 1], params[N * 3 + 2]
        side = compute_required_side_smooth(il, (ocx, ocy), oa, beta=beta)
        pen = overlap_penalty(il)
        return side + 3000.0 * pen

    def obj_joint_exact(params):
        il = []
        for i in range(N):
            il.append([params[i * 3], params[i * 3 + 1], params[i * 3 + 2]])
        ocx, ocy, oa = params[N * 3], params[N * 3 + 1], params[N * 3 + 2]
        side = compute_required_side_fast(il, (ocx, ocy), oa)
        pen = overlap_penalty(il)
        return side + 8000.0 * pen

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
        {'T': 0.6, 'T_min': 0.008, 'cool': 0.9997, 'pos_step': 0.25, 'ang_step': 10.0, 'oc_step': 0.15, 'oa_step': 4.0},
        {'T': 0.25, 'T_min': 0.002, 'cool': 0.9998, 'pos_step': 0.12, 'ang_step': 5.0, 'oc_step': 0.08, 'oa_step': 2.5},
        {'T': 0.08, 'T_min': 0.0005, 'cool': 0.9999, 'pos_step': 0.05, 'ang_step': 2.5, 'oc_step': 0.04, 'oa_step': 1.0},
    ]
    
    for phase in phases:
        if time.time() > deadline - 25:
            break
        T = phase['T']
        T_min = phase['T_min']
        cool = phase['cool']
        max_iter = 6000
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
    """High-precision polish: smooth NM -> exact NM -> L-BFGS-B with very tight tolerances."""
    def obj_smooth(p, beta=80.0):
        il = []
        for i in range(N):
            il.append([p[i * 3], p[i * 3 + 1], p[i * 3 + 2]])
        ocx, ocy, oa = p[N * 3], p[N * 3 + 1], p[N * 3 + 2]
        side = compute_required_side_smooth(il, (ocx, ocy), oa, beta=beta)
        pen = overlap_penalty(il)
        return side + 2000.0 * pen

    def obj_exact(p):
        il = []
        for i in range(N):
            il.append([p[i * 3], p[i * 3 + 1], p[i * 3 + 2]])
        ocx, ocy, oa = p[N * 3], p[N * 3 + 1], p[N * 3 + 2]
        side = compute_required_side_fast(il, (ocx, ocy), oa)
        pen = overlap_penalty(il)
        return side + 8000.0 * pen

    best_p = params.copy()
    best_v = obj_exact(best_p)
    
    if time.time() < deadline - 10:
        try:
            res = minimize(obj_smooth, best_p, method='Nelder-Mead',
                           options={'maxiter': 4000, 'xatol': 1e-11, 'fatol': 1e-13, 'maxfev': 5000, 'adaptive': True})
            if res.fun < best_v:
                best_v = res.fun
                best_p = res.x.copy()
        except Exception:
            pass
    
    if time.time() < deadline - 5:
        try:
            res = minimize(obj_exact, best_p, method='Nelder-Mead',
                           options={'maxiter': 8000, 'xatol': 1e-13, 'fatol': 1e-15, 'maxfev': 10000, 'adaptive': True})
            if res.fun < best_v:
                best_v = res.fun
                best_p = res.x.copy()
        except Exception:
            pass
    
    if time.time() < deadline - 2:
        il_tmp = []
        for i in range(N):
            il_tmp.append([best_p[i * 3], best_p[i * 3 + 1], best_p[i * 3 + 2]])
        bounds = []
        for i in range(N):
            bounds.append((best_p[i * 3] - 0.2, best_p[i * 3] + 0.2))
            bounds.append((best_p[i * 3 + 1] - 0.2, best_p[i * 3 + 1] + 0.2))
            bounds.append((-180.0, 180.0))
        bounds.append((best_p[N * 3] - 0.8, best_p[N * 3] + 0.8))
        bounds.append((best_p[N * 3 + 1] - 0.8, best_p[N * 3 + 1] + 0.8))
        bounds.append((best_p[N * 3 + 2] - 10.0, best_p[N * 3 + 2] + 10.0))
        try:
            res = minimize(obj_exact, best_p, method='L-BFGS-B',
                           bounds=bounds, options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxfun': 15000})
            if res.fun < best_v:
                best_v = res.fun
                best_p = res.x.copy()
        except Exception:
            pass
    
    if time.time() < deadline - 1:
        try:
            res = minimize(obj_exact, best_p, method='Nelder-Mead',
                           options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-16, 'maxfev': 12000, 'adaptive': True})
            if res.fun < best_v:
                best_v = res.fun
                best_p = res.x.copy()
        except Exception:
            pass
    
    return best_p

def construct():
    t0 = time.time()
    deadline = t0 + 148.0  # 2s safety margin

    # --- Define multiple candidate polyhex topologies ---
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
    
    axial8 = [(-1, -1), (0, -1), (1, -1), (2, -1),
              (-2, 0), (-1, 0), (0, 0), (1, 0),
              (0, 1), (1, 1), (2, 1)]
    
    axial9 = [(-2, -1), (-1, -1), (0, -1),
              (-1, 0), (0, 0), (1, 0), (2, 0),
              (0, 1), (1, 1), (2, 1), (3, 1)]
    
    configs = [axial1, axial2, axial3, axial4, axial5, axial6, axial7, axial8, axial9]
    
    best_side = float('inf')
    best_inner = None
    best_outer_center = None
    best_outer_angle = 0.0
    best_params = None
    
    # Phase 1a: Rigid-body optimization for each topology (reduces 36-DOF to 4-DOF)
    for ci, axial in enumerate(configs):
        if time.time() > deadline - 60:
            break
        s0 = 1.0 + 1e-8
        base_inner = build_lattice_config(axial, s0)
        
        r_inner, r_oc, r_oa, r_val = optimize_rigid(base_inner, deadline, n_starts=15)
        if r_inner is not None and r_val < best_side:
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
    
    # Phase 1b: If rigid didn't find anything good, fall back to orientation-only search
    if best_inner is None or best_side > 5.0:
        for ci, axial in enumerate(configs):
            if time.time() > deadline - 55:
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
    if time.time() < deadline - 35:
        joint_inner, joint_oc, joint_oa, joint_params = joint_optimize(
            best_inner, best_outer_center, best_outer_angle, deadline)
        
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
    
    # Phase 2b: Also try joint optimization from perturbed best config if time permits
    if time.time() < deadline - 25 and best_params is not None:
        np.random.seed(789)
        p2 = best_params.copy()
        for i in range(N):
            p2[i*3] += np.random.normal(0, 0.06)
            p2[i*3+1] += np.random.normal(0, 0.06)
            p2[i*3+2] += np.random.normal(0, 6.0)
        p2[N*3] += np.random.normal(0, 0.05)
        p2[N*3+1] += np.random.normal(0, 0.05)
        p2[N*3+2] += np.random.normal(0, 3.0)
        
        init_inner2 = []
        for i in range(N):
            init_inner2.append([float(p2[i*3]), float(p2[i*3+1]), float(p2[i*3+2])])
        init_oc2 = (float(p2[N*3]), float(p2[N*3+1]))
        init_oa2 = float(p2[N*3+2])
        
        joint_inner2, joint_oc2, joint_oa2, joint_params2 = joint_optimize(
            init_inner2, init_oc2, init_oa2, deadline)
        
        if not check_disjoint(joint_inner2, eps=1e-5):
            for repair_iter in range(40):
                if time.time() > deadline - 20:
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
                                push = 0.015
                                joint_inner2[i][0] -= push * dx / d
                                joint_inner2[i][1] -= push * dy / d
                                joint_inner2[j][0] += push * dx / d
                                joint_inner2[j][1] += push * dy / d
                            else:
                                joint_inner2[i][0] -= 0.015
                                joint_inner2[j][0] += 0.015
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
    if time.time() < deadline - 4 and best_params is not None:
        polished = final_polish(best_params, deadline)
        polish_inner = []
        for i in range(N):
            polish_inner.append([float(polished[i * 3]), float(polished[i * 3 + 1]), float(polished[i * 3 + 2])])
        polish_oc = (float(polished[N * 3]), float(polished[N * 3 + 1]))
        polish_oa = float(polished[N * 3 + 2])
        
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
