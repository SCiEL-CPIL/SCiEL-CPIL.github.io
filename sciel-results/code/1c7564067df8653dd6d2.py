"""Improved program: contact-graph based packing with multi-start orientation optimization."""
import math
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

# Precompute hexagon unit vertices (angle 0)
_UNIT_VERTS = np.column_stack([np.cos(np.arange(6) * np.pi / 3.0),
                                np.sin(np.arange(6) * np.pi / 3.0)])
_NORMALS = np.array([[math.cos(math.radians(30 + i * 60)),
                      math.sin(math.radians(30 + i * 60))] for i in range(6)])

def hex_vertices(cx, cy, s, angle_rad):
    """Return list of (x, y) vertices for a regular hexagon."""
    c = math.cos(angle_rad)
    s_r = math.sin(angle_rad)
    # Vertices of unit hexagon rotated by angle_rad, scaled by s, translated by (cx, cy)
    # Unit vertices are at angles 0, 60, 120, 180, 240, 300
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        x = cx + s * math.cos(theta)
        y = cy + s * math.sin(theta)
        verts.append((x, y))
    return verts

def _get_rotated_verts(angle_rad):
    """Return 6 vertices of unit hexagon rotated by angle_rad."""
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    return _UNIT_VERTS @ R.T

def compute_required_side_fast(inner, outer_center, outer_angle_deg):
    """
    Compute minimum outer side length to contain all inner hexagons.
    inner: list of (x, y, angle_degrees)
    """
    if not inner:
        return 0.0
    ox, oy = outer_center
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    max_proj = -np.inf
    for (ix, iy, iang) in inner:
        verts = _get_rotated_verts(math.radians(iang))
        local = (verts + np.array([ix - ox, iy - oy])) @ R.T
        projs = local @ _NORMALS.T
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
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    all_projs = []
    for (ix, iy, iang) in inner:
        verts = _get_rotated_verts(math.radians(iang))
        local = (verts + np.array([ix - ox, iy - oy])) @ R.T
        projs = local @ _NORMALS.T
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

def build_contact_graph_config(axial_coords, s=1.0 + 1e-8, angle_offset=0.0):
    """Build inner hexagons with slight angle perturbation for contact-graph init."""
    centers = [(1.5 * s * q, SQRT3 * s * (r + q / 2.0)) for q, r in axial_coords]
    return [[x, y, angle_offset] for x, y in centers]

def try_orientation_optimization(inner, outer_center, outer_angle_deg, deadline):
    """Optimize outer hexagon orientation and center for fixed inner positions."""
    def obj_smooth(p):
        return compute_required_side_smooth(inner, (p[0], p[1]), p[2], beta=60.0)
    
    def obj_exact(p):
        return compute_required_side_fast(inner, (p[0], p[1]), p[2])
    
    best_val = float('inf')
    best_p = None
    # Try multiple starting angles - finer grid for better coverage
    for ia in range(0, 60, 2):
        if time.time() > deadline - 12:
            break
        p0 = np.array([outer_center[0], outer_center[1], float(ia)])
        try:
            r1 = minimize(obj_smooth, p0, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 400})
            r2 = minimize(obj_exact, r1.x, method='Nelder-Mead',
                          options={'maxiter': 400, 'xatol': 1e-11, 'fatol': 1e-13, 'maxfev': 800})
            if r2.fun < best_val:
                best_val = r2.fun
                best_p = r2.x.copy()
        except Exception:
            pass
    
    # 1D sweep on angle with optimal center
    if time.time() < deadline - 8:
        def angle_obj(a):
            return compute_required_side_fast(inner, (outer_center[0], outer_center[1]), a)
        
        best_1d = float('inf')
        best_1d_angle = outer_angle_deg
        for a in np.arange(0, 60, 0.5):
            v = angle_obj(float(a))
            if v < best_1d:
                best_1d = v
                best_1d_angle = float(a)
        # Refine 1D
        try:
            res = minimize_scalar(angle_obj, bounds=(best_1d_angle - 2, best_1d_angle + 2),
                                  method='bounded', options={'xatol': 1e-10})
            if res.fun < best_1d:
                best_1d = res.fun
                best_1d_angle = float(res.x)
        except Exception:
            pass
        
        # Optimize center for best angle
        def center_obj(c):
            return compute_required_side_fast(inner, (c[0], c[1]), best_1d_angle)
        try:
            rc = minimize(center_obj, np.array(outer_center), method='Nelder-Mead',
                          options={'maxiter': 600, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 1200})
            if rc.fun < best_val:
                best_val = rc.fun
                best_p = np.array([rc.x[0], rc.x[1], best_1d_angle])
        except Exception:
            pass
    
    if best_p is not None:
        return (float(best_p[0]), float(best_p[1])), float(best_p[2]), best_val
    else:
        return outer_center, outer_angle_deg, compute_required_side_fast(inner, outer_center, outer_angle_deg)

def joint_optimize(init_inner, init_outer_center, init_outer_angle, deadline):
    """Joint optimization of inner positions/angles and outer hexagon using contact-graph awareness."""
    def _unpack(params):
        il = []
        for i in range(N):
            il.append([params[i * 3], params[i * 3 + 1], params[i * 3 + 2]])
        ocx, ocy, oa = params[N * 3], params[N * 3 + 1], params[N * 3 + 2]
        return il, (ocx, ocy), oa

    def obj_joint_smooth(params, beta=60.0):
        il, oc, oa = _unpack(params)
        side = compute_required_side_smooth(il, oc, oa, beta=beta)
        pen = overlap_penalty(il)
        return side + 1000.0 * pen

    def obj_joint_exact(params):
        il, oc, oa = _unpack(params)
        side = compute_required_side_fast(il, oc, oa)
        pen = overlap_penalty(il)
        return side + 8000.0 * pen

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
        {'T': 1.0, 'T_min': 0.02, 'cool': 0.9997, 'pos_step': 0.35, 'ang_step': 15.0, 'oc_step': 0.25, 'oa_step': 6.0},
        {'T': 0.4, 'T_min': 0.005, 'cool': 0.99985, 'pos_step': 0.18, 'ang_step': 8.0, 'oc_step': 0.12, 'oa_step': 4.0},
        {'T': 0.15, 'T_min': 0.002, 'cool': 0.9999, 'pos_step': 0.08, 'ang_step': 4.0, 'oc_step': 0.06, 'oa_step': 2.0},
    ]
    
    for phase in phases:
        if time.time() > deadline - 30:
            break
        T = phase['T']
        T_min = phase['T_min']
        cool = phase['cool']
        max_iter = 6000
        for it in range(max_iter):
            if time.time() > deadline - 30:
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
    """High-precision Nelder-Mead polish with multi-start and coordinate descent."""
    def obj_polish(p):
        il = []
        for i in range(N):
            il.append([p[i * 3], p[i * 3 + 1], p[i * 3 + 2]])
        ocx, ocy, oa = p[N * 3], p[N * 3 + 1], p[N * 3 + 2]
        side = compute_required_side_fast(il, (ocx, ocy), oa)
        pen = overlap_penalty(il)
        return side + 8000.0 * pen

    best_p = params.copy()
    best_v = obj_polish(best_p)
    
    # Multi-start from slightly perturbed initial points
    np.random.seed(456)
    for trial in range(6):
        if time.time() > deadline - 10:
            break
        p_start = params.copy()
        p_start += np.random.normal(0, 0.008, size=p_start.shape)
        try:
            res = minimize(obj_polish, p_start, method='Nelder-Mead',
                           options={'maxiter': 2500, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': 4000})
            if res.fun < best_v:
                best_v = res.fun
                best_p = res.x.copy()
        except Exception:
            pass
    
    # Coordinate descent refinement
    if time.time() < deadline - 8:
        for _ in range(3):
            if time.time() > deadline - 8:
                break
            improved = False
            for i in range(N * 3 + 3):
                if time.time() > deadline - 8:
                    break
                for delta in [-0.005, 0.005]:
                    trial_p = best_p.copy()
                    trial_p[i] += delta
                    trial_v = obj_polish(trial_p)
                    if trial_v < best_v:
                        best_p = trial_p
                        best_v = trial_v
                        improved = True
            if not improved:
                break
    
    # Final tight polish from best
    if time.time() < deadline - 3:
        try:
            res = minimize(obj_polish, best_p, method='Nelder-Mead',
                           options={'maxiter': 6000, 'xatol': 1e-13, 'fatol': 1e-15, 'maxfev': 10000})
            if res.fun < best_v:
                best_p = res.x.copy()
        except Exception:
            pass
    
    return best_p

def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # 5s safety margin

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
    
    configs = [axial1, axial2, axial3, axial4, axial5, axial6, axial7]
    
    best_side = float('inf')
    best_inner = None
    best_outer_center = None
    best_outer_angle = 0.0
    best_params = None
    
    # Phase 1: For each config, optimize outer hexagon
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
            # Store params for joint optimization
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
        
        if check_disjoint(joint_inner, eps=1e-5):
            joint_side = compute_required_side_fast(joint_inner, joint_oc, joint_oa)
            if joint_side < best_side:
                best_inner = joint_inner
                best_outer_center = joint_oc
                best_outer_angle = joint_oa
                best_side = joint_side
                best_params = joint_params
        else:
            # Try to repair: use params but verify
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
        # Try a second joint optimization from a perturbed version
        np.random.seed(789)
        p2 = best_params.copy()
        for i in range(N):
            p2[i*3] += np.random.normal(0, 0.04)
            p2[i*3+1] += np.random.normal(0, 0.04)
            p2[i*3+2] += np.random.normal(0, 4.0)
        
        init_inner2 = []
        for i in range(N):
            init_inner2.append([float(p2[i*3]), float(p2[i*3+1]), float(p2[i*3+2])])
        init_oc2 = (float(p2[N*3]), float(p2[N*3+1]))
        init_oa2 = float(p2[N*3+2])
        
        joint_inner2, joint_oc2, joint_oa2, joint_params2 = joint_optimize(
            init_inner2, init_oc2, init_oa2, deadline)
        
        if check_disjoint(joint_inner2, eps=1e-5):
            joint_side2 = compute_required_side_fast(joint_inner2, joint_oc2, joint_oa2)
            if joint_side2 < best_side:
                best_inner = joint_inner2
                best_outer_center = joint_oc2
                best_outer_angle = joint_oa2
                best_side = joint_side2
                best_params = joint_params2
    
    # Phase 3: Final high-precision polish
    if time.time() < deadline - 12 and best_params is not None:
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
        # Fallback to first lattice config
        s0 = 1.0 + 1e-8
        best_inner = build_lattice_config(configs[0], s0)
        cx0 = sum(c[0] for c in best_inner) / len(best_inner)
        cy0 = sum(c[1] for c in best_inner) / len(best_inner)
        best_outer_center = (cx0, cy0)
        best_outer_angle = 0.0
        best_side = compute_required_side_fast(best_inner, best_outer_center, best_outer_angle)
    
    return best_inner, best_outer_center, best_side, best_outer_angle

# EVOLVE-BLOCK-END
