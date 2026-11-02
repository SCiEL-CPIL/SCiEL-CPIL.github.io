"""Optimal packing of 11 unit hexagons in minimal outer hexagon.
Strategy: flower topology with optimized rotation and container alignment."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

def hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon."""
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

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=80.0):
    """Smooth surrogate for the max projection."""
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
    """SAT overlap check for two convex polygons."""
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

def build_flower_config(outer_angle_deg=0.0, inner_rotation_deg=0.0, spacing=1.0 + 1e-8):
    """
    Build the flower configuration: 1 center + 6 ring + 4 outer.
    The 4 outer hexagons are placed at alternating positions in a second ring.
    """
    s = spacing
    # Center hexagon
    centers = [(0.0, 0.0)]
    
    # Ring 1: 6 hexagons at distance sqrt(3)*s from center
    for i in range(6):
        angle = math.radians(60 * i)
        x = SQRT3 * s * math.cos(angle)
        y = SQRT3 * s * math.sin(angle)
        centers.append((x, y))
    
    # Ring 2: 4 hexagons at positions between ring 1 hexagons
    # These are at axial positions (2,0), (1,1), (-1,1), (-2,0) in a hex grid
    # In Cartesian: distance from center = sqrt(3)*s*sqrt(3) = 3*s
    # Actually let's use axial coordinates properly
    # Axial (q,r) -> Cartesian: x = 1.5*s*q, y = sqrt(3)*s*(r + q/2)
    ring2_axial = [(2, -1), (1, 1), (-1, 1), (-2, -1)]
    for q, r in ring2_axial:
        x = 1.5 * s * q
        y = SQRT3 * s * (r + q / 2.0)
        centers.append((x, y))
    
    inner = []
    for (cx, cy) in centers:
        inner.append([cx, cy, inner_rotation_deg])
    
    return inner

def optimize_flower(deadline):
    """
    Optimize the flower configuration.
    Parameters: outer_angle, inner_rotation, spacing, and possibly per-hex adjustments.
    """
    def obj_smooth(params):
        outer_angle = params[0]
        inner_rot = params[1]
        spacing = params[2]
        inner = build_flower_config(outer_angle, inner_rot, spacing)
        # Center the configuration at origin
        cx = sum(c[0] for c in inner) / len(inner)
        cy = sum(c[1] for c in inner) / len(inner)
        inner_centered = [[c[0] - cx, c[1] - cy, c[2]] for c in inner]
        return compute_required_side_smooth(inner_centered, (0.0, 0.0), outer_angle, beta=60.0)
    
    def obj_exact(params):
        outer_angle = params[0]
        inner_rot = params[1]
        spacing = params[2]
        inner = build_flower_config(outer_angle, inner_rot, spacing)
        cx = sum(c[0] for c in inner) / len(inner)
        cy = sum(c[1] for c in inner) / len(inner)
        inner_centered = [[c[0] - cx, c[1] - cy, c[2]] for c in inner]
        return compute_required_side_fast(inner_centered, (0.0, 0.0), outer_angle)
    
    best_val = float('inf')
    best_params = None
    
    # Multi-start
    for oa in range(0, 60, 5):
        for ir in [-15, -10, -5, 0, 5, 10, 15]:
            if time.time() > deadline - 30:
                break
            p0 = np.array([float(oa), float(ir), 1.0 + 1e-8])
            try:
                r1 = minimize(obj_smooth, p0, method='Nelder-Mead',
                              options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 300})
                # Constrain spacing to be >= 1.0
                if r1.x[2] < 1.0:
                    r1.x[2] = 1.0
                r2 = minimize(obj_exact, r1.x, method='Nelder-Mead',
                              options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': 800})
                if r2.x[2] < 1.0:
                    r2.x[2] = 1.0
                val = obj_exact(r2.x)
                if val < best_val:
                    best_val = val
                    best_params = r2.x.copy()
            except Exception:
                pass
    
    if best_params is None:
        return None, None, None, None, float('inf')
    
    outer_angle = float(best_params[0])
    inner_rot = float(best_params[1])
    spacing = float(best_params[2])
    if spacing < 1.0:
        spacing = 1.0
    
    inner = build_flower_config(outer_angle, inner_rot, spacing)
    cx = sum(c[0] for c in inner) / len(inner)
    cy = sum(c[1] for c in inner) / len(inner)
    inner_centered = [[c[0] - cx, c[1] - cy, c[2]] for c in inner]
    
    return inner_centered, (0.0, 0.0), outer_angle, best_params, best_val

def full_optimize(init_inner, init_oc, init_oa, deadline):
    """Full joint optimization with simulated annealing."""
    def obj_joint_smooth(params, beta=50.0):
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
    T_min = 0.002
    cool = 0.9997
    max_iter = 10000
    np.random.seed(42)
    for it in range(max_iter):
        if time.time() > deadline - 15:
            break
        nb = current.copy()
        r = np.random.random()
        if r < 0.60:
            idx = np.random.randint(0, N)
            dim = np.random.randint(0, 2)
            nb[idx * 3 + dim] += np.random.normal(0, 0.2)
        elif r < 0.80:
            idx = np.random.randint(0, N)
            nb[idx * 3 + 2] += np.random.normal(0, 8.0)
        elif r < 0.90:
            dim = np.random.randint(0, 2)
            nb[N * 3 + dim] += np.random.normal(0, 0.1)
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

    try:
        res = minimize(obj_polish, params, method='Nelder-Mead',
                       options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': 8000})
        return res.x
    except Exception:
        return params

def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # 5s safety margin

    best_side = float('inf')
    best_inner = None
    best_outer_center = None
    best_outer_angle = 0.0
    best_params = None

    # Phase 1: Optimize flower configuration
    if time.time() < deadline - 60:
        inner_f, oc_f, oa_f, params_f, val_f = optimize_flower(deadline)
        if inner_f is not None and check_disjoint(inner_f, eps=1e-5):
            best_side = val_f
            best_inner = inner_f
            best_outer_center = oc_f
            best_outer_angle = oa_f
            p0 = np.zeros(N * 3 + 3)
            for i, (x, y, a) in enumerate(inner_f):
                p0[i * 3] = x
                p0[i * 3 + 1] = y
                p0[i * 3 + 2] = a
            p0[N * 3] = oc_f[0]
            p0[N * 3 + 1] = oc_f[1]
            p0[N * 3 + 2] = oa_f
            best_params = p0.copy()

    # Fallback: simple lattice if flower failed
    if best_inner is None:
        axial = [(0, 0),
                 (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
                 (2, -1), (1, 1), (-2, 1), (-1, -1)]
        s0 = 1.0 + 1e-8
        best_inner = []
        for q, r in axial:
            x = 1.5 * s0 * q
            y = SQRT3 * s0 * (r + q / 2.0)
            best_inner.append([x, y, 0.0])
        cx0 = sum(c[0] for c in best_inner) / len(best_inner)
        cy0 = sum(c[1] for c in best_inner) / len(best_inner)
        best_inner = [[c[0] - cx0, c[1] - cy0, c[2]] for c in best_inner]
        best_outer_center = (0.0, 0.0)
        best_outer_angle = 0.0
        best_side = compute_required_side_fast(best_inner, best_outer_center, best_outer_angle)
        p0 = np.zeros(N * 3 + 3)
        for i, (x, y, a) in enumerate(best_inner):
            p0[i * 3] = x
            p0[i * 3 + 1] = y
            p0[i * 3 + 2] = a
        p0[N * 3] = 0.0
        p0[N * 3 + 1] = 0.0
        p0[N * 3 + 2] = 0.0
        best_params = p0.copy()

    # Phase 2: Full joint optimization
    if time.time() < deadline - 25:
        joint_inner, joint_oc, joint_oa, joint_params = full_optimize(
            best_inner, best_outer_center, best_outer_angle, deadline)
        
        if check_disjoint(joint_inner, eps=1e-5):
            joint_side = compute_required_side_fast(joint_inner, joint_oc, joint_oa)
            if joint_side < best_side:
                best_inner = joint_inner
                best_outer_center = joint_oc
                best_outer_angle = joint_oa
                best_side = joint_side
                best_params = joint_params

    # Phase 3: Final high-precision polish
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
        axial = [(0, 0),
                 (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
                 (2, -1), (1, 1), (-2, 1), (-1, -1)]
        s0 = 1.0 + 1e-8
        best_inner = []
        for q, r in axial:
            x = 1.5 * s0 * q
            y = SQRT3 * s0 * (r + q / 2.0)
            best_inner.append([x, y, 0.0])
        cx0 = sum(c[0] for c in best_inner) / len(best_inner)
        cy0 = sum(c[1] for c in best_inner) / len(best_inner)
        best_inner = [[c[0] - cx0, c[1] - cy0, c[2]] for c in best_inner]
        best_outer_center = (0.0, 0.0)
        best_outer_angle = 0.0
        best_side = compute_required_side_fast(best_inner, best_outer_center, best_outer_angle)

    return best_inner, best_outer_center, best_side, best_outer_angle

# EVOLVE-BLOCK-END
