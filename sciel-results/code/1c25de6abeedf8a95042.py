"""Combined: rigid-composite support-function objective + contact-graph penalty + multi-start."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)
APOTHEM = math.sqrt(3) / 2.0

# EVOLVE-BLOCK-START

def hex_verts(cx, cy, side, angle_rad):
    """6 vertices of a regular hexagon as numpy array."""
    thetas = angle_rad + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + side * np.cos(thetas), cy + side * np.sin(thetas)])

def container_normals(outer_angle_deg):
    """6 outward normals of the container hexagon."""
    base = np.radians(30)
    a = math.radians(outer_angle_deg)
    cs, sn = math.cos(a), math.sin(a)
    normals = []
    for i in range(6):
        t = base + i * np.pi / 3.0
        nx = math.cos(t)
        ny = math.sin(t)
        rx = nx * cs - ny * sn
        ry = nx * sn + ny * cs
        normals.append((rx, ry))
    return normals

def support_function(inner, outer_center, outer_angle_deg):
    """
    Max projection of all inner hexagon vertices onto the 6 container normals.
    Returns the required side length.
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

def support_function_smooth(inner, outer_center, outer_angle_deg, beta=60.0):
    """Smooth log-sum-exp version of support function."""
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
    smooth_max = m + math.log(np.sum(np.exp(beta * (P - m)))) / beta
    if smooth_max < 0:
        smooth_max = 0.0
    return 2.0 * smooth_max / SQRT3

def sat_overlap(v1, v2, eps=1e-9):
    """SAT overlap for two convex polygons (numpy arrays)."""
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
    """Pairwise SAT disjointness."""
    n = len(inner)
    polys = [hex_verts(x, y, 1.0, math.radians(a)) for (x, y, a) in inner]
    for i in range(n):
        for j in range(i + 1, n):
            if sat_overlap(polys[i], polys[j], eps):
                return False
    return True

def contact_penalty(inner, min_dist=SQRT3 - 0.003):
    """Quadratic penalty for center-distance violations."""
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

def build_axial(axial_coords, s=1.0 + 1e-8):
    """Build inner hexagons from axial coords."""
    centers = [(1.5 * s * q, SQRT3 * s * (r + q / 2.0)) for q, r in axial_coords]
    return [[x, y, 0.0] for x, y in centers]

def rigid_composite_pose(local_positions, comp_angle_deg, offset):
    """
    Apply rigid-body pose to local positions.
    local_positions: list of (x, y, angle_deg) in composite frame
    comp_angle_deg: rotation of composite about its centroid
    offset: (dx, dy) translation
    Returns absolute positions.
    """
    cx0 = sum(p[0] for p in local_positions) / len(local_positions)
    cy0 = sum(p[1] for p in local_positions) / len(local_positions)
    ca = math.cos(math.radians(comp_angle_deg))
    sa = math.sin(math.radians(comp_angle_deg))
    result = []
    for (lx, ly, la) in local_positions:
        rx = lx - cx0
        ry = ly - cy0
        nx = cx0 + ca * rx - sa * ry + offset[0]
        ny = cy0 + sa * rx + ca * ry + offset[1]
        na = la + comp_angle_deg
        result.append([nx, ny, na])
    return result

def objective_rigid_smooth(params, local_positions, beta=50.0):
    """
    params: [comp_angle, dx, dy, ox, oy, outer_angle]
    """
    comp_angle = params[0]
    dx, dy = params[1], params[2]
    ox, oy = params[3], params[4]
    outer_angle = params[5]
    inner = rigid_composite_pose(local_positions, comp_angle, (dx, dy))
    return support_function_smooth(inner, (ox, oy), outer_angle, beta=beta)

def objective_rigid_exact(params, local_positions):
    comp_angle = params[0]
    dx, dy = params[1], params[2]
    ox, oy = params[3], params[4]
    outer_angle = params[5]
    inner = rigid_composite_pose(local_positions, comp_angle, (dx, dy))
    return support_function(inner, (ox, oy), outer_angle)

def optimize_rigid(local_positions, deadline, n_starts=8):
    """Multi-start optimization of rigid composite pose."""
    cx0 = sum(p[0] for p in local_positions) / len(local_positions)
    cy0 = sum(p[1] for p in local_positions) / len(local_positions)
    
    best_val = float('inf')
    best_params = None
    
    angle_candidates = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135]
    
    for k in range(n_starts):
        if time.time() > deadline - 30:
            break
        init = [
            angle_candidates[k % len(angle_candidates)],
            0.0, 0.0,
            cx0, cy0,
            angle_candidates[k % len(angle_candidates)]
        ]
        try:
            res_s = minimize(lambda p: objective_rigid_smooth(p, local_positions, beta=40.0),
                           init, method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-7, 'maxfev': 600})
            res_e = minimize(lambda p: objective_rigid_exact(p, local_positions),
                           res_s.x, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 1000})
            if res_e.fun < best_val:
                best_val = res_e.fun
                best_params = res_e.x.copy()
        except Exception:
            pass
    
    return best_val, best_params

def objective_joint(params, local_positions, min_dist=SQRT3 - 0.003):
    """
    params: [x0,y0,a0, x1,y1,a1, ..., xN,yN,aN, comp_angle, dx, dy, ox, oy, outer_angle]
    """
    n = len(local_positions)
    inner = []
    for i in range(n):
        x = params[i*3]
        y = params[i*3+1]
        a = params[i*3+2]
        inner.append([x, y, a])
    
    base = n * 3
    comp_angle = params[base]
    dx, dy = params[base+1], params[base+2]
    ox, oy = params[base+3], params[base+4]
    outer_angle = params[base+5]
    
    side = support_function(inner, (ox, oy), outer_angle)
    pen = contact_penalty(inner, min_dist)
    
    return side + 100.0 * pen

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    candidates = []

    # Candidate 1: Center + ring of 6 + 4 more (symmetric cross)
    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append((axial_1, "symmetric_cross"))

    # Candidate 2: Compact 3-4-3 hexagonal cluster
    axial_2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2)
    ]
    candidates.append((axial_2, "cluster343"))

    # Candidate 3: 4-4-3 block
    axial_3 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    candidates.append((axial_3, "443"))

    # Candidate 4: Diamond with center
    axial_4 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append((axial_4, "diamond"))

    # Candidate 5: 5-6 two rows
    axial_5 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)
    ]
    candidates.append((axial_5, "56rows"))

    best_global_side = float('inf')
    best_global_inner = None
    best_global_center = None
    best_global_angle = 0.0

    for idx, (axial_coords, name) in enumerate(candidates):
        if time.time() > deadline - 30:
            break

        if len(axial_coords) > N:
            axial_coords = axial_coords[:N]

        local_positions = build_axial(axial_coords)

        cx0 = sum(p[0] for p in local_positions) / len(local_positions)
        cy0 = sum(p[1] for p in local_positions) / len(local_positions)

        # Phase 1: Rigid composite optimization
        rigid_val, rigid_params = optimize_rigid(local_positions, deadline, n_starts=8)

        if rigid_params is not None:
            comp_angle = rigid_params[0]
            dx, dy = rigid_params[1], rigid_params[2]
            ox, oy = rigid_params[3], rigid_params[4]
            outer_angle = rigid_params[5]
            inner_rigid = rigid_composite_pose(local_positions, comp_angle, (dx, dy))
            current_side = rigid_val
            current_center = (ox, oy)
            current_angle = outer_angle
        else:
            inner_rigid = local_positions
            current_side = support_function(inner_rigid, (cx0, cy0), 0.0)
            current_center = (cx0, cy0)
            current_angle = 0.0

        # Phase 2: Joint optimization with simulated annealing
        def objective_joint_local(params):
            n = len(local_positions)
            inner = []
            for i in range(n):
                x = params[i*3]
                y = params[i*3+1]
                a = params[i*3+2]
                inner.append([x, y, a])
            
            base = n * 3
            ox = params[base]
            oy = params[base+1]
            outer_angle = params[base+2]
            
            side = support_function(inner, (ox, oy), outer_angle)
            pen = contact_penalty(inner, SQRT3 - 0.003)
            
            return side + 100.0 * pen

        initial_params = []
        for x, y, a in inner_rigid:
            initial_params.extend([x, y, a])
        initial_params.extend([current_center[0], current_center[1], current_angle])

        current_params = np.array(initial_params)
        current_val = objective_joint_local(current_params)
        best_params = current_params.copy()
        best_val_joint = current_val

        # Adaptive simulated annealing
        T = 1.0
        T_min = 0.002
        cooling = 0.99985

        max_iter = 8000
        no_improve_count = 0
        patience = 1200

        for i in range(max_iter):
            if time.time() > deadline - 18:
                break

            neighbor = current_params.copy()

            if np.random.random() < 0.7:
                dim_idx = np.random.randint(0, N*3)
                delta = np.random.normal(0, 0.12 * (1 + 0.5 * T))
                neighbor[dim_idx] += delta
            else:
                dim_idx = np.random.randint(N*3, N*3+3)
                if dim_idx == N*3+2:
                    delta = np.random.normal(0, 2.5 * (1 + 0.3 * T))
                else:
                    delta = np.random.normal(0, 0.08 * (1 + 0.5 * T))
                neighbor[dim_idx] += delta

            neighbor_val = objective_joint_local(neighbor)

            if neighbor_val < current_val or np.random.random() < math.exp(-(neighbor_val - current_val) / max(T, T_min)):
                current_params = neighbor
                current_val = neighbor_val

                if current_val < best_val_joint:
                    best_params = current_params.copy()
                    best_val_joint = current_val
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                if no_improve_count > patience:
                    no_improve_count = 0
                    T = max(T * 0.5, T_min * 2)

            T *= cooling
            if T < T_min:
                break

        best_inner_joint = []
        for i in range(N):
            x = best_params[i*3]
            y = best_params[i*3+1]
            a = best_params[i*3+2]
            best_inner_joint.append([x, y, a])

        best_ocx = best_params[N*3]
        best_ocy = best_params[N*3+1]
        best_oangle = best_params[N*3+2]

        # Phase 3: Final high-precision polish on outer container
        def objective_outer_polish(params):
            cx, cy, angle_deg = params
            return support_function(best_inner_joint, (cx, cy), angle_deg)

        polish_starts = []
        base_angle = best_oangle
        for da in [-12, -6, -3, 0, 3, 6, 12]:
            for dc in [(-0.1, 0), (0.1, 0), (0, -0.1), (0, 0.1), (0, 0)]:
                polish_starts.append([best_ocx + dc[0], best_ocy + dc[1], base_angle + da])

        best_polish_val = support_function(best_inner_joint, (best_ocx, best_ocy), best_oangle)
        best_polish_params = [best_ocx, best_ocy, best_oangle]

        for ps in polish_starts:
            if time.time() > deadline - 4:
                break
            try:
                res = minimize(objective_outer_polish, ps, method='Nelder-Mead',
                              options={'maxiter': 3000, 'xatol': 1e-13, 'fatol': 1e-15, 'maxfev': 6000})
                if res.fun < best_polish_val:
                    best_polish_val = res.fun
                    best_polish_params = res.x
            except Exception:
                pass

        final_inner = best_inner_joint
        final_center = (best_polish_params[0], best_polish_params[1])
        final_angle = best_polish_params[2]
        final_side = best_polish_val

        # Verify disjointness
        if check_disjoint(final_inner, eps=1e-4):
            if final_side < best_global_side:
                best_global_side = final_side
                best_global_inner = final_inner
                best_global_center = final_center
                best_global_angle = final_angle
        else:
            # Fall back to initial layout
            fallback_side = support_function(local_positions, (cx0, cy0), 0.0)
            if fallback_side < best_global_side:
                best_global_side = fallback_side
                best_global_inner = local_positions
                best_global_center = (cx0, cy0)
                best_global_angle = 0.0

    if best_global_inner is None:
        axial_coords = candidates[0][0][:N]
        centers = [(1.5 * q, SQRT3 * (r + q / 2.0)) for q, r in axial_coords]
        best_global_inner = [[x, y, 0.0] for x, y in centers]
        cx_avg = sum(c[0] for c in centers) / len(centers)
        cy_avg = sum(c[1] for c in centers) / len(centers)
        best_global_center = (cx_avg, cy_avg)
        best_global_angle = 0.0
        best_global_side = support_function(best_global_inner, best_global_center, best_global_angle)

    return best_global_inner, best_global_center, best_global_side, best_global_angle

# EVOLVE-BLOCK-END
