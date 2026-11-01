"""Packing unit hexagons inside a regular hexagon using smooth optimization."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)
APOTHEM = SQRT3 / 2.0
DEADLINE = 145.0  # safety margin before 150s kill

# Axial coordinate candidates for initial layouts
CANDIDATE_AXIALS = [
    # Classic hexagonal cluster (7+4)
    [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2),
    ],
    # Alternative ring
    [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, 2),
    ],
    # Rectangular-ish
    [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1),
    ],
    # Another variation
    [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2), (3, 2),
    ],
    # Star-like
    [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (1, 2), (-1, 2), (0, 3),
    ],
    # Dense cluster
    [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3),
    ],
    # More diverse candidates
    [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ],
    [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)
    ],
    [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ],
    [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1), (0, -1)
    ],
    # Additional topologies
    [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2), (3, 2)
    ],
    [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 1)
    ],
    [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3), (1, -1)
    ],
]


def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon."""
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        vx = cx + side * math.cos(theta)
        vy = cy + side * math.sin(theta)
        verts.append((vx, vy))
    return verts


def compute_required_side(inner, outer_center, outer_angle_deg):
    """
    Compute the minimum outer side length to contain all inner hexagons.
    inner: list of (x, y, angle_deg)
    outer_center: (x, y)
    outer_angle_deg: rotation of the outer hexagon
    """
    if not inner:
        return 0.0

    ox, oy = outer_center
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))

    max_projection = 0.0

    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a

            for i in range(6):
                theta = math.radians(30 + i * 60)
                nx = math.cos(theta)
                ny = math.sin(theta)
                proj = lx * nx + ly * ny
                if proj > max_projection:
                    max_projection = proj

    return 2.0 * max_projection / SQRT3


def sat_overlap_check(v1, v2, eps=1e-9):
    """
    Check if two convex polygons (defined by their vertices) overlap using SAT.
    v1, v2: lists of (x, y) tuples.
    Returns True if they overlap (including touching).
    """
    axes = []

    for i in range(len(v1)):
        p1 = v1[i]
        p2 = v1[(i + 1) % len(v1)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0] ** 2 + normal[1] ** 2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))

    for i in range(len(v2)):
        p1 = v2[i]
        p2 = v2[(i + 1) % len(v2)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0] ** 2 + normal[1] ** 2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))

    for axis in axes:
        min1 = min(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        max1 = max(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        min2 = min(p[0] * axis[0] + p[1] * axis[1] for p in v2)
        max2 = max(p[0] * axis[0] + p[1] * axis[1] for p in v2)

        if max1 < min2 + eps or max2 < min1 + eps:
            return False

    return True


def check_all_disjoint(inner, eps=1e-6):
    """Check if all inner hexagons are pairwise disjoint using SAT."""
    n = len(inner)
    for i in range(n):
        for j in range(i + 1, n):
            vi = get_hex_vertices(inner[i][0], inner[i][1], 1.0, math.radians(inner[i][2]))
            vj = get_hex_vertices(inner[j][0], inner[j][1], 1.0, math.radians(inner[j][2]))
            if sat_overlap_check(vi, vj, eps):
                return False
    return True


def check_inside_outer(inner, outer_center, outer_side, outer_angle_deg, eps=1e-6):
    """Check all inner hexagons are inside the outer hexagon."""
    ox, oy = outer_center
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))

    apothem = outer_side * SQRT3 / 2.0

    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a

            for i in range(6):
                theta = math.radians(30 + i * 60)
                nx = math.cos(theta)
                ny = math.sin(theta)
                if lx * nx + ly * ny > apothem + eps:
                    return False
    return True


def axial_to_cartesian(axial_list, scale=1.0 + 1e-6):
    """Convert axial coordinates to cartesian with slight scaling to avoid touching."""
    pts = []
    for q, r in axial_list:
        x = 1.5 * scale * q
        y = SQRT3 * scale * (r + q / 2.0)
        pts.append((x, y))
    return pts


def pbd_resolve_collision(inner_list, outer_center, outer_side, outer_angle_deg,
                          iterations=50, damping=0.8, start_time=None, deadline=145.0):
    """
    Position-Based Dynamics style collision resolution.
    Iteratively projects hexagons to resolve overlaps with each other and the container.
    """
    if start_time is None:
        start_time = time.time()
    
    velocities = [[0.0, 0.0] for _ in range(len(inner_list))]
    
    ox, oy = outer_center
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    apothem = outer_side * SQRT3 / 2.0
    
    for it in range(iterations):
        if time.time() - start_time > deadline - 5:
            break
        
        for i in range(len(inner_list)):
            velocities[i][0] *= damping
            velocities[i][1] *= damping
        
        for i in range(len(inner_list)):
            inner_list[i][0] += velocities[i][0]
            inner_list[i][1] += velocities[i][1]
        
        for i in range(len(inner_list)):
            for j in range(i + 1, len(inner_list)):
                vi = get_hex_vertices(inner_list[i][0], inner_list[i][1], 1.0, math.radians(inner_list[i][2]))
                vj = get_hex_vertices(inner_list[j][0], inner_list[j][1], 1.0, math.radians(inner_list[j][2]))
                
                if sat_overlap_check(vi, vj, eps=-1e-6):
                    ci = (inner_list[i][0], inner_list[i][1])
                    cj = (inner_list[j][0], inner_list[j][1])
                    dx = cj[0] - ci[0]
                    dy = cj[1] - ci[1]
                    dist = math.sqrt(dx * dx + dy * dy)
                    
                    if dist < 1e-9:
                        angle = np.random.uniform(0, 2 * math.pi)
                        dx = math.cos(angle)
                        dy = math.sin(angle)
                        dist = 1.0
                    
                    min_sep = SQRT3 - 0.01
                    if dist < min_sep:
                        push = (min_sep - dist) / 2.0
                        nx = dx / dist
                        ny = dy / dist
                        
                        inner_list[i][0] -= push * nx
                        inner_list[i][1] -= push * ny
                        inner_list[j][0] += push * nx
                        inner_list[j][1] += push * ny
                        
                        velocities[i][0] -= push * nx * 0.5
                        velocities[i][1] -= push * ny * 0.5
                        velocities[j][0] += push * nx * 0.5
                        velocities[j][1] += push * ny * 0.5
        
        for i in range(len(inner_list)):
            verts = get_hex_vertices(inner_list[i][0], inner_list[i][1], 1.0, math.radians(inner_list[i][2]))
            
            for v in verts:
                dx = v[0] - ox
                dy = v[1] - oy
                lx = dx * cos_a + dy * sin_a
                ly = -dx * sin_a + dy * cos_a
                
                for k in range(6):
                    theta = math.radians(30 + k * 60)
                    nx = math.cos(theta)
                    ny = math.sin(theta)
                    proj = lx * nx + ly * ny
                    
                    if proj > apothem:
                        push = proj - apothem
                        wx = nx * cos_a - ny * sin_a
                        wy = nx * sin_a + ny * cos_a
                        
                        inner_list[i][0] -= push * wx
                        inner_list[i][1] -= push * wy
                        
                        velocities[i][0] -= push * wx * 0.3
                        velocities[i][1] -= push * wy * 0.3
    
    return inner_list


def smooth_objective(params, inner_angles, outer_angle_deg, temperature=0.01):
    """
    Smooth surrogate objective: minimize the outer side length using soft-max.
    params: [x1, y1, x2, y2, ..., xN, yN, ox, oy]
    inner_angles: list of N angles in radians
    outer_angle_deg: rotation of outer hexagon
    temperature: soft-max temperature (lower = sharper)
    """
    N = len(inner_angles)
    inner_pts = np.array([[params[2*i], params[2*i+1]] for i in range(N)])
    ox, oy = params[2*N], params[2*N+1]
    
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    
    # Compute all vertex projections
    all_projs = []
    for i in range(N):
        cx, cy = inner_pts[i]
        angle = inner_angles[i]
        for k in range(6):
            theta = angle + k * math.pi / 3.0
            vx = cx + math.cos(theta)
            vy = cy + math.sin(theta)
            dx = vx - ox
            dy = vy - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            
            for m in range(6):
                theta2 = math.radians(30 + m * 60)
                nx = math.cos(theta2)
                ny = math.sin(theta2)
                proj = lx * nx + ly * ny
                all_projs.append(proj)
    
    all_projs = np.array(all_projs)
    
    # Soft-max approximation
    max_proj = np.max(all_projs)
    exp_projs = np.exp((all_projs - max_proj) / temperature)
    soft_max = max_proj + temperature * np.log(np.sum(exp_projs))
    
    side = 2.0 * soft_max / SQRT3
    
    # Add penalty for overlaps (smooth approximation)
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = inner_pts[j][0] - inner_pts[i][0]
            dy = inner_pts[j][1] - inner_pts[i][1]
            dist = math.sqrt(dx * dx + dy * dy)
            min_sep = SQRT3
            if dist < min_sep:
                penalty += (min_sep - dist) ** 2
    
    return side + 10.0 * penalty


def smooth_gradient(params, inner_angles, outer_angle_deg, temperature=0.01):
    """
    Numerical gradient of the smooth objective.
    """
    eps = 1e-5
    grad = np.zeros(len(params))
    base = smooth_objective(params, inner_angles, outer_angle_deg, temperature)
    
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        f_plus = smooth_objective(params_plus, inner_angles, outer_angle_deg, temperature)
        grad[i] = (f_plus - base) / eps
    
    return grad


def optimize_layout(inner_init, outer_angle_deg, start_time, deadline=DEADLINE):
    """
    Optimize inner positions and outer center using smooth optimization.
    """
    N = len(inner_init)
    inner_angles = [math.radians(a) for _, _, a in inner_init]
    
    # Initial parameters: [x1, y1, ..., xN, yN, ox, oy]
    params = np.array([inner_init[i][0] for i in range(N)] + 
                       [inner_init[i][1] for i in range(N)] +
                       [sum(p[0] for p in inner_init) / N, 
                        sum(p[1] for p in inner_init) / N])
    
    best_params = params.copy()
    best_val = smooth_objective(params, inner_angles, outer_angle_deg, 0.01)
    
    # Multi-start optimization
    for restart in range(3):
        if time.time() - start_time > deadline - 10:
            break
        
        if restart > 0:
            np.random.seed(42 + restart)
            params = best_params + np.random.normal(0, 0.05, len(best_params))
        
        # Use Nelder-Mead for smooth optimization
        try:
            result = minimize(
                lambda p: smooth_objective(p, inner_angles, outer_angle_deg, 0.01),
                params,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x
        except Exception:
            pass
    
    return best_params, best_val


def pbd_optimize_outer(inner_list, start_time=None, deadline=DEADLINE):
    """
    Optimize outer hexagon parameters (center and angle) for a fixed inner layout.
    """
    if start_time is None:
        start_time = time.time()
    
    cx = sum(x[0] for x in inner_list) / len(inner_list)
    cy = sum(x[1] for x in inner_list) / len(inner_list)
    
    best_side = float('inf')
    best_params = (cx, cy, 0.0)
    
    init_angles = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]
    
    for init_angle in init_angles:
        if time.time() - start_time > deadline - 10:
            break
        
        for trial in range(3):
            if time.time() - start_time > deadline - 10:
                break
            
            perturb_r = 0.1 if trial > 0 else 0.0
            px = cx + np.random.uniform(-perturb_r, perturb_r) if trial > 0 else cx
            py = cy + np.random.uniform(-perturb_r, perturb_r) if trial > 0 else cy
            
            side = compute_required_side(inner_list, (px, py), init_angle)
            
            if side < best_side:
                best_side = side
                best_params = (px, py, init_angle)
    
    return best_params, best_side


def construct():
    start_time = time.time()

    best_side = float('inf')
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0

    # Try multiple candidate initial layouts
    for idx, axial_coords in enumerate(CANDIDATE_AXIALS):
        if time.time() - start_time > DEADLINE - 25:
            break

        centers = axial_to_cartesian(axial_coords)
        initial_inner = [[x, y, 0.0] for x, y in centers]

        # Phase 1: PBD collision resolution
        cx_avg = sum(c[0] for c in centers) / len(centers)
        cy_avg = sum(c[1] for c in centers) / len(centers)
        est_side = compute_required_side(initial_inner, (cx_avg, cy_avg), 0.0)
        
        if time.time() - start_time < DEADLINE - 40:
            pbd_resolve_collision(initial_inner, (cx_avg, cy_avg), est_side * 1.1, 0.0,
                                  iterations=80, damping=0.7, start_time=start_time, deadline=DEADLINE)
        
        # Phase 2: Optimize outer hexagon parameters
        if time.time() - start_time < DEADLINE - 20:
            outer_params, cur_side = pbd_optimize_outer(initial_inner, start_time, DEADLINE)
            cur_outer_center = (outer_params[0], outer_params[1])
            cur_outer_angle = outer_params[2]
        else:
            cur_outer_center = (cx_avg, cy_avg)
            cur_outer_angle = 0.0
            cur_side = compute_required_side(initial_inner, cur_outer_center, cur_outer_angle)

        # Phase 3: Smooth optimization of inner positions + outer center
        best_smooth_side = cur_side
        best_smooth_inner = [list(x) for x in initial_inner]
        best_smooth_center = cur_outer_center
        best_smooth_angle = cur_outer_angle
        
        if time.time() - start_time < DEADLINE - 15:
            # Try smooth optimization
            for angle_try in [0, 15, 30, 45]:
                if time.time() - start_time > DEADLINE - 12:
                    break
                
                params, val = optimize_layout(initial_inner, angle_try, start_time, DEADLINE)
                
                # Extract inner positions from params
                N = len(initial_inner)
                smooth_inner = []
                for i in range(N):
                    smooth_inner.append([params[2*i], params[2*i+1], initial_inner[i][2]])
                
                smooth_center = (params[2*N], params[2*N+1])
                
                # Compute actual side length (not the smooth approximation)
                actual_side = compute_required_side(smooth_inner, smooth_center, angle_try)
                
                # Check validity
                if check_all_disjoint(smooth_inner) and check_inside_outer(smooth_inner, smooth_center, actual_side, angle_try):
                    if actual_side < best_smooth_side:
                        best_smooth_side = actual_side
                        best_smooth_inner = smooth_inner
                        best_smooth_center = smooth_center
                        best_smooth_angle = angle_try
        
        # Phase 4: Joint PBD optimization
        best_joint_side = min(cur_side, best_smooth_side)
        if best_smooth_side < cur_side:
            best_joint_inner = best_smooth_inner
            best_joint_center = best_smooth_center
            best_joint_angle = best_smooth_angle
        else:
            best_joint_inner = [list(x) for x in initial_inner]
            best_joint_center = cur_outer_center
            best_joint_angle = cur_outer_angle
        
        for restart in range(6):
            if time.time() - start_time > DEADLINE - 8:
                break
            
            current_inner = [list(x) for x in best_joint_inner]
            if restart > 0:
                np.random.seed(42 + restart + idx * 100)
                for i in range(len(current_inner)):
                    current_inner[i][0] += np.random.uniform(-0.02, 0.02)
                    current_inner[i][1] += np.random.uniform(-0.02, 0.02)
            
            if time.time() - start_time < DEADLINE - 6:
                pbd_resolve_collision(current_inner, best_joint_center, best_joint_side * 1.05, best_joint_angle,
                                      iterations=60, damping=0.75, start_time=start_time, deadline=DEADLINE)
            
            side = compute_required_side(current_inner, best_joint_center, best_joint_angle)
            
            if check_all_disjoint(current_inner) and check_inside_outer(current_inner, best_joint_center, side, best_joint_angle):
                if side < best_joint_side:
                    best_joint_side = side
                    best_joint_inner = [list(x) for x in current_inner]
        
        # Phase 5: Final polish - optimize outer params
        if time.time() - start_time < DEADLINE - 5:
            polish_params, polish_side = pbd_optimize_outer(best_joint_inner, start_time, DEADLINE)
            if polish_side < best_joint_side:
                best_joint_center = (polish_params[0], polish_params[1])
                best_joint_angle = polish_params[2]
                best_joint_side = polish_side

        # Update global best
        if best_joint_side < best_side:
            best_side = best_joint_side
            best_inner = [list(x) for x in best_joint_inner]
            best_outer_center = best_joint_center
            best_outer_angle = best_joint_angle

    # Fallback if no valid solution found
    if best_inner is None:
        axial_coords = CANDIDATE_AXIALS[0]
        scale_fallback = 1.05
        fallback_centers = []
        for q, r in axial_coords:
            x = 1.5 * scale_fallback * q
            y = SQRT3 * scale_fallback * (r + q / 2.0)
            fallback_centers.append((x, y))

        best_inner = [[x, y, 0.0] for x, y in fallback_centers]
        best_outer_center = (0.0, 0.0)
        best_outer_angle = 0.0
        best_side = compute_required_side(best_inner, best_outer_center, best_outer_angle)

    # Final verification
    if not check_all_disjoint(best_inner) or not check_inside_outer(best_inner, best_outer_center, best_side, best_outer_angle):
        scale_fix = 1.01
        axial_coords = CANDIDATE_AXIALS[0]
        fixed_centers = []
        for q, r in axial_coords:
            x = 1.5 * scale_fix * q
            y = SQRT3 * scale_fix * (r + q / 2.0)
            fixed_centers.append((x, y))

        best_inner = [[x, y, 0.0] for x, y in fixed_centers]
        best_outer_center = (0.0, 0.0)
        best_outer_angle = 0.0
        best_side = compute_required_side(best_inner, best_outer_center, best_outer_angle)

    return best_inner, best_outer_center, best_side, best_outer_angle
