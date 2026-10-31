"""Packing unit hexagons inside a regular hexagon."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)
DEADLINE = 135.0  # safety margin before 150s kill


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
    # Normals for a hexagon with angle 0 (vertices at 0, 60, ...):
    # Outward normals are at 30, 90, 150, 210, 270, 330 degrees.
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))

    max_projection = 0.0

    # Rotate outer frame to match outer_angle
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))

    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            # Translate to outer center
            dx = v[0] - ox
            dy = v[1] - oy
            # Rotate points by -outer_angle to align with standard hex axes
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a

            for nx, ny in normals:
                proj = lx * nx + ly * ny
                if proj > max_projection:
                    max_projection = proj

    # S * sqrt(3) / 2 >= max_projection
    # S >= 2 * max_projection / sqrt(3)
    return 2.0 * max_projection / SQRT3


def sat_overlap_check(v1, v2, eps=1e-9):
    """
    Check if two convex polygons (defined by their vertices) overlap using SAT.
    v1, v2: lists of (x, y) tuples.
    Returns True if they overlap (including touching).
    """
    axes = []

    # Get axes from edges of v1
    for i in range(len(v1)):
        p1 = v1[i]
        p2 = v1[(i + 1) % len(v1)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0] ** 2 + normal[1] ** 2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))

    # Get axes from edges of v2
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

        # Check for gap
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


def construct():
    start_time = time.time()

    # 1. Define 11 hexagons on a hexagonal lattice - tighter configuration
    # Use a more compact hexagonal arrangement
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (-1, 2), (-2, 1)
    ]

    if len(axial_coords) > N:
        axial_coords = axial_coords[:N]

    # 2. Convert axial to Cartesian with tight spacing
    epsilon = 1e-4  # Slightly larger epsilon for numerical stability
    scale = 1.0 + epsilon

    centers = []
    for q, r in axial_coords:
        x = 1.5 * scale * q
        y = SQRT3 * scale * (r + q / 2.0)
        centers.append((x, y))

    # Initial layout: All hexagons aligned (angle 0)
    initial_inner = [[x, y, 0.0] for x, y in centers]

    best_side = float('inf')
    best_inner = [list(x) for x in initial_inner]
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0

    # 3. Optimize Outer Hexagon using scipy with multiple starts
    # Variables: [cx, cy, angle_deg]

    def objective_outer(params, inner_list):
        cx, cy, angle_deg = params
        side = compute_required_side(inner_list, (cx, cy), angle_deg)
        return side

    # Initial guess: centroid and 0
    cx_avg = sum(c[0] for c in centers) / len(centers)
    cy_avg = sum(c[1] for c in centers) / len(centers)

    # Try multiple diverse initial angles to avoid local minima
    best_result = None
    best_val = float('inf')

    # More comprehensive angle sampling
    init_angles = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 
                   95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180]

    for init_angle in init_angles:
        if time.time() - start_time > DEADLINE - 25:
            break

        init_params = [cx_avg, cy_avg, float(init_angle)]

        try:
            res = minimize(
                lambda p: objective_outer(p, initial_inner),
                init_params,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 500}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_result = res.x
        except Exception:
            pass

    if best_result is not None:
        best_outer_center = (best_result[0], best_result[1])
        best_outer_angle = best_result[2]
        best_side = best_val
    else:
        # Fallback
        best_outer_center = (cx_avg, cy_avg)
        best_outer_angle = 0.0
        best_side = compute_required_side(initial_inner, best_outer_center, best_outer_angle)

    # 4. Joint optimization of inner positions and outer parameters
    # Variables: [x1,y1,a1, x2,y2,a2, ..., xN,yN,aN, cx, cy, angle]
    # Total: 3*N + 3 = 36 variables

    def objective_joint(params):
        inner_list = []
        for i in range(N):
            x = params[i * 3]
            y = params[i * 3 + 1]
            a = params[i * 3 + 2]
            inner_list.append([x, y, a])

        cx = params[N * 3]
        cy = params[N * 3 + 1]
        angle = params[N * 3 + 2]

        side = compute_required_side(inner_list, (cx, cy), angle)

        # Penalty for overlaps: use center distance as proxy
        penalty = 0.0
        min_dist = SQRT3 - 0.01  # Safe margin
        for i in range(N):
            for j in range(i + 1, N):
                xi, yi, _ = inner_list[i]
                xj, yj, _ = inner_list[j]
                dist = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2 * 100.0

        # Penalty for going outside (soft constraint)
        ox, oy = cx, cy
        cos_a = math.cos(math.radians(angle))
        sin_a = math.sin(math.radians(angle))
        apothem = side * SQRT3 / 2.0
        for (ix, iy, iangle) in inner_list:
            verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
            for v in verts:
                dx = v[0] - ox
                dy = v[1] - oy
                lx = dx * cos_a + dy * sin_a
                ly = -dx * sin_a + dy * cos_a
                for k in range(6):
                    theta = math.radians(30 + k * 60)
                    proj = lx * math.cos(theta) + ly * math.sin(theta)
                    if proj > apothem:
                        penalty += (proj - apothem) ** 2 * 50.0

        return side + penalty

    # Initial joint params
    init_params = []
    for x, y, a in initial_inner:
        init_params.extend([x, y, a])
    init_params.extend([best_outer_center[0], best_outer_center[1], best_outer_angle])

    # Try joint optimization with more restarts
    best_joint_side = best_side
    best_joint_inner = [list(x) for x in initial_inner]
    best_joint_center = best_outer_center
    best_joint_angle = best_outer_angle

    num_restarts = 5
    for restart in range(num_restarts):
        if time.time() - start_time > DEADLINE - 15:
            break

        current_params = list(init_params)
        if restart > 0:
            # Add random perturbation with increasing magnitude
            np.random.seed(42 + restart * 7)
            perturb_scale = 0.02 + restart * 0.01
            for i in range(len(current_params)):
                if i < N * 3:
                    current_params[i] += np.random.uniform(-perturb_scale, perturb_scale)
                elif i == N * 3 + 2:  # angle
                    current_params[i] += np.random.uniform(-5, 5)

        try:
            res = minimize(
                objective_joint,
                current_params,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-9, 'maxfev': 600}
            )

            # Extract result
            inner_list = []
            for i in range(N):
                inner_list.append([res.x[i * 3], res.x[i * 3 + 1], res.x[i * 3 + 2]])
            oc = (res.x[N * 3], res.x[N * 3 + 1])
            oa = res.x[N * 3 + 2]

            side = compute_required_side(inner_list, oc, oa)

            # Check validity
            if check_all_disjoint(inner_list) and check_inside_outer(inner_list, oc, side, oa):
                if side < best_joint_side:
                    best_joint_side = side
                    best_joint_inner = inner_list
                    best_joint_center = oc
                    best_joint_angle = oa
        except Exception:
            pass

    # 5. Final high-precision polish pass
    if time.time() - start_time < DEADLINE - 5:
        # Refine the best solution found with tighter tolerances
        polish_params = []
        for x, y, a in best_joint_inner:
            polish_params.extend([x, y, a])
        polish_params.extend([best_joint_center[0], best_joint_center[1], best_joint_angle])

        try:
            res = minimize(
                objective_joint,
                polish_params,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 1500}
            )

            inner_list = []
            for i in range(N):
                inner_list.append([res.x[i * 3], res.x[i * 3 + 1], res.x[i * 3 + 2]])
            oc = (res.x[N * 3], res.x[N * 3 + 1])
            oa = res.x[N * 3 + 2]

            side = compute_required_side(inner_list, oc, oa)

            if check_all_disjoint(inner_list) and check_inside_outer(inner_list, oc, side, oa):
                if side < best_joint_side:
                    best_joint_side = side
                    best_joint_inner = inner_list
                    best_joint_center = oc
                    best_joint_angle = oa
        except Exception:
            pass

    # 6. If joint optimization didn't find valid, try local search on initial layout
    if best_joint_side >= best_side:
        # Use the outer-optimized result with original inner positions
        best_joint_inner = [list(x) for x in initial_inner]
        best_joint_side = best_side
        best_joint_center = best_outer_center
        best_joint_angle = best_outer_angle

    # 7. Final verification and fallback
    final_inner = best_joint_inner
    final_center = best_joint_center
    final_side = best_joint_side
    final_angle = best_joint_angle

    if not check_all_disjoint(final_inner) or not check_inside_outer(final_inner, final_center, final_side, final_angle):
        # Fallback: use original layout with safe spacing
        scale_fallback = 1.02
        fallback_centers = []
        for q, r in axial_coords:
            x = 1.5 * scale_fallback * q
            y = SQRT3 * scale_fallback * (r + q / 2.0)
            fallback_centers.append((x, y))

        fallback_inner = [[x, y, 0.0] for x, y in fallback_centers]
        final_center = (0.0, 0.0)
        final_angle = 0.0
        final_side = compute_required_side(fallback_inner, final_center, final_angle)
        final_inner = fallback_inner

    return final_inner, final_center, final_side, final_angle
