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
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))

    max_projection = 0.0
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))

    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                if proj > max_projection:
                    max_projection = proj

    return 2.0 * max_projection / SQRT3


def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=50.0):
    """
    Smooth surrogate for compute_required_side using log-sum-exp.
    """
    if not inner:
        return 0.0

    ox, oy = outer_center
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))

    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))

    # Collect all projections
    projs = []
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                projs.append(proj)

    if not projs:
        return 0.0

    projs = np.array(projs)
    # Soft-max via log-sum-exp
    m = np.max(projs)
    if m < 0:
        m = 0.0
    # log-sum-exp of projs, then divide by beta for smoothness
    # We want: (1/beta) * log(sum(exp(beta * p))) ≈ max(p) when beta is large
    shifted = beta * projs
    shifted_max = np.max(shifted)
    lse = shifted_max / beta + math.log(np.sum(np.exp(shifted - shifted_max)) / len(projs)) / beta
    # The soft-max gives us a smooth approximation to max_projection
    # But we need max(0, ...) since side length must be positive
    lse = max(0.0, lse)
    return 2.0 * lse / SQRT3


def sat_overlap_check(v1, v2, eps=1e-7):
    """
    Check if two convex polygons (defined by their vertices) overlap using SAT.
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
    verts_cache = []
    for i in range(n):
        x, y, ang = inner[i]
        verts = get_hex_vertices(x, y, 1.0, math.radians(ang))
        verts_cache.append(verts)
    for i in range(n):
        for j in range(i + 1, n):
            if sat_overlap_check(verts_cache[i], verts_cache[j], eps):
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
    deadline = start_time + 135.0  # safety margin before 150s kill

    # Define multiple candidate configurations
    # Config 1: Standard hexagonal lattice
    axial_coords_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (-1, 2), (-2, 1)
    ]

    # Config 2: Alternative arrangement with some rotations
    axial_coords_2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, -1), (1, 2), (-2, 2), (-1, -1)
    ]

    # Config 3: More compact arrangement
    axial_coords_3 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (0, 2), (-2, 2), (1, -2)
    ]

    def axial_to_cartesian(axial_coords, scale=1.001):
        centers = []
        for q, r in axial_coords:
            x = 1.5 * scale * q
            y = SQRT3 * scale * (r + q / 2.0)
            centers.append((x, y))
        return centers

    # Generate initial configurations
    initial_configs = []
    for coords in [axial_coords_1, axial_coords_2, axial_coords_3]:
        if len(coords) > N:
            coords = coords[:N]
        centers = axial_to_cartesian(coords)
        config = [[x, y, 0.0] for x, y in centers]
        if check_all_disjoint(config):
            initial_configs.append(config)

    # Fallback if no valid configs
    if not initial_configs:
        initial_configs = [[[i * 2.5, 0.0, 0.0] for i in range(N)]]

    # Use the first valid config as primary initial
    initial_inner = initial_configs[0]

    best_side = float('inf')
    best_inner = [list(x) for x in initial_inner]
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0

    # --- Phase 1: For each initial config, optimize outer hexagon ---
    def objective_outer(params, inner_list):
        cx, cy, angle_deg = params
        side = compute_required_side(inner_list, (cx, cy), angle_deg)
        return side

    best_config_inner = None
    best_config_center = None
    best_config_angle = None

    for config_idx, initial_inner in enumerate(initial_configs):
        if time.time() > deadline - 60:
            break

        cx_avg = sum(c[0] for c in [tuple(x[:2]) for x in initial_inner]) / len(initial_inner)
        cy_avg = sum(c[1] for c in [tuple(x[:2]) for x in initial_inner]) / len(initial_inner)

        init_angles = [0, 15, 30, 45, 60, 75, 90, 120, 150, 180]
        best_outer_val = float('inf')
        best_outer_params = [cx_avg, cy_avg, 0.0]

        for init_angle in init_angles:
            if time.time() > deadline - 60:
                break
            init_params = [cx_avg, cy_avg, float(init_angle)]
            try:
                res = minimize(
                    lambda p: objective_outer(p, initial_inner),
                    init_params,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 400}
                )
                if res.fun < best_outer_val:
                    best_outer_val = res.fun
                    best_outer_params = res.x
            except Exception:
                pass

        if best_outer_val < best_side:
            best_side = best_outer_val
            best_inner = [list(x) for x in initial_inner]
            best_outer_center = (best_outer_params[0], best_outer_params[1])
            best_outer_angle = best_outer_params[2]
            best_config_inner = [list(x) for x in initial_inner]
            best_config_center = (best_outer_params[0], best_outer_params[1])
            best_config_angle = best_outer_params[2]

    # --- Phase 2: Joint Optimization with Smooth Objective ---
    # Variables: [x1,y1,a1, ..., xN,yN,aN, cx, cy, angle]
    # Total: 3*N + 3 = 36 variables

    if best_config_inner is None:
        best_config_inner = [list(x) for x in initial_inner]
        best_config_center = (0.0, 0.0)
        best_config_angle = 0.0
        best_side = compute_required_side(best_config_inner, best_config_center, best_config_angle)

    def objective_joint_smooth(params, beta=30.0):
        inner_list = []
        for i in range(N):
            x = params[i * 3]
            y = params[i * 3 + 1]
            a = params[i * 3 + 2]
            inner_list.append([x, y, a])

        cx = params[N * 3]
        cy = params[N * 3 + 1]
        angle = params[N * 3 + 2]

        # Smooth container side
        side = compute_required_side_smooth(inner_list, (cx, cy), angle, beta=beta)

        # Penalty for overlaps: use center distance as proxy
        penalty = 0.0
        min_dist = SQRT3 - 0.01
        for i in range(N):
            for j in range(i + 1, N):
                xi, yi, _ = inner_list[i]
                xj, yj, _ = inner_list[j]
                dist = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2 * 100.0

        # Soft boundary violation penalty
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

    def objective_joint_exact(params):
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

        penalty = 0.0
        min_dist = SQRT3 - 0.01
        for i in range(N):
            for j in range(i + 1, N):
                xi, yi, _ = inner_list[i]
                xj, yj, _ = inner_list[j]
                dist = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2 * 100.0

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

    # Initial joint params from best config
    init_params = []
    for x, y, a in best_config_inner:
        init_params.extend([x, y, a])
    init_params.extend([best_config_center[0], best_config_center[1], best_config_angle])

    best_joint_side = best_side
    best_joint_inner = [list(x) for x in best_config_inner]
    best_joint_center = best_config_center
    best_joint_angle = best_config_angle
    best_joint_params = list(init_params)

    # --- Phase 2a: Smooth optimization with multiple restarts from different configs ---
    num_restarts_smooth = 8
    for restart in range(num_restarts_smooth):
        if time.time() > deadline - 20:
            break

        # Use different initial configs for different restarts
        config_idx = restart % len(initial_configs)
        current_params = []
        for x, y, a in initial_configs[config_idx]:
            current_params.extend([x, y, a])
        
        # Add outer params (use best known or default)
        if best_config_center is not None:
            current_params.extend([best_config_center[0], best_config_center[1], best_config_angle])
        else:
            cx_avg = sum(p[0] for p in current_params[0:N*3:3]) / N
            cy_avg = sum(p[1] for p in current_params[1:N*3:3]) / N
            current_params.extend([cx_avg, cy_avg, 0.0])

        if restart > 0:
            np.random.seed(42 + restart * 7)
            perturb_scale = 0.03 + restart * 0.015
            for i in range(len(current_params)):
                if i < N * 3:
                    current_params[i] += np.random.uniform(-perturb_scale, perturb_scale)
                elif i == N * 3 + 2:
                    current_params[i] += np.random.uniform(-10, 10)

        try:
            res = minimize(
                lambda p: objective_joint_smooth(p, beta=30.0),
                current_params,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-9, 'maxfev': 1000}
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
                    best_joint_params = list(res.x)
        except Exception:
            pass

    # --- Phase 2b: Exact objective refinement from smooth solution ---
    if time.time() < deadline - 15:
        # Try multiple exact refinements from different starting points
        for refine_attempt in range(3):
            if time.time() > deadline - 15:
                break
            try:
                start_params = list(best_joint_params)
                if refine_attempt > 0:
                    np.random.seed(100 + refine_attempt * 11)
                    for i in range(len(start_params)):
                        start_params[i] += np.random.uniform(-0.01, 0.01)
                
                res = minimize(
                    lambda p: objective_joint_exact(p),
                    start_params,
                    method='Nelder-Mead',
                    options={'maxiter': 600, 'xatol': 1e-9, 'fatol': 1e-11, 'maxfev': 1200}
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
                        best_joint_params = list(res.x)
            except Exception:
                pass

    # --- Phase 3: High-precision polish pass ---
    if time.time() < deadline - 5:
        try:
            res = minimize(
                lambda p: objective_joint_exact(p),
                best_joint_params,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-11, 'fatol': 1e-13, 'maxfev': 2500}
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

    # --- Phase 4: If joint didn't improve, fall back to outer-optimized result ---
    if best_joint_side >= best_side:
        best_joint_inner = [list(x) for x in best_config_inner]
        best_joint_side = best_side
        best_joint_center = best_config_center
        best_joint_angle = best_config_angle

    # --- Final verification and fallback ---
    final_inner = best_joint_inner
    final_center = best_joint_center
    final_side = best_joint_side
    final_angle = best_joint_angle

    if not check_all_disjoint(final_inner) or not check_inside_outer(final_inner, final_center, final_side, final_angle):
        # Try each initial config as fallback
        found_valid = False
        for config in initial_configs:
            config_center = (0.0, 0.0)
            config_angle = 0.0
            config_side = compute_required_side(config, config_center, config_angle)
            if check_all_disjoint(config) and check_inside_outer(config, config_center, config_side, config_angle):
                final_inner = [list(x) for x in config]
                final_center = config_center
                final_side = config_side
                final_angle = config_angle
                found_valid = True
                break
        
        if not found_valid:
            # Last resort: spread out hexagons
            final_inner = [[i * 3.0, 0.0, 0.0] for i in range(N)]
            final_center = (0.0, 0.0)
            final_angle = 0.0
            final_side = compute_required_side(final_inner, final_center, final_angle)

    return final_inner, final_center, final_side, final_angle
