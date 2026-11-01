"""Packing unit hexagons inside a regular hexagon."""
import math
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar, linprog

N = 11
SQRT3 = math.sqrt(3)
APOTHEM = SQRT3 / 2.0
DEADLINE = 145.0  # safety margin before 150s kill


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


def logsumexp(a, b=1.0):
    """Stable log-sum-exp for array a with temperature b."""
    a = np.asarray(a, dtype=np.float64)
    amax = np.max(a)
    if not np.isfinite(amax):
        return amax
    return amax + math.log(np.sum(np.exp((a - amax) / b))) * b


def axial_to_cartesian(axial_list):
    """Convert axial coordinates to Cartesian for unit hexagons."""
    pts = []
    for q, r in axial_list:
        x = SQRT3 * (q + r / 2.0)
        y = 1.5 * r
        pts.append((x, y))
    return pts


def construct():
    start_time = time.time()

    # Define multiple candidate layouts in axial coordinates
    candidates = []

    # Layout 1: Center + Ring 1 + partial Ring 2 (11 hexagons)
    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2)
    ]
    candidates.append(axial_1)

    # Layout 2: Center + Ring 1 + different partial Ring 2
    axial_2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_2)

    # Layout 3: 3-4-3-1 arrangement
    axial_3 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1)
    ]
    candidates.append(axial_3)

    # Layout 4: 4-3-4 arrangement
    axial_4 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    candidates.append(axial_4)

    # Layout 5: 2-3-3-3 arrangement
    axial_5 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_5)

    # Layout 6: 3-3-3-2 arrangement
    axial_6 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_6)

    # Layout 7: Center + Ring 1 + cross pattern
    axial_7 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_7)

    # Layout 8: 5-6 arrangement
    axial_8 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)
    ]
    candidates.append(axial_8)

    # Layout 9: Center + Ring 1 + diagonal pattern
    axial_9 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (-2, 2), (-1, 3), (1, -2)
    ]
    candidates.append(axial_9)

    # Layout 10: 3-3-3-2 compact
    axial_10 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_10)

    # Layout 11: 2-4-4-1 arrangement
    axial_11 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3)
    ]
    candidates.append(axial_11)

    # Layout 12: Staircase pattern
    axial_12 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3),
        (0, 4)
    ]
    candidates.append(axial_12)

    # Layout 13: Center + Ring 1 + asymmetric Ring 2
    axial_13 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, 2)
    ]
    candidates.append(axial_13)

    # Layout 14: Center + Ring 1 + opposite corners
    axial_14 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, -1)
    ]
    candidates.append(axial_14)

    # Layout 15: 3-2-3-3 arrangement
    axial_15 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1), (0, -1)
    ]
    candidates.append(axial_15)

    # Layout 16: 2-3-2-4 arrangement
    axial_16 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 2)
    ]
    candidates.append(axial_16)

    # Layout 17: Center + Ring 1 + top cluster
    axial_17 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (1, 2), (-1, 2), (0, 3)
    ]
    candidates.append(axial_17)

    # Layout 18: 4-3-4 arrangement variant
    axial_18 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_18)

    # Layout 19: 2-4-3-2 arrangement
    axial_19 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 3)
    ]
    candidates.append(axial_19)

    # Layout 20: Center + Ring 1 + symmetric corners
    axial_20 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, 1)
    ]
    candidates.append(axial_20)

    best_side = float('inf')
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0

    # Process each candidate layout
    for idx, axial_coords in enumerate(candidates):
        if time.time() - start_time > DEADLINE - 20:
            break

        # Convert to Cartesian
        centers = axial_to_cartesian(axial_coords)

        # Initial layout: All hexagons aligned (angle 0)
        initial_inner = [[x, y, 0.0] for x, y in centers]

        # Centroid
        cx_avg = sum(c[0] for c in centers) / len(centers)
        cy_avg = sum(c[1] for c in centers) / len(centers)

        # Try multiple diverse initial angles to avoid local minima
        best_result = None
        best_val = float('inf')

        init_angles = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]

        for init_angle in init_angles:
            if time.time() - start_time > DEADLINE - 15:
                break

            init_params = [cx_avg, cy_avg, float(init_angle)]

            try:
                # Use BFGS with the smooth objective for better convergence
                def objective_outer(p, temp=0.01):
                    ox, oy = p[0], p[1]
                    angle = p[2]
                    ox, oy = p[0], p[1]
                    normals = []
                    for i in range(6):
                        theta = math.radians(30 + i * 60)
                        normals.append((math.cos(theta), math.sin(theta)))

                    cos_a = math.cos(math.radians(angle))
                    sin_a = math.sin(math.radians(angle))

                    all_projections = []
                    for (ix, iy, iangle) in initial_inner:
                        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
                        for v in verts:
                            dx = v[0] - ox
                            dy = v[1] - oy
                            lx = dx * cos_a + dy * sin_a
                            ly = -dx * sin_a + dy * cos_a

                            for nx, ny in normals:
                                proj = lx * nx + ly * ny
                                all_projections.append(proj)

                    if len(all_projections) == 0:
                        return 0.0

                    proj_array = np.array(all_projections, dtype=np.float64)
                    smooth_max = logsumexp(proj_array, temp)

                    return 2.0 * smooth_max / SQRT3

                res = minimize(
                    objective_outer,
                    init_params,
                    method='BFGS',
                    options={'maxiter': 200, 'gtol': 1e-8}
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_result = res.x
            except Exception:
                pass

        if best_result is not None:
            current_outer_center = (best_result[0], best_result[1])
            current_outer_angle = best_result[2]
            current_side = compute_required_side(initial_inner, current_outer_center, current_outer_angle)
        else:
            current_outer_center = (cx_avg, cy_avg)
            current_outer_angle = 0.0
            current_side = compute_required_side(initial_inner, current_outer_center, current_outer_angle)

        # High-precision polish pass
        if time.time() - start_time < DEADLINE - 10:
            try:
                polish_res = minimize(
                    lambda p: compute_required_side(initial_inner, (p[0], p[1]), p[2]),
                    [current_outer_center[0], current_outer_center[1], current_outer_angle],
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 2000}
                )
                if polish_res.fun < current_side:
                    current_outer_center = (polish_res.x[0], polish_res.x[1])
                    current_outer_angle = polish_res.x[2]
                    current_side = polish_res.fun
            except Exception:
                pass

        # Joint optimization with smooth surrogate
        def objective_joint_smooth(params, temperature=0.01):
            inner_list = []
            for i in range(N):
                x = params[i * 3]
                y = params[i * 3 + 1]
                a = params[i * 3 + 2]
                inner_list.append([x, y, a])

            cx = params[N * 3]
            cy = params[N * 3 + 1]
            angle = params[N * 3 + 2]

            ox, oy = cx, cy
            normals = []
            for i in range(6):
                theta = math.radians(30 + i * 60)
                normals.append((math.cos(theta), math.sin(theta)))

            cos_a = math.cos(math.radians(angle))
            sin_a = math.sin(math.radians(angle))

            all_projections = []
            for (ix, iy, iangle) in inner_list:
                verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
                for v in verts:
                    dx = v[0] - ox
                    dy = v[1] - oy
                    lx = dx * cos_a + dy * sin_a
                    ly = -dx * sin_a + dy * cos_a

                    for nx, ny in normals:
                        proj = lx * nx + ly * ny
                        all_projections.append(proj)

            if len(all_projections) == 0:
                side = 0.0
            else:
                proj_array = np.array(all_projections, dtype=np.float64)
                smooth_max = logsumexp(proj_array, temperature)
                side = 2.0 * smooth_max / SQRT3

            # Penalty for overlaps
            penalty = 0.0
            min_dist = SQRT3 - 0.02
            for i in range(N):
                for j in range(i + 1, N):
                    xi, yi, _ = inner_list[i]
                    xj, yj, _ = inner_list[j]
                    dist = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                    if dist < min_dist:
                        penalty += (min_dist - dist) ** 2

            return side + 10.0 * penalty

        # Initial joint params
        init_params = []
        for x, y, a in initial_inner:
            init_params.extend([x, y, a])
        init_params.extend([current_outer_center[0], current_outer_center[1], current_outer_angle])

        # Try joint optimization with a few restarts
        best_joint_side = current_side
        best_joint_inner = [list(x) for x in initial_inner]
        best_joint_center = current_outer_center
        best_joint_angle = current_outer_angle

        for restart in range(3):
            if time.time() - start_time > DEADLINE - 10:
                break

            current_params = list(init_params)
            if restart > 0:
                np.random.seed(42 + idx * 100 + restart)
                for i in range(len(current_params)):
                    if i < N * 3:
                        current_params[i] += np.random.uniform(-0.05, 0.05)

            try:
                res = minimize(
                    lambda p: objective_joint_smooth(p, temperature=0.01),
                    current_params,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
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

        # Final polish on the best joint solution
        if best_joint_side < best_side and time.time() - start_time < DEADLINE - 5:
            try:
                polish_params = [best_joint_center[0], best_joint_center[1], best_joint_angle]
                polish_res = minimize(
                    lambda p: compute_required_side(best_joint_inner, (p[0], p[1]), p[2]),
                    polish_params,
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 3000}
                )
                if polish_res.fun < best_joint_side:
                    best_joint_center = (polish_res.x[0], polish_res.x[1])
                    best_joint_angle = polish_res.x[2]
                    best_joint_side = polish_res.fun
            except Exception:
                pass

        # Update global best
        if best_joint_side < best_side:
            best_side = best_joint_side
            best_inner = best_joint_inner
            best_outer_center = best_joint_center
            best_outer_angle = best_joint_angle

    # Final verification and fallback
    if best_inner is None or not check_all_disjoint(best_inner) or not check_inside_outer(best_inner, best_outer_center, best_side, best_outer_angle):
        # Fallback: use first layout with safe spacing
        scale_fallback = 1.05
        centers = axial_to_cartesian(candidates[0])
        fallback_inner = [[x * scale_fallback, y * scale_fallback, 0.0] for x, y in centers]
        best_outer_center = (0.0, 0.0)
        best_outer_angle = 0.0
        best_side = compute_required_side(fallback_inner, best_outer_center, best_outer_angle)
        best_inner = fallback_inner

    return best_inner, best_outer_center, best_side, best_outer_angle
