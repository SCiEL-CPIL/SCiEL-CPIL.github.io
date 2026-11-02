"""Packing 12 unit hexagons in a regular hexagon using smooth caging + SA."""
import numpy as np
import time
import math

N = 12
SQRT3 = math.sqrt(3.0)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        dx = px - cx
        dy = py - cy
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)
        local_x = dx * cos_a - dy * sin_a
        local_y = dx * sin_a + dy * cos_a
        apothem = side * SQRT3 / 2
        normal_angles = np.radians(np.arange(30, 360, 60))
        for na in normal_angles:
            nx = math.cos(na)
            ny = math.sin(na)
            dist = local_x * nx + local_y * ny
            if dist > apothem + 1e-9:
                return False
        return True

    def distance_to_hexagon_edge(px, py, cx, cy, side, angle_deg):
        dx = px - cx
        dy = py - cy
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)
        local_x = dx * cos_a - dy * sin_a
        local_y = dx * sin_a + dy * cos_a
        apothem = side * SQRT3 / 2
        normal_angles = np.radians(np.arange(30, 360, 60))
        min_dist = float('inf')
        for na in normal_angles:
            nx = math.cos(na)
            ny = math.sin(na)
            dist = local_x * nx + local_y * ny
            min_dist = min(min_dist, apothem - dist)
        return min_dist

    def polygons_disjoint_sat(v1, v2):
        axes = []
        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i + 1) % 6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                axes.append(normal / norm)
        for i in range(6):
            p1 = v2[i]
            p2 = v2[(i + 1) % 6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                axes.append(normal / norm)
        for axis in axes:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            if np.max(proj1) < np.min(proj2) - 1e-9 or np.max(proj2) < np.min(proj1) - 1e-9:
                return True
        return False

    def all_inside_outer(inner_positions, outer_center, outer_side, outer_angle):
        for i in range(N):
            cx, cy, angle = inner_positions[i]
            verts = hexagon_vertices(cx, cy, 1.0, angle)
            for vx, vy in verts:
                if not point_in_hexagon(vx, vy, outer_center[0], outer_center[1], outer_side, outer_angle):
                    return False
        return True

    def pairwise_disjoint(inner_positions):
        for i in range(N):
            for j in range(i + 1, N):
                v1 = hexagon_vertices(inner_positions[i][0], inner_positions[i][1], 1.0, inner_positions[i][2])
                v2 = hexagon_vertices(inner_positions[j][0], inner_positions[j][1], 1.0, inner_positions[j][2])
                if not polygons_disjoint_sat(v1, v2):
                    return False
        return True

    def is_valid(inner, oc, os, oa):
        return all_inside_outer(inner, oc, os, oa) and pairwise_disjoint(inner)

    def min_outer_side_for_config(inner_positions, outer_center, outer_angle):
        lo = 1.0
        hi = 20.0
        for _ in range(50):
            mid = (lo + hi) / 2
            if all_inside_outer(inner_positions, outer_center, mid, outer_angle):
                hi = mid
            else:
                lo = mid
        return hi

    def smooth_penalty(params):
        n_inner = N
        inner_positions = []
        for i in range(n_inner):
            inner_positions.append((params[3 * i], params[3 * i + 1], params[3 * i + 2]))

        ox = params[3 * n_inner]
        oy = params[3 * n_inner + 1]
        os = max(params[3 * n_inner + 2], 0.5)
        oa = params[3 * n_inner + 3]

        outer_center = (ox, oy)
        outer_side = os
        outer_angle = oa

        penalty = 0.0

        # Containment penalty: each inner vertex should be inside outer
        for i in range(n_inner):
            cx, cy, angle = inner_positions[i]
            verts = hexagon_vertices(cx, cy, 1.0, angle)
            for vx, vy in verts:
                d = distance_to_hexagon_edge(vx, vy, ox, oy, outer_side, outer_angle)
                if d < 0:
                    penalty += d * d

        # Separation penalty: SAT penetration depth approximation
        for i in range(n_inner):
            for j in range(i + 1, n_inner):
                ci = inner_positions[i]
                cj = inner_positions[j]
                v1 = hexagon_vertices(ci[0], ci[1], 1.0, ci[2])
                v2 = hexagon_vertices(cj[0], cj[1], 1.0, cj[2])

                # Quick distance check
                cx1, cy1 = ci[0], ci[1]
                cx2, cy2 = cj[0], cj[1]
                dist_centers = math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
                if dist_centers > 4.0:
                    continue

                # SAT check for penetration
                axes = []
                for k in range(6):
                    p1 = v1[k]
                    p2 = v1[(k + 1) % 6]
                    edge = p2 - p1
                    normal = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(normal)
                    if norm > 1e-9:
                        axes.append(normal / norm)
                for k in range(6):
                    p1 = v2[k]
                    p2 = v2[(k + 1) % 6]
                    edge = p2 - p1
                    normal = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(normal)
                    if norm > 1e-9:
                        axes.append(normal / norm)

                min_overlap = float('inf')
                for axis in axes:
                    proj1 = np.dot(v1, axis)
                    proj2 = np.dot(v2, axis)
                    min1, max1 = np.min(proj1), np.max(proj1)
                    min2, max2 = np.min(proj2), np.max(proj2)
                    overlap = min(max1, max2) - max(min1, min2)
                    if overlap > 0:
                        min_overlap = min(min_overlap, overlap)

                if min_overlap < float('inf') and min_overlap > 0:
                    penalty += min_overlap * min_overlap

        return outer_side + penalty * 10.0

    def get_initial_configurations():
        configs = []
        sqrt3 = SQRT3

        # Arrangement A: 4x3 grid with hexagonal offset
        positions_a = []
        for row in range(3):
            for col in range(4):
                x = (col - 1.5) * 3.0
                y = (row - 1) * 2.598
                if row % 2 == 1:
                    x += 1.5
                positions_a.append([x, y, 0.0])
        configs.append(positions_a)

        # Arrangement B: More compact hexagonal packing
        positions_b = [[0.0, 0.0, 0.0]]
        for k in range(6):
            angle = k * math.pi / 3
            x = 2.0 * math.cos(angle)
            y = 2.0 * math.sin(angle)
            positions_b.append([x, y, 0.0])
        for k in range(6):
            angle = k * math.pi / 3 + math.pi / 6
            x = 3.0 * math.cos(angle)
            y = 3.0 * math.sin(angle)
            positions_b.append([x, y, 0.0])
        configs.append(positions_b[:12])

        # Arrangement C: Two rows of 6
        positions_c = []
        for i in range(6):
            positions_c.append([(i - 2.5) * 2.0, -1.0, 0.0])
            positions_c.append([(i - 2.5) * 2.0 + 1.0, 1.0, 0.0])
        configs.append(positions_c)

        # Arrangement D: 3 rows of 4 with hexagonal offset
        positions_d = []
        for row in range(3):
            for col in range(4):
                x = (col - 1.5) * 2.0
                y = (row - 1) * 2.598
                if row % 2 == 1:
                    x += 1.0
                positions_d.append([x, y, 0.0])
        configs.append(positions_d)

        return configs

    def simulated_annealing(params, max_iter=5000):
        current_params = params.copy()
        current_score = smooth_penalty(current_params)
        best_params = current_params.copy()
        best_score = current_score

        T = 1.0
        cooling = 0.9995

        np.random.seed(42)

        for it in range(max_iter):
            if time.time() > deadline - 5:
                break

            # Generate perturbation
            new_params = current_params.copy()
            step_size = T * 0.5
            noise = np.random.normal(0, step_size, size=len(new_params))
            new_params += noise

            # Keep outer side positive
            new_params[3 * N + 2] = max(new_params[3 * N + 2], 0.5)

            new_score = smooth_penalty(new_params)

            if new_score < current_score or np.random.random() < math.exp((current_score - new_score) / T):
                current_params = new_params
                current_score = new_score

                if current_score < best_score:
                    best_score = current_score
                    best_params = current_params.copy()

            T *= cooling

            if T < 0.001:
                break

        return best_params, best_score

    # Main optimization loop
    best_params = None
    best_score = float('inf')

    configs = get_initial_configurations()

    for config_idx, config in enumerate(configs):
        if time.time() > deadline - 20:
            break

        outer_center = [0.0, 0.0]
        outer_angle = 30.0

        inner_tuples = [(p[0], p[1], p[2]) for p in config]
        outer_side = min_outer_side_for_config(inner_tuples, outer_center, outer_angle)

        params = np.array(config).flatten()
        params = np.append(params, [outer_center[0], outer_center[1], outer_side, outer_angle])

        # Run simulated annealing
        try:
            sa_params, sa_score = simulated_annealing(params, max_iter=3000)
            if sa_score < best_score:
                best_score = sa_score
                best_params = sa_params.copy()
        except Exception:
            pass

        # If we have a good result, do a refinement with smaller steps
        if best_score < 5.0 and time.time() < deadline - 10:
            try:
                refine_params, refine_score = simulated_annealing(best_params, max_iter=2000)
                if refine_score < best_score:
                    best_score = refine_score
                    best_params = refine_params.copy()
            except Exception:
                pass

    # If no valid result found, use a fallback
    if best_params is None:
        config = configs[0]
        inner_tuples = [(p[0], p[1], p[2]) for p in config]
        outer_center = [0.0, 0.0]
        outer_angle = 30.0
        outer_side = 8.0
        while not all_inside_outer(inner_tuples, outer_center, outer_side, outer_angle):
            outer_side += 0.5
        best_params = np.array(config).flatten()
        best_params = np.append(best_params, [outer_center[0], outer_center[1], outer_side, outer_angle])
        best_score = outer_side

    # Final validation and return
    inner = []
    for i in range(N):
        inner.append((best_params[3 * i], best_params[3 * i + 1], best_params[3 * i + 2]))
    oc = (best_params[3 * N], best_params[3 * N + 1])
    os = best_params[3 * N + 2]
    oa = best_params[3 * N + 3]

    if not is_valid(inner, oc, os, oa):
        # Fallback: expand outer hexagon
        while not all_inside_outer(inner, oc, os, oa):
            os += 0.1
            if os > 20:
                break

    return inner, oc, os, oa
