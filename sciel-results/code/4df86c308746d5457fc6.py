"""Rewritten program for packing 12 unit hexagons in a regular hexagon."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    def hexagon_vertices(cx, cy, side, angle_deg):
        """Return 6 vertices of a regular hexagon."""
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        """Check if point (px, py) is inside hexagon centered at (cx, cy)."""
        dx = px - cx
        dy = py - cy
        angle_rad = np.radians(angle_deg)
        cos_a = np.cos(-angle_rad)
        sin_a = np.sin(-angle_rad)
        local_x = dx * cos_a - dy * sin_a
        local_y = dx * sin_a + dy * cos_a

        apothem = side * np.sqrt(3) / 2
        normal_angles = np.radians(np.arange(30, 360, 60))

        for na in normal_angles:
            nx = np.cos(na)
            ny = np.sin(na)
            dist = local_x * nx + local_y * ny
            if dist > apothem + 1e-9:
                return False
        return True

    def polygons_disjoint_sat(v1, v2):
        """Check if two convex polygons are disjoint using SAT."""
        axes = []
        for i in range(len(v1)):
            p1 = v1[i]
            p2 = v1[(i + 1) % len(v1)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                axes.append(normal / norm)

        for i in range(len(v2)):
            p1 = v2[i]
            p2 = v2[(i + 1) % len(v2)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                axes.append(normal / norm)

        for axis in axes:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)

            if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
                return True
        return False

    def all_inside_outer(inner_positions, outer_center, outer_side, outer_angle):
        """Check all inner hexagons are inside the outer hexagon."""
        for i in range(len(inner_positions)):
            cx, cy, angle = inner_positions[i]
            verts = hexagon_vertices(cx, cy, 1.0, angle)
            for vx, vy in verts:
                if not point_in_hexagon(vx, vy, outer_center[0], outer_center[1], outer_side, outer_angle):
                    return False
        return True

    def pairwise_disjoint(inner_positions):
        """Check all pairs of inner hexagons are disjoint."""
        for i in range(len(inner_positions)):
            for j in range(i + 1, len(inner_positions)):
                ci = (inner_positions[i][0], inner_positions[i][1])
                cj = (inner_positions[j][0], inner_positions[j][1])
                ai = inner_positions[i][2]
                aj = inner_positions[j][2]
                v1 = hexagon_vertices(ci[0], ci[1], 1.0, ai)
                v2 = hexagon_vertices(cj[0], cj[1], 1.0, aj)
                if not polygons_disjoint_sat(v1, v2):
                    return False
        return True

    def is_valid(inner, oc, os, oa):
        """Full validity check."""
        return all_inside_outer(inner, oc, os, oa) and pairwise_disjoint(inner)

    def make_params(inner, oc, os, oa):
        """Convert to flat parameter array."""
        params = []
        for pos in inner:
            params.extend([pos[0], pos[1], pos[2]])
        params.extend([oc[0], oc[1], os, oa])
        return np.array(params)

    def parse_params(params):
        """Convert flat parameter array back to structured form."""
        inner = []
        for i in range(N):
            inner.append((params[3 * i], params[3 * i + 1], params[3 * i + 2]))
        oc = (params[3 * N], params[3 * N + 1])
        os = params[3 * N + 2]
        oa = params[3 * N + 3]
        return inner, oc, os, oa

    def objective(params):
        """Objective function: minimize outer side length with penalties."""
        n_inner = N
        inner_positions = []
        for i in range(n_inner):
            x = params[3 * i]
            y = params[3 * i + 1]
            a = params[3 * i + 2]
            inner_positions.append((x, y, a))

        ox = params[3 * n_inner]
        oy = params[3 * n_inner + 1]
        os = params[3 * n_inner + 2]
        oa = params[3 * n_inner + 3]

        outer_center = (ox, oy)
        outer_side = max(os, 0.5)
        outer_angle = oa

        inside_ok = all_inside_outer(inner_positions, outer_center, outer_side, outer_angle)
        disjoint_ok = pairwise_disjoint(inner_positions)

        if inside_ok and disjoint_ok:
            return outer_side

        penalty = 0.0
        if not inside_ok:
            for i in range(n_inner):
                cx, cy, angle = inner_positions[i]
                verts = hexagon_vertices(cx, cy, 1.0, angle)
                for vx, vy in verts:
                    if not point_in_hexagon(vx, vy, outer_center[0], outer_center[1], outer_side, outer_angle):
                        penalty += 1.0
        if not disjoint_ok:
            for i in range(n_inner):
                for j in range(i + 1, n_inner):
                    ci = (inner_positions[i][0], inner_positions[i][1])
                    cj = (inner_positions[j][0], inner_positions[j][1])
                    ai = inner_positions[i][2]
                    aj = inner_positions[j][2]
                    v1 = hexagon_vertices(ci[0], ci[1], 1.0, ai)
                    v2 = hexagon_vertices(cj[0], cj[1], 1.0, aj)
                    if not polygons_disjoint_sat(v1, v2):
                        penalty += 1.0

        return outer_side + penalty * 50.0

    def min_outer_side_for_config(inner_positions, outer_center, outer_angle):
        """Find minimum outer side length that contains all inner hexagons."""
        lo = 1.0
        hi = 15.0
        for _ in range(40):
            mid = (lo + hi) / 2
            if all_inside_outer(inner_positions, outer_center, mid, outer_angle):
                hi = mid
            else:
                lo = mid
        return hi

    # Generate diverse initial configurations with various rotations
    def get_initial_configurations():
        configs = []
        sqrt3 = np.sqrt(3)

        # For each base arrangement, try several outer angles
        outer_angles = [0.0, 30.0, 15.0, 45.0, 60.0]

        # Base arrangement A: 4x3 grid with hexagonal offset (aligned hexagons)
        for outer_angle in outer_angles:
            positions_a = []
            inner_angle = outer_angle  # align inner with outer
            for row in range(3):
                for col in range(4):
                    x = (col - 1.5) * 3.0
                    y = (row - 1) * 2.598
                    if row % 2 == 1:
                        x += 1.5
                    positions_a.append([x, y, inner_angle])
            configs.append((positions_a, outer_angle))

        # Base arrangement B: Hexagonal ring packing
        for outer_angle in outer_angles:
            positions_b = [[0.0, 0.0, outer_angle]]
            for k in range(6):
                angle = k * np.pi / 3
                x = 2.0 * np.cos(angle)
                y = 2.0 * np.sin(angle)
                positions_b.append([x, y, outer_angle])
            for k in range(5):
                angle = k * np.pi / 3 + np.pi / 6
                x = 3.0 * np.cos(angle)
                y = 3.0 * np.sin(angle)
                positions_b.append([x, y, outer_angle])
            configs.append((positions_b[:12], outer_angle))

        # Base arrangement C: Two rows of 6
        for outer_angle in [0.0, 30.0]:
            positions_c = []
            for i in range(6):
                positions_c.append([(i - 2.5) * 2.0, -1.5, outer_angle])
                positions_c.append([(i - 2.5) * 2.0 + 1.0, 1.5, outer_angle])
            configs.append((positions_c, outer_angle))

        # Base arrangement D: 3x4 grid with tighter spacing
        for outer_angle in [0.0, 30.0]:
            positions_d = []
            for row in range(3):
                for col in range(4):
                    x = (col - 1.5) * 2.1
                    y = (row - 1) * 2.6
                    if row % 2 == 1:
                        x += 1.05
                    positions_d.append([x, y, outer_angle])
            configs.append((positions_d, outer_angle))

        return configs

    best_params = None
    best_score = float('inf')

    configs = get_initial_configurations()

    # Sort configs by initial outer side length (tighter first)
    def config_initial_score(config_tuple):
        positions, outer_angle = config_tuple
        inner_tuples = [(p[0], p[1], p[2]) for p in positions]
        if not pairwise_disjoint(inner_tuples):
            return 100.0
        os = min_outer_side_for_config(inner_tuples, [0.0, 0.0], outer_angle)
        return os

    # Evaluate and sort
    scored_configs = []
    for idx, ct in enumerate(configs):
        if time.time() > deadline - 30:
            break
        s = config_initial_score(ct)
        scored_configs.append((s, idx, ct))
    scored_configs.sort(key=lambda x: x[0])

    # Try top configurations
    for score, idx, config_tuple in scored_configs:
        if time.time() > deadline - 15:
            break
        if score >= best_score - 0.1:
            break  # not promising

        positions, outer_angle = config_tuple
        inner_tuples = [(p[0], p[1], p[2]) for p in positions]

        # Check disjoint
        if not pairwise_disjoint(inner_tuples):
            # Try small perturbations
            np.random.seed(idx * 137)
            found = False
            for attempt in range(20):
                noise = np.random.uniform(-0.3, 0.3, size=2 * N)
                perturbed = [[p[0] + noise[2*i], p[1] + noise[2*i+1], p[2]] for i, p in enumerate(positions)]
                inner_tuples_p = [(p[0], p[1], p[2]) for p in perturbed]
                if pairwise_disjoint(inner_tuples_p):
                    inner_tuples = inner_tuples_p
                    positions = perturbed
                    found = True
                    break
            if not found:
                continue

        # Find tight outer side
        outer_side = min_outer_side_for_config(inner_tuples, [0.0, 0.0], outer_angle)
        outer_center = [0.0, 0.0]

        params = make_params(positions, outer_center, outer_side, outer_angle)

        # Run Nelder-Mead optimization
        try:
            res = minimize(objective, params, method='Nelder-Mead',
                           options={'maxiter': 800, 'xatol': 0.003, 'fatol': 0.003, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

        # Refinement if good
        if best_score < 5.5 and time.time() < deadline - 10:
            try:
                res2 = minimize(objective, best_params, method='Nelder-Mead',
                                options={'maxiter': 400, 'xatol': 0.001, 'fatol': 0.001, 'adaptive': True})
                if res2.fun < best_score:
                    best_score = res2.fun
                    best_params = res2.x.copy()
            except Exception:
                pass

    # Fallback
    if best_params is None:
        positions_a = []
        for row in range(3):
            for col in range(4):
                x = (col - 1.5) * 3.0
                y = (row - 1) * 2.598
                if row % 2 == 1:
                    x += 1.5
                positions_a.append([x, y, 0.0])
        inner_tuples = [(p[0], p[1], p[2]) for p in positions_a]
        outer_center = [0.0, 0.0]
        outer_angle = 30.0
        outer_side = 8.0
        while not all_inside_outer(inner_tuples, outer_center, outer_side, outer_angle):
            outer_side += 0.5
        best_params = make_params(positions_a, outer_center, outer_side, outer_angle)
        best_score = outer_side

    # Final validation and return
    inner, oc, os, oa = parse_params(best_params)

    if not is_valid(inner, oc, os, oa):
        while not all_inside_outer(inner, oc, os, oa):
            os += 0.05
            if os > 20:
                break

    return inner, oc, os, oa
# EVOLVE-BLOCK-END
