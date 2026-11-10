"""Pack 11 unit hexagons inside a regular hexagon, minimizing outer side length."""
import numpy as np
import math
import time
import random

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    sqrt3 = math.sqrt(3)

    def hexagon_vertices(cx, cy, side, angle_deg):
        """Return 6 vertices of regular hexagon with given center, side, rotation."""
        angle_rad = math.radians(angle_deg)
        verts = []
        for k in range(6):
            theta = angle_rad + k * math.pi / 3
            verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
        return verts

    def compute_min_side(inner_positions, outer_center, outer_angle):
        """
        Compute minimum outer side to contain all inner hexagons.
        """
        cx, cy = outer_center
        max_side = 0.0
        angle_rad = math.radians(outer_angle)
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)

        # Pre-compute the 6 outward normals of a unit hexagon
        normals = []
        for k in range(6):
            theta = math.radians(30 + k * 60)
            normals.append((math.cos(theta), math.sin(theta)))

        for ix, iy, iang in inner_positions:
            inner_verts = hexagon_vertices(ix, iy, 1.0, iang)
            for vx, vy in inner_verts:
                dx = vx - cx
                dy = vy - cy
                rx = dx * cos_a - dy * sin_a
                ry = dx * sin_a + dy * cos_a

                for nx, ny in normals:
                    proj = rx * nx + ry * ny
                    required_side = proj / (sqrt3 / 2)
                    if required_side > max_side:
                        max_side = required_side
        return max_side + 1e-10

    def hex_distance(p1, p2):
        """
        Exact minimum distance between two unit regular hexagons using SAT.
        Returns 0 if they overlap, positive if separated.
        """
        verts1 = hexagon_vertices(p1[0], p1[1], 1.0, p1[2])
        verts2 = hexagon_vertices(p2[0], p2[1], 1.0, p2[2])

        # Collect all edge normals (candidate separating axes)
        axes = []
        for i in range(6):
            for verts in (verts1, verts2):
                x1, y1 = verts[i]
                x2, y2 = verts[(i + 1) % 6]
                dx = x2 - x1
                dy = y2 - y1
                length = math.sqrt(dx * dx + dy * dy)
                if length > 1e-12:
                    axes.append((-dy / length, dx / length))

        # Also add the direction between centers as a candidate axis
        dx_c = p2[0] - p1[0]
        dy_c = p2[1] - p1[1]
        dist_c = math.sqrt(dx_c * dx_c + dy_c * dy_c)
        if dist_c > 1e-12:
            axes.append((dx_c / dist_c, dy_c / dist_c))

        # For each axis, project both polygons and check for separation
        for nx, ny in axes:
            min1 = min(vx * nx + vy * ny for vx, vy in verts1)
            max1 = max(vx * nx + vy * ny for vx, vy in verts1)
            min2 = min(vx * nx + vy * ny for vx, vy in verts2)
            max2 = max(vx * nx + vy * ny for vx, vy in verts2)

            gap = max(min2 - max1, min1 - max2)
            if gap > 0:
                return gap

        # No separating axis found -> they overlap
        return 0.0

    def is_disjoint(p1, p2, tol=1e-9):
        """Check if two hexagons are disjoint (no overlap)."""
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        d = math.sqrt(dx * dx + dy * dy)
        if d >= 2.0 - tol:
            return True
        return hex_distance(p1, p2) > tol

    def all_disjoint(positions, tol=1e-9):
        """Check if all hexagons are pairwise disjoint."""
        n = len(positions)
        for i in range(n):
            for j in range(i + 1, n):
                if not is_disjoint(positions[i], positions[j], tol):
                    return False
        return True

    def nelder_mead_polish(positions, outer_angle, max_iter=2000, tol=1e-10):
        """
        Nelder-Mead optimization to polish positions.
        """
        if time.time() > deadline - 10.0:
            return positions, compute_min_side(positions, (0.0, 0.0), outer_angle)

        # Flatten positions to a single vector
        def flatten(pos):
            return [v for p in pos for v in p]

        def unflatten(vec):
            return [[vec[3*i], vec[3*i+1], vec[3*i+2]] for i in range(N)]

        def objective(vec):
            pos = unflatten(vec)
            if not all_disjoint(pos):
                return 100.0 + compute_min_side(pos, (0.0, 0.0), outer_angle)
            return compute_min_side(pos, (0.0, 0.0), outer_angle)

        # Initialize simplex
        x0 = flatten(positions)
        simplex = [x0[:]]
        for i in range(3 * N):
            xi = x0[:]
            xi[i] += 0.1
            simplex.append(xi)

        # Sort simplex by objective value
        obj_vals = [objective(x) for x in simplex]
        order = sorted(range(len(simplex)), key=lambda i: obj_vals[i])
        simplex = [simplex[i] for i in order]
        obj_vals = [obj_vals[i] for i in order]

        best_x = simplex[0][:]
        best_f = obj_vals[0]

        alpha, gamma, rho = 1.0, 2.0, 0.5

        for it in range(max_iter):
            if time.time() > deadline - 10.0:
                break

            # Compute centroid of all but worst point
            centroid = [0.0] * (3 * N)
            for i in range(len(simplex) - 1):
                for j in range(3 * N):
                    centroid[j] += simplex[i][j]
            for j in range(3 * N):
                centroid[j] /= (len(simplex) - 1)

            # Reflection
            x_r = [centroid[j] + alpha * (centroid[j] - simplex[-1][j]) for j in range(3 * N)]
            f_r = objective(x_r)

            if f_r < obj_vals[0]:
                # Expansion
                x_e = [centroid[j] + gamma * (x_r[j] - centroid[j]) for j in range(3 * N)]
                f_e = objective(x_e)
                if f_e < f_r:
                    simplex[-1] = x_e
                    obj_vals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    obj_vals[-1] = f_r
            elif f_r < obj_vals[-2]:
                simplex[-1] = x_r
                obj_vals[-1] = f_r
            else:
                # Contraction
                x_c = [centroid[j] + rho * (simplex[-1][j] - centroid[j]) for j in range(3 * N)]
                f_c = objective(x_c)
                if f_c < obj_vals[-1]:
                    simplex[-1] = x_c
                    obj_vals[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = [simplex[0][j] + 0.5 * (simplex[i][j] - simplex[0][j]) for j in range(3 * N)]
                        obj_vals[i] = objective(simplex[i])

            # Sort
            order = sorted(range(len(simplex)), key=lambda i: obj_vals[i])
            simplex = [simplex[i] for i in order]
            obj_vals = [obj_vals[i] for i in order]

            if obj_vals[0] < best_f:
                best_f = obj_vals[0]
                best_x = simplex[0][:]

            # Check convergence
            if max(abs(obj_vals[i] - obj_vals[0]) for i in range(len(simplex))) < tol:
                break

        return unflatten(best_x), best_f

    def simulated_annealing(initial_positions, outer_angle=0.0, max_iter=50000, start_temp=2.0, end_temp=0.01):
        """
        Simulated annealing to optimize positions of hexagons.
        Returns best positions found.
        """
        current_positions = [list(p) for p in initial_positions]
        current_side = compute_min_side(current_positions, (0.0, 0.0), outer_angle)
        best_positions = [list(p) for p in current_positions]
        best_side = current_side

        temp = start_temp
        cooling = (end_temp / start_temp) ** (1.0 / max_iter)

        for it in range(max_iter):
            if time.time() > deadline - 15.0:
                break

            # Perturb a random hexagon
            idx = random.randint(0, N - 1)
            old_x, old_y, old_ang = current_positions[idx]

            # Small perturbation
            dx = random.gauss(0, 0.15)
            dy = random.gauss(0, 0.15)
            da = random.gauss(0, 5.0)

            new_x = old_x + dx
            new_y = old_y + dy
            new_ang = old_ang + da

            # Check if still disjoint from others
            candidate = [new_x, new_y, new_ang]
            disjoint = True
            for j in range(N):
                if j != idx and not is_disjoint(candidate, current_positions[j]):
                    disjoint = False
                    break

            if not disjoint:
                continue

            # Compute new side
            trial_positions = [list(p) for p in current_positions]
            trial_positions[idx] = candidate
            trial_side = compute_min_side(trial_positions, (0.0, 0.0), outer_angle)

            delta = trial_side - current_side
            if delta < 0 or random.random() < math.exp(-delta / max(temp, 1e-10)):
                current_positions[idx] = candidate
                current_side = trial_side
                if current_side < best_side:
                    best_side = current_side
                    best_positions = [list(p) for p in current_positions]

            temp *= cooling

        return best_positions, best_side

    def build_rigid_cluster(ring_r, gap_angle, gap_dist):
        """
        Build a rigid cluster: 1 center + 6 ring + 4 outer.
        """
        positions = [[0.0, 0.0, 0.0]]
        for k in range(6):
            ang = math.radians(60 * k)
            x = ring_r * math.cos(ang)
            y = ring_r * math.sin(ang)
            positions.append([x, y, 0.0])
        for g in range(4):
            ang_deg = gap_angle + g * 90.0
            ang = math.radians(ang_deg)
            x = gap_dist * math.cos(ang)
            y = gap_dist * math.sin(ang)
            positions.append([x, y, 0.0])
        return positions

    # Main optimization
    best_inner = None
    best_side = float('inf')
    best_angle = 0.0

    # Try various initial configurations
    # 1. Rigid cluster approach
    ring_r_values = [sqrt3 + 0.05, sqrt3 + 0.1, sqrt3 + 0.15, sqrt3 + 0.2]
    gap_angle_values = [0, 15, 30, 45]
    gap_dist_values = [2.5, 2.8, 3.0, 3.2, 3.5]

    for ring_r in ring_r_values:
        if time.time() > deadline - 30.0:
            break

        for gap_angle in gap_angle_values:
            if time.time() > deadline - 30.0:
                break

            for gap_dist in gap_dist_values:
                if time.time() > deadline - 30.0:
                    break

                positions = build_rigid_cluster(ring_r, gap_angle, gap_dist)

                # Check if valid (all disjoint)
                if all_disjoint(positions):
                    side = compute_min_side(positions, (0.0, 0.0), 0.0)

                    if side < best_side:
                        best_side = side
                        best_inner = [list(p) for p in positions]
                        best_angle = 0.0

    # 2. Simulated annealing from multiple starting points
    start_configs = []

    # Add the best rigid cluster found
    if best_inner is not None:
        start_configs.append(best_inner)

    # Add some heuristic starting configurations
    # Configuration 1: Center + 6 around + 4 at corners
    config1 = [[0.0, 0.0, 0.0]]
    for k in range(6):
        ang = math.radians(60 * k)
        x = (sqrt3 + 0.1) * math.cos(ang)
        y = (sqrt3 + 0.1) * math.sin(ang)
        config1.append([x, y, 0.0])
    for g in range(4):
        ang_deg = 30 + g * 90
        ang = math.radians(ang_deg)
        x = 3.0 * math.cos(ang)
        y = 3.0 * math.sin(ang)
        config1.append([x, y, 0.0])
    if all_disjoint(config1):
        start_configs.append(config1)

    # Configuration 2: Different arrangement
    config2 = [[0.0, 0.0, 0.0]]
    for k in range(6):
        ang = math.radians(60 * k + 30)
        x = (sqrt3 + 0.15) * math.cos(ang)
        y = (sqrt3 + 0.15) * math.sin(ang)
        config2.append([x, y, 0.0])
    for g in range(4):
        ang_deg = 0 + g * 90
        ang = math.radians(ang_deg)
        x = 3.2 * math.cos(ang)
        y = 3.2 * math.sin(ang)
        config2.append([x, y, 0.0])
    if all_disjoint(config2):
        start_configs.append(config2)

    # Run simulated annealing from each starting configuration
    for start_pos in start_configs:
        if time.time() > deadline - 25.0:
            break

        # Try different outer angles
        for outer_angle in [0, 15, 30]:
            if time.time() > deadline - 25.0:
                break

            refined_positions, refined_side = simulated_annealing(start_pos, outer_angle, max_iter=15000, start_temp=1.5, end_temp=0.005)

            # Polish with Nelder-Mead
            polished_positions, polished_side = nelder_mead_polish(refined_positions, outer_angle, max_iter=1500)

            if polished_side < best_side:
                best_side = polished_side
                best_inner = polished_positions
                best_angle = outer_angle

    # 3. Final optimization of outer angle
    if best_inner is not None and time.time() < deadline - 5.0:
        for outer_angle in range(0, 60, 1):
            side = compute_min_side(best_inner, (0.0, 0.0), outer_angle)
            if side < best_side:
                best_side = side
                best_angle = outer_angle

    # Ensure we have a valid result
    if best_inner is None:
        # Absolute fallback: safe configuration
        best_inner = []
        for i in range(N):
            if i == 0:
                best_inner.append([0.0, 0.0, 0.0])
            else:
                r = 2.5 * (i // 6 + 1)
                angle = math.radians(60 * (i % 6))
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                best_inner.append([x, y, 0.0])
        best_side = compute_min_side(best_inner, (0.0, 0.0), 0.0)
        best_angle = 0.0

    # Final verification
    if not all_disjoint(best_inner):
        # Fallback to safe configuration
        safe_inner = []
        for i in range(N):
            if i == 0:
                safe_inner.append([0.0, 0.0, 0.0])
            else:
                r = 2.5 * (i // 6 + 1)
                angle = math.radians(60 * (i % 6))
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                safe_inner.append([x, y, 0.0])
        best_inner = safe_inner
        best_side = compute_min_side(safe_inner, (0.0, 0.0), 0.0)
        best_angle = 0.0

    return best_inner, (0.0, 0.0), best_side, best_angle
