"""Pack 11 unit hexagons inside a regular hexagon, minimizing outer side length."""
import numpy as np
import math
import time
import random

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 142.0  # safety margin before 150s limit

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

        dx_c = p2[0] - p1[0]
        dy_c = p2[1] - p1[1]
        dist_c = math.sqrt(dx_c * dx_c + dy_c * dy_c)
        if dist_c > 1e-12:
            axes.append((dx_c / dist_c, dy_c / dist_c))

        for nx, ny in axes:
            min1 = min(vx * nx + vy * ny for vx, vy in verts1)
            max1 = max(vx * nx + vy * ny for vx, vy in verts1)
            min2 = min(vx * nx + vy * ny for vx, vy in verts2)
            max2 = max(vx * nx + vy * ny for vx, vy in verts2)

            gap = max(min2 - max1, min1 - max2)
            if gap > 0:
                return gap

        return 0.0

    def check_overlap(p1, p2):
        """Return overlap penalty: 0 if disjoint, positive if overlapping."""
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        d = math.sqrt(dx * dx + dy * dy)
        if d >= 2.0:
            return 0.0

        dist = hex_distance(p1, p2)
        if dist > 0:
            return 0.0
        else:
            return max(0.0, 2.0 - d)

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

    # ---- New structural approach: smooth objective + multi-start Nelder-Mead ----
    # Use a smooth approximation of the max projection (log-sum-exp) to remove
    # kinks, enabling more effective gradient-free optimization.

    def smooth_min_side(inner_positions, outer_center, outer_angle, beta=50.0):
        """
        Smooth approximation of compute_min_side using log-sum-exp.
        beta controls smoothness: higher = closer to exact max.
        """
        cx, cy = outer_center
        angle_rad = math.radians(outer_angle)
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)

        normals = []
        for k in range(6):
            theta = math.radians(30 + k * 60)
            normals.append((math.cos(theta), math.sin(theta)))

        all_projs = []
        for ix, iy, iang in inner_positions:
            inner_verts = hexagon_vertices(ix, iy, 1.0, iang)
            for vx, vy in inner_verts:
                dx = vx - cx
                dy = vy - cy
                rx = dx * cos_a - dy * sin_a
                ry = dx * sin_a + dy * cos_a

                for nx, ny in normals:
                    proj = (rx * nx + ry * ny) / (sqrt3 / 2)
                    all_projs.append(proj)

        # Log-sum-exp for smooth max
        max_p = max(all_projs)
        if max_p < -1000:
            return 0.0
        sum_exp = sum(math.exp(p - max_p) for p in all_projs)
        smooth_max = max_p + math.log(sum_exp) / beta
        return smooth_max + 1e-10

    def overlap_penalty_smooth(inner_positions, beta=20.0):
        """Smooth overlap penalty using softplus approximation."""
        penalty = 0.0
        n = len(inner_positions)
        for i in range(n):
            for j in range(i + 1, n):
                dx = inner_positions[i][0] - inner_positions[j][0]
                dy = inner_positions[i][1] - inner_positions[j][1]
                d = math.sqrt(dx * dx + dy * dy)
                # Softplus: log(1 + exp(-beta * (d - 2.0))) approximates max(0, 2-d)
                arg = -beta * (d - 2.0)
                if arg > 50:
                    penalty += arg / beta
                elif arg < -50:
                    penalty += 0.0
                else:
                    penalty += math.log(1.0 + math.exp(arg)) / beta
        return penalty

    def objective_smooth(params, outer_angle, beta_side=50.0, beta_overlap=20.0):
        """
        Combined smooth objective: outer side + weighted overlap penalty.
        params: flat list of [x0, y0, a0, x1, y1, a1, ...] for N hexagons.
        """
        positions = []
        for i in range(N):
            positions.append([params[3*i], params[3*i+1], params[3*i+2]])

        side = smooth_min_side(positions, (0.0, 0.0), outer_angle, beta_side)
        pen = overlap_penalty_smooth(positions, beta_overlap)
        return side + 10.0 * pen

    def nelder_mead_smooth(f, x0, tol=1e-9, max_iter=8000, step=0.5, time_limit=None):
        """
        Nelder-Mead with time limit support.
        """
        n = len(x0)
        simplex = [list(x0)]
        for i in range(n):
            pt = list(x0)
            pt[i] = pt[i] + step if abs(pt[i]) > 1e-6 else step
            simplex.append(pt)
        values = [f(pt) for pt in simplex]

        for it in range(max_iter):
            if time_limit is not None and time.time() > time_limit:
                break

            order = sorted(range(n + 1), key=lambda i: values[i])
            simplex = [simplex[i] for i in order]
            values = [values[i] for i in order]

            if values[-1] - values[0] < tol:
                break

            centroid = [sum(simplex[i][j] for i in range(n)) / n for j in range(n)]
            xr = [centroid[j] + step * (centroid[j] - simplex[-1][j]) for j in range(n)]
            fr = f(xr)

            if fr < values[0]:
                xe = [centroid[j] + 2 * step * (centroid[j] - simplex[-1][j]) for j in range(n)]
                fe = f(xe)
                if fe < fr:
                    simplex[-1] = xe
                    values[-1] = fe
                else:
                    simplex[-1] = xr
                    values[-1] = fr
            elif fr < values[-2]:
                simplex[-1] = xr
                values[-1] = fr
            else:
                xc = [centroid[j] + 0.5 * step * (simplex[-1][j] - centroid[j]) for j in range(n)]
                fc = f(xc)
                if fc < values[-1]:
                    simplex[-1] = xc
                    values[-1] = fc
                else:
                    for i in range(1, n + 1):
                        simplex[i] = [simplex[0][j] + 0.5 * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                        values[i] = f(simplex[i])

        best_idx = min(range(n + 1), key=lambda i: values[i])
        return simplex[best_idx]

    def params_to_positions(params):
        positions = []
        for i in range(N):
            positions.append([params[3*i], params[3*i+1], params[3*i+2]])
        return positions

    def positions_to_params(positions):
        params = []
        for p in positions:
            params.extend([p[0], p[1], p[2]])
        return params

    # ---- Multi-start initialization strategies ----

    def init_hexagonal_cluster():
        """Standard hexagonal cluster: 1 center + 6 ring + 4 extra."""
        params = []
        # Center
        params.extend([0.0, 0.0, 0.0])
        # Ring of 6
        ring_r = sqrt3 + 0.15
        for k in range(6):
            ang = math.radians(60 * k)
            x = ring_r * math.cos(ang)
            y = ring_r * math.sin(ang)
            params.extend([x, y, 0.0])
        # 4 more at gaps
        gap_dist = 2 * ring_r * math.cos(math.pi / 6) + 0.15
        for g in range(4):
            ang = math.radians(30 + 120 * g)
            x = gap_dist * math.cos(ang)
            y = gap_dist * math.sin(ang)
            params.extend([x, y, 0.0])
        return params

    def init_random_ring():
        """Random positions in a ring-like structure."""
        params = []
        params.extend([random.uniform(-0.3, 0.3), random.uniform(-0.3, 0.3), random.uniform(0, 60)])
        r1 = random.uniform(1.8, 2.5)
        for k in range(6):
            ang = math.radians(60 * k + random.uniform(-10, 10))
            x = r1 * math.cos(ang)
            y = r1 * math.sin(ang)
            params.extend([x, y, random.uniform(0, 60)])
        r2 = random.uniform(3.0, 4.0)
        for g in range(4):
            ang = math.radians(30 + 120 * g + random.uniform(-15, 15))
            x = r2 * math.cos(ang)
            y = r2 * math.sin(ang)
            params.extend([x, y, random.uniform(0, 60)])
        return params

    def init_tight_ring():
        """Tightly packed ring with small perturbations."""
        params = []
        params.extend([0.0, 0.0, 0.0])
        ring_r = sqrt3 + 0.05
        for k in range(6):
            ang = math.radians(60 * k)
            x = ring_r * math.cos(ang) + random.gauss(0, 0.1)
            y = ring_r * math.sin(ang) + random.gauss(0, 0.1)
            params.extend([x, y, random.uniform(-10, 10)])
        gap_dist = 2 * ring_r * math.cos(math.pi / 6) + 0.05
        for g in range(4):
            ang = math.radians(30 + 120 * g)
            x = gap_dist * math.cos(ang) + random.gauss(0, 0.1)
            y = gap_dist * math.sin(ang) + random.gauss(0, 0.1)
            params.extend([x, y, random.uniform(-10, 10)])
        return params

    def init_staggered():
        """Staggered two-row arrangement."""
        params = []
        # Row 1: 6 hexagons
        for i in range(6):
            x = i * 1.8 - 4.5
            y = 0.0
            params.extend([x, y, 0.0])
        # Row 2: 5 hexagons staggered
        for i in range(5):
            x = i * 1.8 - 3.6 + 0.9
            y = 1.55
            params.extend([x, y, 0.0])
        return params

    def init_spiral():
        """Spiral arrangement."""
        params = []
        params.extend([0.0, 0.0, 0.0])
        for k in range(10):
            ang = k * 0.55 + 0.3
            r = 1.5 + k * 0.35
            x = r * math.cos(ang)
            y = r * math.sin(ang)
            params.extend([x, y, random.uniform(0, 60)])
        return params

    # ---- Main optimization: multi-start smooth NM + hard refinement ----

    best_inner = None
    best_side = float('inf')
    best_angle = 0.0

    init_strategies = [
        init_hexagonal_cluster,
        init_tight_ring,
        init_random_ring,
        init_staggered,
        init_spiral,
    ]

    outer_angles = [0, 15, 30, 45]

    # Phase 1: Multi-start smooth Nelder-Mead
    for angle in outer_angles:
        if time.time() > deadline - 30.0:
            break

        for init_fn in init_strategies:
            if time.time() > deadline - 30.0:
                break

            x0 = init_fn()
            # Run smooth NM
            t_limit = min(deadline - 25.0, time.time() + 15.0)
            result = nelder_mead_smooth(
                lambda p: objective_smooth(p, angle, beta_side=50.0, beta_overlap=20.0),
                x0,
                tol=1e-10,
                max_iter=5000,
                step=0.4,
                time_limit=t_limit
            )

            positions = params_to_positions(result)

            # Check if valid (no overlaps)
            if all_disjoint(positions, tol=1e-8):
                side = compute_min_side(positions, (0.0, 0.0), angle)
                if side < best_side:
                    best_side = side
                    best_inner = positions
                    best_angle = angle

            # Phase 2: Hard refinement with Nelder-Mead on exact objective
            if time.time() > deadline - 15.0:
                break

            def objective_hard(p):
                pos = params_to_positions(p)
                # Penalty for overlaps
                pen = 0.0
                for i in range(N):
                    for j in range(i + 1, N):
                        dx = pos[i][0] - pos[j][0]
                        dy = pos[i][1] - pos[j][1]
                        d = math.sqrt(dx * dx + dy * dy)
                        if d < 2.0:
                            pen += (2.0 - d) ** 2
                side = compute_min_side(pos, (0.0, 0.0), angle)
                return side + 50.0 * pen

            t_limit2 = min(deadline - 10.0, time.time() + 12.0)
            result2 = nelder_mead_smooth(
                objective_hard,
                result,
                tol=1e-11,
                max_iter=4000,
                step=0.2,
                time_limit=t_limit2
            )

            positions2 = params_to_positions(result2)
            if all_disjoint(positions2, tol=1e-8):
                side2 = compute_min_side(positions2, (0.0, 0.0), angle)
                if side2 < best_side:
                    best_side = side2
                    best_inner = positions2
                    best_angle = angle

    # Phase 3: If we have a good solution, try optimizing outer angle finely
    if best_inner is not None and time.time() < deadline - 5.0:
        for outer_angle in range(0, 60, 2):
            side = compute_min_side(best_inner, (0.0, 0.0), outer_angle)
            if side < best_side:
                best_side = side
                best_angle = outer_angle

    # Phase 4: Final local search with small perturbations
    if best_inner is not None and time.time() < deadline - 3.0:
        current_positions = [list(p) for p in best_inner]
        current_side = best_side
        temp = 0.3
        cooling = 0.99

        for it in range(2000):
            if time.time() > deadline - 2.0:
                break

            idx = random.randint(0, N - 1)
            old_x, old_y, old_ang = current_positions[idx]
            dx = random.gauss(0, 0.08)
            dy = random.gauss(0, 0.08)
            da = random.gauss(0, 3.0)
            new_x = old_x + dx
            new_y = old_y + dy
            new_ang = old_ang + da

            candidate = [new_x, new_y, new_ang]
            disjoint = True
            for j in range(N):
                if j != idx and not is_disjoint(candidate, current_positions[j]):
                    disjoint = False
                    break

            if not disjoint:
                continue

            trial_positions = [list(p) for p in current_positions]
            trial_positions[idx] = candidate
            trial_side = compute_min_side(trial_positions, (0.0, 0.0), best_angle)

            delta = trial_side - current_side
            if delta < 0 or random.random() < math.exp(-delta / max(temp, 1e-10)):
                current_positions[idx] = candidate
                current_side = trial_side
                if current_side < best_side:
                    best_side = current_side
                    best_inner = [list(p) for p in current_positions]

            temp *= cooling

    # Final verification
    if best_inner is not None:
        if not all_disjoint(best_inner):
            # Fallback to safe configuration
            safe_inner = []
            for i in range(N):
                if i == 0:
                    safe_inner.append([0.0, 0.0, 0.0])
                else:
                    r = 2.2 * (i // 6 + 1)
                    angle = math.radians(60 * (i % 6))
                    x = r * math.cos(angle)
                    y = r * math.sin(angle)
                    safe_inner.append([x, y, 0.0])
            best_inner = safe_inner
            best_side = compute_min_side(safe_inner, (0.0, 0.0), 0.0)
            best_angle = 0.0

    if best_inner is None:
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

    return best_inner, (0.0, 0.0), best_side, best_angle
# EVOLVE-BLOCK-END
