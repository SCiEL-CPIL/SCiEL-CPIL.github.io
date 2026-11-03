"""Rewritten: physics-based packing with repulsive forces + annealing, then SLSQP polish."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit

    def hex_vertices(cx, cy, angle_deg):
        """Get 6 vertices of a unit regular hexagon centered at (cx,cy) rotated by angle_deg."""
        angle_rad = math.radians(angle_deg)
        verts = np.array([[math.cos(angle_rad + i * math.pi / 3),
                           math.sin(angle_rad + i * math.pi / 3)] for i in range(6)])
        return verts + np.array([cx, cy])

    def sat_overlap(v1, v2):
        """SAT overlap test for two convex hexagons (6x2 arrays)."""
        for i in range(6):
            for pts in (v1, v2):
                edge = pts[(i + 1) % 6] - pts[i]
                normal = np.array([-edge[1], edge[0]])
                proj_a = np.dot(v1, normal)
                proj_b = np.dot(v2, normal)
                if proj_a.max() < proj_b.min() or proj_b.max() < proj_a.min():
                    return False
        return True

    def all_overlap_check(positions, angles):
        """Return True if any pair overlaps."""
        verts = [hex_vertices(positions[i][0], positions[i][1], angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i + 1, N):
                if sat_overlap(verts[i], verts[j]):
                    return True
        return False

    def compute_outer_side(positions, angles, outer_angle_deg):
        """Compute minimum outer hexagon side length to contain all inner hexagons."""
        all_verts = np.vstack([hex_vertices(positions[i][0], positions[i][1], angles[i])
                               for i in range(N)])
        theta = math.radians(outer_angle_deg)
        max_proj = 0.0
        for k in range(6):
            a = theta + k * math.pi / 3
            normal = np.array([math.cos(a), math.sin(a)])
            proj = np.dot(all_verts, normal)
            max_proj = max(max_proj, proj.max())
        return 2.0 * max_proj / math.sqrt(3)

    def score_config(positions, angles, outer_angle_deg):
        """Return outer side length if no overlaps, else inf."""
        if all_overlap_check(positions, angles):
            return float('inf')
        return compute_outer_side(positions, angles, outer_angle_deg)

    # Physics-based repulsion optimization
    def physics_optimize(positions, angles, outer_angle_deg, max_iters=30000):
        """
        Simulate hexagons as rigid bodies with repulsive forces when overlapping.
        Use velocity Verlet integration with cooling.
        """
        pos = np.array(positions, dtype=float)  # (N, 2)
        vel = np.zeros((N, 2))
        best_score = score_config([tuple(p) for p in pos], angles, outer_angle_deg)
        best_pos = pos.copy()

        if best_score < float('inf'):
            return [tuple(p) for p in best_pos], best_score

        dt = 0.02
        damping = 0.98
        k_repulse = 50.0

        for it in range(max_iters):
            if time.time() > deadline - 10:
                break

            # Compute forces from overlapping pairs
            forces = np.zeros((N, 2))
            verts = [hex_vertices(pos[i][0], pos[i][1], angles[i]) for i in range(N)]

            for i in range(N):
                for j in range(i + 1, N):
                    if sat_overlap(verts[i], verts[j]):
                        # Push apart along center-to-center direction
                        d = pos[j] - pos[i]
                        dist = np.linalg.norm(d)
                        if dist < 1e-10:
                            d = np.array([1.0, 0.0])
                            dist = 1.0
                        d_hat = d / dist
                        # Estimate penetration depth (approximate)
                        # For unit hexagons, minimum non-overlap distance is ~2*apothem = sqrt(3)
                        pen = max(0.0, 2.0 * math.sqrt(3) / 2.0 - dist)
                        # Actually for hexagons, the minimum center distance for no overlap
                        # depends on orientation. Use a simpler penalty.
                        pen = max(0.0, 1.5 - dist)
                        f = k_repulse * pen * d_hat
                        forces[i] -= f
                        forces[j] += f

            # Also add a gentle centering force to keep things compact
            center = pos.mean(axis=0)
            for i in range(N):
                forces[i] += -0.1 * (pos[i] - center)

            # Integrate
            vel = vel * damping + forces * dt
            pos = pos + vel * dt

            # Check score
            current_score = score_config([tuple(p) for p in pos], angles, outer_angle_deg)
            if current_score < best_score:
                best_score = current_score
                best_pos = pos.copy()

            if best_score < float('inf'):
                # Once valid, do a few more iterations to tighten
                if it > 100:
                    break

        return [tuple(p) for p in best_pos], best_score

    def slsqp_polish(positions, angles, outer_angle_deg, max_iter=500):
        """
        Use SLSQP to minimize outer side length with non-overlap constraints.
        Variables: 2N positions + N angles (total 3N = 36).
        """
        n = N
        x0 = np.zeros(3 * n)
        for i in range(n):
            x0[2*i] = positions[i][0]
            x0[2*i+1] = positions[i][1]
            x0[3*n + i] = angles[i]

        def unpack(x):
            pos = np.array([(x[2*i], x[2*i+1]) for i in range(n)])
            ang = x[3*n:3*n+n]
            return pos, ang

        def objective(x):
            if time.time() > deadline - 5:
                return 1e10
            pos, ang = unpack(x)
            verts = [hex_vertices(pos[i][0], pos[i][1], ang[i]) for i in range(n)]
            # Overlap penalty (soft)
            overlap_pen = 0.0
            for i in range(n):
                for j in range(i + 1, n):
                    if sat_overlap(verts[i], verts[j]):
                        d = pos[j] - pos[i]
                        dist = np.linalg.norm(d)
                        overlap_pen += max(0.0, 1.5 - dist) ** 2
            if overlap_pen > 1e-8:
                return 1000.0 + overlap_pen
            return compute_outer_side([tuple(p) for p in pos], list(ang), outer_angle_deg)

        # Constraints: no overlaps (hard, using distance lower bounds)
        constraints = []
        for i in range(n):
            for j in range(i + 1, n):
                def make_constraint(i, j):
                    def constraint(x):
                        pos, ang = unpack(x)
                        d = pos[j] - pos[i]
                        dist = np.linalg.norm(d)
                        # Minimum center distance for two unit hexagons (conservative)
                        return dist - 1.7
                    return constraint
                constraints.append({'type': 'ineq', 'fun': make_constraint(i, j)})

        try:
            result = minimize(objective, x0, method='SLSQP',
                             constraints=constraints,
                             options={'maxiter': max_iter, 'ftol': 1e-6})
            x_final = result.x
            pos_final, ang_final = unpack(x_final)

            # Verify
            if not all_overlap_check([tuple(p) for p in pos_final], list(ang_final)):
                s = compute_outer_side([tuple(p) for p in pos_final], list(ang_final), outer_angle_deg)
                return [tuple(p) for p in pos_final], list(ang_final), s
        except Exception:
            pass
        return None

    def generate_initial_layouts():
        """Generate diverse initial layouts."""
        layouts = []

        # Layout 1: Honeycomb ring (center + 6 + 5 from ring 2)
        ring1_dirs = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
        layout = [(0, 0)]
        layout.extend(ring1_dirs)
        # Ring 2 candidates
        ring2 = []
        for (q, r) in ring1_dirs:
            for dq, dr in ring1_dirs:
                nq, nr = q + dq, r + dr
                if (nq, nr) not in layout and (nq, nr) not in ring2:
                    ring2.append((nq, nr))
        layout.extend(ring2[:5])
        layouts.append(layout)

        # Layout 2: 3x4 staggered grid
        layout2 = []
        for row in range(3):
            for col in range(4):
                q = col + (0.5 if row % 2 else 0)
                r = row
                layout2.append((q, r))
        layouts.append(layout2)

        # Layout 3: 4x3 staggered
        layout3 = []
        for row in range(4):
            for col in range(3):
                q = col + (0.5 if row % 2 else 0)
                r = row
                layout3.append((q, r))
        layouts.append(layout3)

        # Layout 4: 2x6
        layout4 = []
        for row in range(2):
            for col in range(6):
                q = col + (0.5 if row % 2 else 0)
                r = row
                layout4.append((q, r))
        layouts.append(layout4)

        # Layout 5: Circular arrangement
        layout5 = []
        for i in range(N):
            a = 2 * math.pi * i / N
            layout5.append((math.cos(a), math.sin(a)))
        layouts.append(layout5)

        # Layout 6: Two concentric rings (6+6, no center)
        layout6 = []
        for i in range(6):
            a = 2 * math.pi * i / 6
            layout6.append((math.cos(a), math.sin(a)))
        for i in range(6):
            a = 2 * math.pi * i / 6 + math.pi / 6
            layout6.append((2 * math.cos(a), 2 * math.sin(a)))
        layouts.append(layout6)

        return layouts

    def axial_to_cartesian(q, r):
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return x, y

    def layout_to_positions(layout, rotation_deg=0.0, scale=1.0):
        """Convert layout to Cartesian positions, centered and optionally rotated/scaled."""
        positions = []
        for item in layout:
            if len(item) == 2 and all(isinstance(v, (int, float)) for v in item):
                if isinstance(layout[0], tuple) and len(layout[0]) == 2 and all(
                        isinstance(v, (int, float)) for v in layout[0]):
                    # Could be axial or cartesian - treat as cartesian if values > 2
                    if max(abs(v) for v in item) > 2.5:
                        positions.append((item[0], item[1]))
                    else:
                        x, y = axial_to_cartesian(item[0], item[1])
                        positions.append((x, y))
                else:
                    positions.append(item)
            else:
                positions.append(item)

        # Center
        cx = sum(p[0] for p in positions) / len(positions)
        cy = sum(p[1] for p in positions) / len(positions)
        positions = [(x - cx, y - cy) for x, y in positions]

        # Scale
        if scale != 1.0:
            positions = [(x * scale, y * scale) for x, y in positions]

        # Rotate
        if rotation_deg != 0:
            theta = math.radians(rotation_deg)
            cos_t, sin_t = math.cos(theta), math.sin(theta)
            positions = [(x * cos_t - y * sin_t, x * sin_t + y * cos_t)
                         for x, y in positions]

        return positions

    # Main search
    best_score = float('inf')
    best_positions = None
    best_angles = None
    best_outer_angle = 0.0

    layouts = generate_initial_layouts()
    rotations = [0, 15, 30, 45, 60, 75, 90, 120, 135, 150]
    outer_angles = [0, 30, 60, 90, 120, 150]

    for layout in layouts:
        if time.time() > deadline - 30:
            break

        for rot in rotations:
            if time.time() > deadline - 15:
                break

            for oa in outer_angles:
                if time.time() > deadline - 10:
                    break

                positions = layout_to_positions(layout, rotation_deg=rot)
                angles = [0.0] * N

                score = score_config(positions, angles, oa)
                if score < best_score:
                    best_score = score
                    best_positions = positions
                    best_angles = angles
                    best_outer_angle = oa

                # If valid, try to improve
                if score < float('inf') and time.time() < deadline - 20:
                    # Physics optimization
                    phys_pos, phys_score = physics_optimize(
                        positions, angles, oa, max_iters=10000)
                    if phys_score < best_score:
                        best_score = phys_score
                        best_positions = phys_pos
                        best_angles = angles
                        best_outer_angle = oa

                    # SLSQP polish
                    if time.time() < deadline - 15:
                        slsqp_result = slsqp_polish(
                            best_positions, best_angles, best_outer_angle, max_iter=300)
                        if slsqp_result is not None:
                            sl_pos, sl_ang, sl_score = slsqp_result
                            if sl_score < best_score:
                                best_score = sl_score
                                best_positions = sl_pos
                                best_angles = sl_ang
                                best_outer_angle = oa

                # If invalid, try physics to fix
                elif score == float('inf') and time.time() < deadline - 20:
                    phys_pos, phys_score = physics_optimize(
                        positions, angles, oa, max_iters=20000)
                    if phys_score < best_score:
                        best_score = phys_score
                        best_positions = phys_pos
                        best_angles = angles
                        best_outer_angle = oa

    # Final SLSQP polish on best
    if best_positions is not None and best_score < float('inf') and time.time() < deadline - 10:
        slsqp_result = slsqp_polish(
            best_positions, best_angles, best_outer_angle, max_iter=500)
        if slsqp_result is not None:
            sl_pos, sl_ang, sl_score = slsqp_result
            if sl_score < best_score:
                best_score = sl_score
                best_positions = sl_pos
                best_angles = sl_ang

    # Fallback
    if best_positions is None or best_score == float('inf'):
        positions = []
        for i in range(N):
            r = i // 4
            c = i % 4
            x = c * 2.2 + (r % 2) * 1.1
            y = r * 1.9
            positions.append((x, y))
        cx = sum(p[0] for p in positions) / N
        cy = sum(p[1] for p in positions) / N
        positions = [(x - cx, y - cy) for x, y in positions]
        best_positions = positions
        best_angles = [0.0] * N
        best_outer_angle = 0.0
        best_score = compute_outer_side(positions, best_angles, 0.0)

    # Center final configuration
    cx = sum(p[0] for p in best_positions) / N
    cy = sum(p[1] for p in best_positions) / N
    final_positions = [(x - cx, y - cy) for x, y in best_positions]

    inner = [[final_positions[i][0], final_positions[i][1], best_angles[i]] for i in range(N)]

    return inner, (0.0, 0.0), best_score, best_outer_angle
# EVOLVE-BLOCK-END
