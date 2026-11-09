"""Packing unit hexagons in a regular hexagon via rigid-body dynamics with velocity-Verlet."""
import time
import math
import random
import numpy as np

N = 11

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s limit

    sqrt3 = math.sqrt(3)

    def hex_vertices(cx, cy, side, angle_rad):
        """Return 6 vertices of regular hexagon centered at (cx,cy) with circumradius=side, rotation angle_rad."""
        verts = np.empty((6, 2))
        for i in range(6):
            a = angle_rad + i * math.pi / 3.0
            verts[i, 0] = cx + side * math.cos(a)
            verts[i, 1] = cy + side * math.sin(a)
        return verts

    def point_in_hexagon(px, py, cx, cy, side, angle_rad):
        """Check if point (px,py) is inside hexagon centered at (cx,cy) with circumradius=side, rotation angle_rad."""
        dx = px - cx
        dy = py - cy
        # Rotate point into hexagon's local frame
        ca = math.cos(-angle_rad)
        sa = math.sin(-angle_rad)
        rx = dx * ca - dy * sa
        ry = dx * sa + dy * ca
        apothem = side * sqrt3 / 2.0
        for k in range(6):
            theta = math.radians(30 + k * 60)
            nx = math.cos(theta)
            ny = math.sin(theta)
            if rx * nx + ry * ny > apothem + 1e-9:
                return False
        return True

    def segments_intersect(p1, p2, p3, p4):
        """Check if segment p1-p2 intersects segment p3-p4."""
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        return False

    def hexagons_overlap(i1, i2, eps=1e-7):
        """Check if two unit hexagons overlap. i1, i2 are (x, y, angle_rad)."""
        x1, y1, a1 = i1
        x2, y2, a2 = i2
        dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        if dist > 2.0 + eps:
            return False
        if abs(a1 - a2) < 1e-6:
            return dist < 2.0 - eps
        v1 = hex_vertices(x1, y1, 1.0, a1)
        v2 = hex_vertices(x2, y2, 1.0, a2)
        # Vertex containment
        for vx, vy in v1:
            if point_in_hexagon(vx, vy, x2, y2, 1.0, a2):
                return True
        for vx, vy in v2:
            if point_in_hexagon(vx, vy, x1, y1, 1.0, a1):
                return True
        # Edge intersections
        for i in range(6):
            ax, ay = v1[i]
            bx, by = v1[(i + 1) % 6]
            for j in range(6):
                cx, cy = v2[j]
                dx, dy = v2[(j + 1) % 6]
                if segments_intersect((ax, ay), (bx, by), (cx, cy), (dx, dy)):
                    return True
        return False

    def all_disjoint(positions, angles, eps=1e-7):
        """Check all pairs are disjoint."""
        for i in range(N):
            for j in range(i + 1, N):
                if hexagons_overlap((positions[i, 0], positions[i, 1], angles[i]),
                                    (positions[j, 0], positions[j, 1], angles[j]), eps):
                    return False
        return True

    def compute_enclosing_side(positions, angles, center, outer_angle_rad):
        """Compute minimum outer hexagon side (circumradius) to contain all inner hexagons."""
        cx, cy = center
        min_side = 0.0
        for i in range(N):
            verts = hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
            for k in range(6):
                vx, vy = verts[k]
                rx = vx - cx
                ry = vy - cy
                for m in range(6):
                    a = outer_angle_rad + m * math.pi / 3.0
                    proj = rx * math.cos(a) + ry * math.sin(a)
                    required = proj / (sqrt3 / 2.0)
                    if required > min_side:
                        min_side = required
        return max(min_side, 1.0)

    def evaluate(positions, angles, center, outer_angle_rad):
        """Full evaluation. Returns (side, valid)."""
        if not all_disjoint(positions, angles):
            return None, False
        side = compute_enclosing_side(positions, angles, center, outer_angle_rad)
        return side, True

    # --- Rigid-body dynamics simulation ---
    def run_dynamics(init_positions, init_angles, outer_angle_rad, dt=0.005, max_steps=50000):
        """
        Velocity-Verlet integration with contact impulses.
        Inner hexagons are attracted to center; collisions resolved via impulse.
        Returns best (positions, angles, side, valid) found during simulation.
        """
        n = N
        pos = init_positions.copy()
        ang = init_angles.copy()
        vel = np.zeros((n, 2))
        ang_vel = np.zeros(n)

        # Attraction strength and damping
        attract_k = 0.5
        damping = 0.98
        ang_damping = 0.95
        ang_attract = 0.0  # no torque attraction, just free rotation
        contact_stiffness = 50.0
        contact_damping = 5.0
        max_force = 10.0

        best_side = float('inf')
        best_pos = pos.copy()
        best_ang = ang.copy()
        best_valid = False

        for step in range(max_steps):
            if time.time() > deadline - 10:
                break

            # Compute forces
            forces = np.zeros((n, 2))
            torques = np.zeros(n)

            # Attraction to center (pulls everything inward)
            for i in range(n):
                rx = -pos[i, 0]
                ry = -pos[i, 1]
                forces[i, 0] += attract_k * rx
                forces[i, 1] += attract_k * ry

            # Pairwise repulsion (soft contact)
            for i in range(n):
                for j in range(i + 1, n):
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist < 1e-9:
                        dist = 1e-9
                        dx = 0.1
                        dy = 0.1
                    # Minimum distance depends on relative angle
                    # For same orientation: 2.0; for different: ~1.732 (sqrt(3))
                    da = abs(ang[i] - ang[j]) % math.pi
                    if da > math.pi / 2:
                        da = math.pi - da
                    # Approximate minimum distance
                    min_dist = 2.0 * math.cos(da / 2.0) + 0.1 * math.sin(da)
                    if dist < min_dist:
                        overlap = min_dist - dist
                        nx = dx / dist
                        ny = dy / dist
                        f = contact_stiffness * overlap + contact_damping * (vel[i, 0] - vel[j, 0]) * nx + contact_damping * (vel[i, 1] - vel[j, 1]) * ny
                        f = min(f, max_force)
                        forces[i, 0] += f * nx
                        forces[i, 1] += f * ny
                        forces[j, 0] -= f * nx
                        forces[j, 1] -= f * ny
                        # Torque from tangential component
                        tx = -ny
                        ty = nx
                        vt = (vel[i, 0] - vel[j, 0]) * tx + (vel[i, 1] - vel[j, 1]) * ty
                        t = contact_stiffness * 0.3 * overlap * vt
                        t = max(-max_force, min(max_force, t))
                        torques[i] += t
                        torques[j] -= t

            # Velocity-Verlet: update velocity half-step
            vel_half = vel + 0.5 * dt * forces
            ang_vel_half = ang_vel + 0.5 * dt * torques

            # Update positions
            pos_new = pos + dt * vel_half
            ang_new = ang + dt * ang_vel_half

            # Compute forces at new positions
            forces_new = np.zeros((n, 2))
            torques_new = np.zeros(n)
            for i in range(n):
                forces_new[i, 0] += attract_k * (-pos_new[i, 0])
                forces_new[i, 1] += attract_k * (-pos_new[i, 1])
            for i in range(n):
                for j in range(i + 1, n):
                    dx = pos_new[i, 0] - pos_new[j, 0]
                    dy = pos_new[i, 1] - pos_new[j, 1]
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist < 1e-9:
                        continue
                    da = abs(ang_new[i] - ang_new[j]) % math.pi
                    if da > math.pi / 2:
                        da = math.pi - da
                    min_dist = 2.0 * math.cos(da / 2.0) + 0.1 * math.sin(da)
                    if dist < min_dist:
                        overlap = min_dist - dist
                        nx = dx / dist
                        ny = dy / dist
                        f = contact_stiffness * overlap
                        f = min(f, max_force)
                        forces_new[i, 0] += f * nx
                        forces_new[i, 1] += f * ny
                        forces_new[j, 0] -= f * nx
                        forces_new[j, 1] -= f * ny
                        tx = -ny
                        ty = nx
                        vt = (vel_half[i, 0] - vel_half[j, 0]) * tx + (vel_half[i, 1] - vel_half[j, 1]) * ty
                        t = contact_stiffness * 0.3 * overlap * vt
                        t = max(-max_force, min(max_force, t))
                        torques_new[i] += t
                        torques_new[j] -= t

            # Update velocity full-step with damping
            vel = (vel_half + 0.5 * dt * forces_new) * damping
            ang_vel = (ang_vel_half + 0.5 * dt * torques_new) * ang_damping

            pos = pos_new
            ang = ang_new

            # Evaluate periodically
            if step % 50 == 0 or step == max_steps - 1:
                side, valid = evaluate(pos, ang, (0.0, 0.0), outer_angle_rad)
                if valid and side < best_side:
                    best_side = side
                    best_pos = pos.copy()
                    best_ang = ang.copy()
                    best_valid = True

        return best_pos, best_ang, best_side, best_valid

    # --- Initial layout: hexagonal cluster ---
    def axial_to_cartesian(q, r):
        x = 2.0 * q + r
        y = sqrt3 * r
        return x, y

    def hex_ring_positions(ring):
        if ring == 0:
            return [(0, 0)]
        positions = []
        q, r = ring, 0
        for side_idx in range(6):
            for step in range(ring):
                positions.append((q, r))
                dq = [1, 0, -1, -1, 0, 1][side_idx]
                dr = [0, -1, -1, 0, 1, 1][side_idx]
                q += dq
                r += dr
        return positions

    all_axial = [(0, 0)]
    ring = 1
    while len(all_axial) < N:
        all_axial.extend(hex_ring_positions(ring))
        ring += 1
    selected = all_axial[:N]

    # Convert to cartesian and center
    init_pos_list = []
    for q, r in selected:
        x, y = axial_to_cartesian(q, r)
        init_pos_list.append([x, y])
    cx_mean = sum(p[0] for p in init_pos_list) / N
    cy_mean = sum(p[1] for p in init_pos_list) / N
    init_positions = np.array([[p[0] - cx_mean, p[1] - cy_mean] for p in init_pos_list])
    init_angles = np.zeros(N)

    # --- Multi-restart dynamics with different outer angles ---
    best_side_overall = float('inf')
    best_result = None

    outer_angles_to_try = [0.0, math.pi / 6, math.pi / 12, math.pi / 4, math.pi / 3, 5 * math.pi / 6]

    for outer_angle_rad in outer_angles_to_try:
        if time.time() > deadline - 20:
            break

        # Try a few different initial perturbations
        for trial in range(3):
            if time.time() > deadline - 15:
                break

            # Slight random perturbation for diversity
            trial_pos = init_positions.copy()
            trial_ang = init_angles.copy()
            if trial > 0:
                trial_pos += np.random.normal(0, 0.1, (N, 2))
                trial_ang += np.random.uniform(-0.1, 0.1, N)

            pos, ang, side, valid = run_dynamics(trial_pos, trial_ang, outer_angle_rad)
            if valid and side < best_side_overall:
                best_side_overall = side
                best_result = (pos, ang, (0.0, 0.0), outer_angle_rad)

    # --- Local refinement with simulated annealing on best result ---
    if best_result is not None:
        pos, ang, center, outer_angle_rad = best_result
        current_pos = pos.copy()
        current_ang = ang.copy()
        current_side = best_side_overall

        T = 0.1
        T_min = 0.001
        cooling = 0.997
        max_sa_iters = 20000

        for it in range(max_sa_iters):
            if time.time() > deadline - 5:
                break

            # Perturb one hexagon
            i = random.randint(0, N - 1)
            current_pos[i, 0] += random.gauss(0, T * 0.3)
            current_pos[i, 1] += random.gauss(0, T * 0.3)
            if random.random() < 0.3:
                current_ang[i] += random.gauss(0, T * 0.2)
                current_ang[i] %= math.pi

            new_side, new_valid = evaluate(current_pos, current_ang, center, outer_angle_rad)
            if new_valid:
                delta = new_side - current_side
                if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                    current_side = new_side
                    if new_side < best_side_overall:
                        best_side_overall = new_side
                        best_result = (current_pos.copy(), current_ang.copy(), center, outer_angle_rad)
                else:
                    # Revert
                    current_pos[i, 0] -= random.gauss(0, T * 0.3)
                    current_pos[i, 1] -= random.gauss(0, T * 0.3)
                    if random.random() < 0.3:
                        current_ang[i] -= random.gauss(0, T * 0.2)
                        current_ang[i] %= math.pi
            else:
                # Revert
                current_pos[i, 0] -= random.gauss(0, T * 0.3)
                current_pos[i, 1] -= random.gauss(0, T * 0.3)
                if random.random() < 0.3:
                    current_ang[i] -= random.gauss(0, T * 0.2)
                    current_ang[i] %= math.pi

            T *= cooling
            if T < T_min:
                T = T_min

    # --- Final answer ---
    if best_result is not None:
        pos, ang, center, outer_angle_rad = best_result
        final_side = compute_enclosing_side(pos, ang, center, outer_angle_rad)
        # Final validity check
        valid = all_disjoint(pos, ang)
        if valid:
            inner = [[pos[i, 0], pos[i, 1], math.degrees(ang[i])] for i in range(N)]
            return inner, center, final_side, math.degrees(outer_angle_rad)

    # Fallback: linear layout
    inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    return inner, (0.0, 0.0), 40.0, 0.0

# EVOLVE-BLOCK-END
