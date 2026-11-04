import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0
    random.seed(42)
    np.random.seed(42)

    sqrt3 = math.sqrt(3)

    # Precompute unit hexagon vertices (pointy-top)
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def get_hex_verts(cx, cy, side, angle_rad):
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def get_all_verts(positions, angles):
        """Get vertices for all hexagons. positions: (N,2), angles: (N,) in radians"""
        verts = []
        for i in range(N):
            verts.append(get_hex_verts(positions[i, 0], positions[i, 1], 1.0, angles[i]))
        return verts

    def calc_min_outer_side(positions, angles, outer_angle_rad=0.0):
        """Calculate minimum side length of regular hexagon containing all inner hexagons."""
        theta_rad = outer_angle_rad
        normals = []
        for k in range(6):
            ang = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang), math.sin(ang)])
        normals = np.array(normals)  # (6, 2)

        max_proj = 0.0
        for i in range(N):
            cx, cy = positions[i, 0], positions[i, 1]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            projs = verts @ normals.T  # (6, 6)
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i

        return max_proj * 2.0 / sqrt3

    def sat_gap(poly1, poly2):
        """
        Compute the minimum gap between two convex polygons using SAT.
        Returns positive if separated, negative if overlapping.
        """
        min_gap = float('inf')

        # Axes from poly1
        for i in range(6):
            edge = poly1[(i+1)%6] - poly1[i]
            norm = np.linalg.norm(edge)
            if norm < 1e-9:
                continue
            axis = np.array([-edge[1], edge[0]]) / norm

            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)

            gap = max(min1 - max2, min2 - max1)
            if gap < 0:
                min_gap = min(min_gap, gap)
            else:
                return gap  # Separated

        # Axes from poly2
        for i in range(6):
            edge = poly2[(i+1)%6] - poly2[i]
            norm = np.linalg.norm(edge)
            if norm < 1e-9:
                continue
            axis = np.array([-edge[1], edge[0]]) / norm

            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)

            gap = max(min1 - max2, min2 - max1)
            if gap < 0:
                min_gap = min(min_gap, gap)
            else:
                return gap  # Separated

        return min_gap

    def total_overlap_penalty(polys, weight=50.0):
        """Smooth quadratic penalty for overlaps."""
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                gap = sat_gap(polys[i], polys[j])
                if gap < 0:
                    penalty += weight * gap * gap
        return penalty

    def is_valid(positions, angles, tol=1e-6):
        """Check if configuration is valid (no overlaps)."""
        polys = get_all_verts(positions, angles)
        for i in range(N):
            for j in range(i+1, N):
                if sat_gap(polys[i], polys[j]) < -tol:
                    return False
        return True

    def smooth_objective(params):
        """Smooth objective: outer side + quadratic overlap penalty."""
        if time.time() > deadline - 5:
            return 1e6
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:]

        base = calc_min_outer_side(positions, angles, 0.0)
        polys = get_all_verts(positions, angles)
        penalty = total_overlap_penalty(polys, weight=50.0)
        return base + penalty

    def generate_honeycomb_config(radius):
        """Generate hexagons in a honeycomb pattern."""
        positions = []
        for q in range(-radius, radius+1):
            for r in range(-radius, radius+1):
                if abs(q + r) <= radius:
                    x = 1.5 * q
                    y = sqrt3 * (r + 0.5 * q)
                    positions.append((x, y))
        positions.sort(key=lambda p: p[0]**2 + p[1]**2)
        return np.array(positions[:N])

    def generate_ring_config():
        """Generate hexagons in a ring pattern around center."""
        positions = []
        positions.append((0, 0))
        for k in range(6):
            ang = k * math.pi / 3
            x = sqrt3 * math.cos(ang)
            y = sqrt3 * math.sin(ang)
            positions.append((x, y))
        for k in range(6):
            ang = k * math.pi / 3 + math.pi/6
            x = 2 * sqrt3 * math.cos(ang)
            y = 2 * sqrt3 * math.sin(ang)
            positions.append((x, y))
        return np.array(positions[:N])

    def generate_lattice_config():
        d = sqrt3 + 0.01
        u = np.array([d, 0.0])
        v = np.array([d / 2.0, d * sqrt3 / 2.0])

        centers = [np.array([0.0, 0.0])]
        ring1_vectors = [u, v, v - u, -u, -v, u - v]
        for vec in ring1_vectors:
            centers.append(vec.copy())

        ring2_candidates = []
        basis = [u, v, v - u, -u, -v, u - v]
        for i in range(6):
            p1 = ring1_vectors[i]
            for b in basis:
                p2 = p1 + b
                is_r0 = np.allclose(p2, np.array([0.0, 0.0]))
                is_r1 = False
                for p1j in ring1_vectors:
                    if np.allclose(p2, p1j):
                        is_r1 = True
                        break
                if not is_r0 and not is_r1:
                    duplicate = False
                    for pc in ring2_candidates:
                        if np.allclose(p2, pc):
                            duplicate = True
                            break
                    if not duplicate:
                        ring2_candidates.append(p2)

        ring2_candidates.sort(key=lambda p: np.linalg.norm(p))
        for i in range(5):
            centers.append(ring2_candidates[i].copy())

        return np.array(centers)

    def generate_random_config(seed=None):
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()

        positions = []
        angles = []
        for i in range(N):
            r = rng.uniform(0, 4.5)
            theta = rng.uniform(0, 2*math.pi)
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            positions.append((x, y))
            angles.append(rng.uniform(0, 2*math.pi))

        pos_arr = np.array(positions)
        pos_arr -= np.mean(pos_arr, axis=0)
        return pos_arr, np.array(angles)

    def optimize_sa(positions_init, angles_init, max_time=25.0):
        """Simulated annealing with smooth penalty."""
        sa_deadline = time.time() + max_time
        T = 0.5
        T_min = 0.001
        alpha = 0.985
        steps_per_temp = 15

        current_positions = positions_init.copy()
        current_angles = angles_init.copy()
        current_cost = smooth_objective(np.concatenate([current_positions.flatten(), current_angles]))

        best_positions = current_positions.copy()
        best_angles = current_angles.copy()
        best_cost = current_cost

        while T > T_min:
            for _ in range(steps_per_temp):
                if time.time() > sa_deadline or time.time() > deadline - 20:
                    break

                new_positions = current_positions.copy()
                new_angles = current_angles.copy()

                i = random.randint(0, N-1)
                step_size = T * 0.3
                new_positions[i, 0] += random.gauss(0, step_size)
                new_positions[i, 1] += random.gauss(0, step_size)
                new_angles[i] += random.gauss(0, step_size * 3)

                if random.random() < 0.03:
                    # Move multiple hexagons
                    for k in range(random.randint(2, 4)):
                        j = random.randint(0, N-1)
                        new_positions[j, 0] += random.gauss(0, T * 0.2)
                        new_positions[j, 1] += random.gauss(0, T * 0.2)

                new_cost = smooth_objective(np.concatenate([new_positions.flatten(), new_angles]))

                delta = new_cost - current_cost
                if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                    current_positions = new_positions
                    current_angles = new_angles
                    current_cost = new_cost

                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_positions = current_positions.copy()
                        best_angles = current_angles.copy()

            T *= alpha
            if time.time() > sa_deadline or time.time() > deadline - 20:
                break

        return best_positions, best_angles, best_cost

    def slsqp_refine(positions_init, angles_init, max_time=15.0):
        """SLSQP refinement with smooth objective."""
        if time.time() > deadline - 10:
            return positions_init, angles_init, smooth_objective(
                np.concatenate([positions_init.flatten(), angles_init]))

        x0 = np.concatenate([positions_init.flatten(), angles_init])

        def obj(v):
            if time.time() > deadline - 5:
                return 1e6
            pos = v[:2*N].reshape(N, 2)
            ang = v[2*N:]
            return smooth_objective(v)

        # Use SLSQP with bounds
        bounds = [(-10, 10)] * (2*N) + [(0, 2*math.pi)] * N

        result = minimize(obj, x0, method='SLSQP', bounds=bounds,
                          options={'maxiter': 200, 'ftol': 1e-8})

        if result.success or result.fun < smooth_objective(x0):
            new_pos = result.x[:2*N].reshape(N, 2)
            new_ang = result.x[2*N:]
            return new_pos, new_ang, result.fun
        return positions_init, angles_init, smooth_objective(x0)

    def polish_outer(positions, angles, max_time=5.0):
        """Polish the outer hexagon center and angle for the fixed inner configuration."""
        if time.time() > deadline - 3:
            return 0.0, 0.0, 0.0, calc_min_outer_side(positions, angles, 0.0)

        def outer_obj(v):
            if time.time() > deadline - 2:
                return 1e6
            # v = [ocx, ocy, otheta]
            otheta = v[2]
            # Compute max projection
            normals = []
            for k in range(6):
                ang = otheta + math.pi/6 + k * math.pi/3
                normals.append([math.cos(ang), math.sin(ang)])
            normals = np.array(normals)

            max_proj = 0.0
            for i in range(N):
                verts = get_hex_verts(positions[i, 0], positions[i, 1], 1.0, angles[i])
                # Project relative to outer center
                dx = verts[:, 0] - v[0]
                dy = verts[:, 1] - v[1]
                projs = dx * normals[:, 0] + dy * normals[:, 1]
                max_proj_i = np.max(projs)
                if max_proj_i > max_proj:
                    max_proj = max_proj_i

            return max_proj * 2.0 / sqrt3

        result = minimize(outer_obj, [0.0, 0.0, 0.0], method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10})

        ocx, ocy, otheta = result.x
        side = result.fun
        return ocx, ocy, otheta, side

    # Generate diverse initial configurations
    initial_configs = []

    # Honeycomb patterns
    for radius in [2, 3]:
        pos = generate_honeycomb_config(radius)
        pos -= np.mean(pos, axis=0)
        initial_configs.append((pos, np.zeros(N)))

    # Ring pattern
    pos = generate_ring_config()
    pos -= np.mean(pos, axis=0)
    initial_configs.append((pos, np.zeros(N)))

    # Lattice
    pos = generate_lattice_config()
    pos -= np.mean(pos, axis=0)
    initial_configs.append((pos, np.zeros(N)))

    # Random configurations
    for seed in range(8):
        pos, ang = generate_random_config(seed)
        initial_configs.append((pos, ang))

    best_overall_cost = float('inf')
    best_positions = None
    best_angles = None
    best_ocx = 0.0
    best_ocy = 0.0
    best_otheta = 0.0

    for idx, (pos_init, ang_init) in enumerate(initial_configs):
        if time.time() > deadline - 30:
            break

        # Phase 1: SA to find good configuration
        sa_pos, sa_ang, sa_cost = optimize_sa(pos_init, ang_init, max_time=20.0)

        if time.time() > deadline - 20:
            # Just use SA result
            if sa_cost < best_overall_cost:
                best_overall_cost = sa_cost
                best_positions = sa_pos
                best_angles = sa_ang
            continue

        # Phase 2: SLSQP refinement
        slq_pos, slq_ang, slq_cost = slsqp_refine(sa_pos, sa_ang, max_time=12.0)

        # Check validity
        if is_valid(slq_pos, slq_ang):
            if slq_cost < best_overall_cost:
                best_overall_cost = slq_cost
                best_positions = slq_pos
                best_angles = slq_ang
        else:
            # Use SA result if valid
            if is_valid(sa_pos, sa_ang):
                if sa_cost < best_overall_cost:
                    best_overall_cost = sa_cost
                    best_positions = sa_pos
                    best_angles = sa_ang

    # Fallback
    if best_positions is None:
        best_positions = generate_honeycomb_config(2)
        best_positions -= np.mean(best_positions, axis=0)
        best_angles = np.zeros(N)
        best_overall_cost = calc_min_outer_side(best_positions, best_angles, 0.0)

    # Phase 3: Polish outer hexagon
    best_ocx, best_ocy, best_otheta, best_side = polish_outer(best_positions, best_angles)

    # Phase 4: Final fine-tuning of inner positions with very small steps
    if time.time() < deadline - 5:
        fine_deadline = time.time() + 8.0
        T_fine = 0.02
        for _ in range(200):
            if time.time() > fine_deadline or time.time() > deadline - 3:
                break
            new_pos = best_positions.copy()
            new_ang = best_angles.copy()
            i = random.randint(0, N-1)
            new_pos[i, 0] += random.gauss(0, 0.01)
            new_pos[i, 1] += random.gauss(0, 0.01)
            new_ang[i] += random.gauss(0, 0.02)

            if is_valid(new_pos, new_ang):
                side_new = calc_min_outer_side(new_pos, new_ang, best_otheta)
                if side_new < best_side:
                    best_side = side_new
                    best_positions = new_pos
                    best_angles = new_ang
                    # Re-polish outer
                    best_ocx, best_ocy, best_otheta, best_side = polish_outer(
                        best_positions, best_angles, max_time=2.0)

    # Final validation
    if not is_valid(best_positions, best_angles):
        # Fall back to a known valid config
        best_positions = generate_honeycomb_config(2)
        best_positions -= np.mean(best_positions, axis=0)
        best_angles = np.zeros(N)
        best_ocx, best_ocy, best_otheta, best_side = polish_outer(best_positions, best_angles)

    inner = [[float(best_positions[i, 0]), float(best_positions[i, 1]),
              float(math.degrees(best_angles[i]))] for i in range(N)]
    outer_center = (float(best_ocx), float(best_ocy))
    outer_side = float(best_side)
    outer_angle = float(math.degrees(best_otheta))

    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
