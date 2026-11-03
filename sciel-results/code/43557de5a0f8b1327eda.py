import numpy as np
import time
import math

N = 11

def construct():
    start_time = time.time()
    deadline = 145.0

    # Unit hexagon vertices (flat-top, vertex at angle 0)
    unit_hex_verts = np.array([
        [1.0, 0.0],
        [0.5, math.sqrt(3)/2],
        [-0.5, math.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -math.sqrt(3)/2],
        [0.5, -math.sqrt(3)/2]
    ])

    # Normals of unit hexagon (for flat-top, normals at 0, 60, 120, 180, 240, 300)
    unit_normals = np.array([
        [1.0, 0.0],
        [0.5, math.sqrt(3)/2],
        [-0.5, math.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -math.sqrt(3)/2],
        [0.5, -math.sqrt(3)/2]
    ])

    def get_hex_verts(x, y, angle_deg):
        angle_rad = math.radians(angle_deg)
        c, s = math.cos(angle_rad), math.sin(angle_rad)
        R = np.array([[c, -s], [s, c]])
        verts = unit_hex_verts @ R.T
        verts += np.array([x, y])
        return verts

    def get_hex_normals(x, y, angle_deg):
        angle_rad = math.radians(angle_deg)
        c, s = math.cos(angle_rad), math.sin(angle_rad)
        R = np.array([[c, -s], [s, c]])
        normals = unit_normals @ R.T
        return normals

    def calc_side(centers, angles, outer_rotation=0.0):
        """Calculate required outer hexagon side length."""
        angle_rad = math.radians(outer_rotation)
        c, s = math.cos(angle_rad), math.sin(angle_rad)
        R = np.array([[c, -s], [s, c]])
        normals = unit_normals @ R.T

        max_apothem = 0.0
        for k in range(N):
            verts = get_hex_verts(centers[k, 0], centers[k, 1], angles[k])
            for v in verts:
                for n in normals:
                    proj = float(np.dot(v, n))
                    if proj > max_apothem:
                        max_apothem = proj
        return 2.0 * max_apothem / math.sqrt(3)

    def hexagon_overlap_test(centers, angles, eps=1e-7):
        """Check if any two hexagons overlap using vertex-in-polygon test."""
        def point_in_hexagon(px, py, cx, cy, angle_deg):
            dx = px - cx
            dy = py - cy
            angle_rad = math.radians(-angle_deg)
            c, s = math.cos(angle_rad), math.sin(angle_rad)
            lx = c * dx - s * dy
            ly = s * dx + c * dy
            for k in range(6):
                v1 = unit_hex_verts[k]
                v2 = unit_hex_verts[(k + 1) % 6]
                ex = v2[0] - v1[0]
                ey = v2[1] - v1[1]
                cross = ex * (ly - v1[1]) - ey * (lx - v1[0])
                if cross > eps:
                    return False
            return True

        for i in range(N):
            for j in range(i + 1, N):
                verts_i = get_hex_verts(centers[i, 0], centers[i, 1], angles[i])
                verts_j = get_hex_verts(centers[j, 0], centers[j, 1], angles[j])
                for v in verts_i:
                    if point_in_hexagon(v[0], v[1], centers[j, 0], centers[j, 1], angles[j]):
                        return False
                for v in verts_j:
                    if point_in_hexagon(v[0], v[1], centers[i, 0], centers[i, 1], angles[i]):
                        return False
        return True

    def smooth_max(values, beta=50.0):
        """Smooth maximum using log-sum-exp."""
        max_val = np.max(values)
        if max_val < -100:
            return max_val
        exp_vals = np.exp((values - max_val) * beta)
        return max_val + math.log(np.sum(exp_vals)) / beta

    def objective_smooth(centers, angles, outer_rotation, S_target, beta=50.0):
        """
        Objective: smooth container violation + overlap penalty.
        Returns (objective_value, container_violation, overlap_penalty)
        """
        angle_rad = math.radians(outer_rotation)
        c, s = math.cos(angle_rad), math.sin(angle_rad)
        R = np.array([[c, -s], [s, c]])
        normals = unit_normals @ R.T

        # Container constraint: each vertex must satisfy dot(v, n) <= S*sqrt(3)/2 for all normals
        # Use smooth max of violations
        apothem_limit = S_target * math.sqrt(3) / 2
        container_violations = []
        for k in range(N):
            verts = get_hex_verts(centers[k, 0], centers[k, 1], angles[k])
            for v in verts:
                for n in normals:
                    proj = float(np.dot(v, n))
                    violation = proj - apothem_limit
                    if violation > 0:
                        container_violations.append(violation)
        
        if container_violations:
            container_viol = smooth_max(np.array(container_violations), beta)
        else:
            container_viol = 0.0

        # Overlap penalty: use center distance as proxy (fast)
        # Min center distance for non-overlapping unit hexagons is sqrt(3)
        # But this is only valid for aligned hexagons; for rotated, it's less
        # Use a conservative threshold
        min_dist = math.sqrt(3) - 0.1  # Slightly less conservative
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < min_dist:
                    overlap_penalty += (min_dist - dist) ** 2 * 500

        obj = container_viol * 100 + overlap_penalty
        return obj, container_viol, overlap_penalty

    def local_search(centers, angles, outer_rotation, S_target, max_iters=20000, step_init=0.15, beta=50.0):
        """Local search to find a valid packing within side length S_target."""
        current_centers = centers.copy()
        current_angles = angles.copy()
        current_outer_rot = outer_rotation
        current_obj, _, _ = objective_smooth(current_centers, current_angles, current_outer_rot, S_target, beta)
        best_centers = current_centers.copy()
        best_angles = current_angles.copy()
        best_outer_rot = current_outer_rot
        best_obj = current_obj
        step = step_init

        for it in range(max_iters):
            if time.time() > deadline - 3:
                break

            idx = np.random.randint(N)
            dx = np.random.normal(0, step)
            dy = np.random.normal(0, step)
            dangle = np.random.normal(0, step * 5)
            douter = np.random.normal(0, step * 2)

            new_centers = current_centers.copy()
            new_angles = current_angles.copy()
            new_outer_rot = current_outer_rot + douter

            new_centers[idx, 0] += dx
            new_centers[idx, 1] += dy
            new_angles[idx] += dangle

            new_obj, _, _ = objective_smooth(new_centers, new_angles, new_outer_rot, S_target, beta)

            if new_obj < current_obj:
                current_centers = new_centers
                current_angles = new_angles
                current_outer_rot = new_outer_rot
                current_obj = new_obj
                if current_obj < best_obj:
                    best_obj = current_obj
                    best_centers = current_centers.copy()
                    best_angles = current_angles.copy()
                    best_outer_rot = current_outer_rot
                    # If objective is very small, check if it's actually feasible
                    if best_obj < 1e-6:
                        if hexagon_overlap_test(best_centers, best_angles):
                            # Verify it fits in S_target
                            side = calc_side(best_centers, best_angles, best_outer_rot)
                            if side <= S_target + 1e-6:
                                return best_centers, best_angles, best_outer_rot, True

            step *= 0.9995

        # Final check
        if hexagon_overlap_test(best_centers, best_angles):
            side = calc_side(best_centers, best_angles, best_outer_rot)
            if side <= S_target + 1e-6:
                return best_centers, best_angles, best_outer_rot, True

        return best_centers, best_angles, best_outer_rot, False

    # Generate initial configurations
    np.random.seed(42)
    d = math.sqrt(3)

    initial_configs = []

    # Config 1: 7-hex cluster + 4 extensions
    centers1 = [(0, 0)]
    for k in range(6):
        angle = k * math.pi / 3
        centers1.append((d * math.cos(angle), d * math.sin(angle)))
    # Add 4 more around the cluster
    for k in [0, 1, 2, 3]:
        angle = k * math.pi / 3 + math.pi / 6
        x = 2 * d * math.cos(angle)
        y = 2 * d * math.sin(angle)
        centers1.append((x, y))
    centers1 = np.array(centers1[:N])
    angles1 = np.zeros(N)
    initial_configs.append((centers1, angles1))

    # Config 2: Two rows offset
    centers2 = []
    for i in range(6):
        centers2.append((i * d - 2.5 * d, 0))
    for i in range(5):
        centers2.append((i * d - 2.0 * d + d / 2, 1.5))
    centers2 = np.array(centers2[:N])
    angles2 = np.zeros(N)
    initial_configs.append((centers2, angles2))

    # Config 3: Linear
    centers3 = np.array([(i * d - (N - 1) * d / 2, 0) for i in range(N)])
    angles3 = np.zeros(N)
    initial_configs.append((centers3, angles3))

    # Config 4: Tighter hexagonal
    centers4 = [
        (-d / 2, 0), (d / 2, 0),
        (-d, 1.5), (0, 1.5), (d, 1.5),
        (-d / 2, 3.0), (d / 2, 3.0),
        (-d, -1.5), (0, -1.5), (d, -1.5),
        (0, -3.0)
    ]
    centers4 = np.array(centers4[:N])
    angles4 = np.zeros(N)
    initial_configs.append((centers4, angles4))

    # Config 5: Random
    centers5 = np.random.uniform(-3 * d, 3 * d, (N, 2))
    angles5 = np.random.uniform(-10, 10, N)
    initial_configs.append((centers5, angles5))

    # Config 6: 7-hex + 4 in a row
    centers6 = [(0, 0)]
    for k in range(6):
        angle = k * math.pi / 3
        centers6.append((d * math.cos(angle), d * math.sin(angle)))
    for k in range(4):
        centers6.append((3 * d + k * d, 0))
    centers6 = np.array(centers6[:N])
    angles6 = np.zeros(N)
    initial_configs.append((centers6, angles6))

    # Binary search for minimal side length
    low = 3.0
    high = 4.5

    # First, find a valid configuration at high
    best_side = high
    best_centers = None
    best_angles = None
    best_outer_rot = 0.0

    for iters in range(15):
        if time.time() > deadline - 5:
            break
        mid = (low + high) / 2

        found = False
        for config_idx, (init_centers, init_angles) in enumerate(initial_configs):
            if time.time() > deadline - 5:
                break
            if found:
                break

            for outer_rot in [0.0, 30.0]:
                result_centers, result_angles, result_outer_rot, feasible = local_search(
                    init_centers, init_angles, outer_rot, mid,
                    max_iters=8000, step_init=0.12, beta=50.0
                )

                if feasible:
                    side = calc_side(result_centers, result_angles, result_outer_rot)
                    if side <= mid + 1e-6:
                        high = mid
                        best_side = side
                        best_centers = result_centers.copy()
                        best_angles = result_angles.copy()
                        best_outer_rot = result_outer_rot
                        found = True
                        break

        if not found:
            low = mid

    # Polish pass: refine the best solution found
    if best_centers is not None and time.time() < deadline - 5:
        current_centers = best_centers.copy()
        current_angles = best_angles.copy()
        current_outer_rot = best_outer_rot
        current_side = calc_side(current_centers, current_angles, current_outer_rot)

        # Tight local search to minimize side directly
        def objective_direct(centers, angles, outer_rotation):
            side = calc_side(centers, angles, outer_rotation)
            # Overlap penalty
            min_dist = math.sqrt(3) - 0.05
            penalty = 0.0
            for i in range(N):
                for j in range(i + 1, N):
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist < min_dist:
                        penalty += (min_dist - dist) ** 2 * 1000
            return side + penalty

        current_obj = objective_direct(current_centers, current_angles, current_outer_rot)
        best_polish_obj = current_obj
        best_polish_centers = current_centers.copy()
        best_polish_angles = current_angles.copy()
        best_polish_outer_rot = current_outer_rot

        step_size = 0.03
        for it in range(10000):
            if time.time() > deadline - 2:
                break

            idx = np.random.randint(N)
            dx = np.random.normal(0, step_size)
            dy = np.random.normal(0, step_size)
            dangle = np.random.normal(0, step_size * 8)
            douter = np.random.normal(0, step_size * 4)

            new_centers = current_centers.copy()
            new_angles = current_angles.copy()
            new_outer_rot = current_outer_rot + douter

            new_centers[idx, 0] += dx
            new_centers[idx, 1] += dy
            new_angles[idx] += dangle

            new_obj = objective_direct(new_centers, new_angles, new_outer_rot)

            if new_obj < current_obj:
                current_centers = new_centers
                current_angles = new_angles
                current_outer_rot = new_outer_rot
                current_obj = new_obj
                if current_obj < best_polish_obj:
                    best_polish_obj = current_obj
                    best_polish_centers = current_centers.copy()
                    best_polish_angles = current_angles.copy()
                    best_polish_outer_rot = current_outer_rot

            step_size *= 0.999

        # Verify final solution
        if hexagon_overlap_test(best_polish_centers, best_polish_angles):
            final_side = calc_side(best_polish_centers, best_polish_angles, best_polish_outer_rot)
            if final_side < best_side:
                best_side = final_side
                best_centers = best_polish_centers
                best_angles = best_polish_angles
                best_outer_rot = best_polish_outer_rot

    # Construct output
    if best_centers is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    inner = []
    for i in range(N):
        x, y = best_centers[i, 0], best_centers[i, 1]
        ang = best_angles[i]
        inner.append([float(x), float(y), float(ang)])

    return inner, (0.0, 0.0), float(best_side), float(best_outer_rot)
