"""Packing 12 unit hexagons in minimal regular hexagon."""
import math
import time
import numpy as np
from itertools import combinations

try:
    from scipy.optimize import minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

N = 12
SQRT3 = math.sqrt(3.0)

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

    sqrt3 = SQRT3

    # Generate all 19 centers of the side-3 hexagon cluster
    # Axial coordinates (q, r) with max(|q|, |r|, |q+r|) <= 2
    # Using Flat-Top hexagon grid coordinates
    all_centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                x = 1.5 * q
                y = sqrt3 * (r + q / 2.0)
                all_centers.append((x, y, q, r))

    n_centers = len(all_centers)  # 19

    # Normal directions for flat-top outer hexagon (pointing outward from center)
    # Normals at 30, 90, 150, 210, 270, 330 degrees
    normal_angles = np.array([math.radians(30 + 60 * j) for j in range(6)])
    normal_cos = np.cos(normal_angles)
    normal_sin = np.sin(normal_angles)

    def min_L_for_hexagon(cx, cy, theta):
        """Find the minimum L needed for a unit hexagon at (cx,cy) with rotation theta to fit in outer hex of side L."""
        vertex_angles = np.array([math.radians(60 * k) + theta for k in range(6)])
        vx = cx + np.cos(vertex_angles)
        vy = cy + np.sin(vertex_angles)

        projs = vx[:, np.newaxis] * normal_cos[np.newaxis, :] + vy[:, np.newaxis] * normal_sin[np.newaxis, :]
        max_proj = np.max(projs)
        return max_proj / (sqrt3 / 2.0)

    def optimal_theta_for_hexagon(cx, cy):
        """Find rotation theta that minimizes the required L for this hexagon."""
        best_L = float('inf')
        best_theta = 0.0
        # Coarse search: 0 to 60 degrees in 1 degree steps
        for t_deg in range(0, 61, 1):
            t = math.radians(t_deg)
            L = min_L_for_hexagon(cx, cy, t)
            if L < best_L:
                best_L = L
                best_theta = t
        # Refine with finer steps
        for step_deg in [0.1, 0.01, 0.001, 0.0001]:
            step = math.radians(step_deg)
            t_start = max(0.0, best_theta - step * 10)
            t_end = min(math.radians(60), best_theta + step * 10)
            for t_deg in np.arange(math.degrees(t_start), math.degrees(t_end) + step_deg, step_deg):
                t = math.radians(t_deg)
                L = min_L_for_hexagon(cx, cy, t)
                if L < best_L:
                    best_L = L
                    best_theta = t
        return best_L, best_theta

    # Small scale to ensure strict disjointness
    SCALE = 1.000001

    best_L = float('inf')
    best_subset = None
    best_thetas = None
    best_cluster_angle = 0.0

    # Search over cluster rotation angles
    # The cluster has 6-fold symmetry, so we only need to search 0 to 60 degrees
    cluster_angles_to_try = []
    # Coarse search
    for a_deg in range(0, 61, 5):
        cluster_angles_to_try.append(math.radians(a_deg))
    # Add key angles explicitly
    for a_deg in [15, 30, 45, 52.5, 7.5, 22.5, 37.5, 52.5]:
        cluster_angles_to_try.append(math.radians(a_deg))
    # Deduplicate
    cluster_angles_to_try = sorted(set(cluster_angles_to_try))

    for cluster_angle in cluster_angles_to_try:
        if time.time() > deadline - 15.0:
            break

        # Rotate all centers by cluster_angle
        rotated_centers = []
        cos_a = math.cos(cluster_angle)
        sin_a = math.sin(cluster_angle)
        for x, y, q, r in all_centers:
            rx = x * cos_a - y * sin_a
            ry = x * sin_a + y * cos_a
            rotated_centers.append((rx, ry, q, r))

        # Precompute optimal theta for each center (independent of subset)
        center_optimal = []
        for i in range(n_centers):
            x, y, _, _ = rotated_centers[i]
            L_h, theta_h = optimal_theta_for_hexagon(x * SCALE, y * SCALE)
            center_optimal.append((L_h, theta_h))

        # Sort centers by their individual L requirement
        sorted_indices = sorted(range(n_centers), key=lambda i: center_optimal[i][0])

        # Try all subsets of size 12 from 19 centers
        for subset in combinations(range(n_centers), 12):
            if time.time() > deadline - 15.0:
                break
            L = max(center_optimal[i][0] for i in subset)
            if L < best_L:
                best_L = L
                best_subset = subset
                best_thetas = [center_optimal[i][1] for i in subset]
                best_cluster_angle = cluster_angle
                # Early termination if we find a very good solution
                if best_L < 9.95:
                    break

    if best_subset is None:
        # Fallback: use first 12 with no cluster rotation
        best_cluster_angle = 0.0
        rotated_centers = all_centers
        center_optimal = []
        for i in range(n_centers):
            x, y, _, _ = rotated_centers[i]
            L_h, theta_h = optimal_theta_for_hexagon(x * SCALE, y * SCALE)
            center_optimal.append((L_h, theta_h))
        best_subset = tuple(range(12))
        best_thetas = [center_optimal[i][1] for i in best_subset]
        best_L = max(center_optimal[i][0] for i in best_subset)

    # Build initial positions from best discrete solution
    init_positions = np.zeros((N, 2))
    init_thetas = np.zeros(N)
    cos_a = math.cos(best_cluster_angle)
    sin_a = math.sin(best_cluster_angle)
    for idx, i in enumerate(best_subset):
        x, y, _, _ = all_centers[i]
        rx = x * cos_a - y * sin_a
        ry = x * sin_a + y * cos_a
        init_positions[idx, 0] = rx * SCALE
        init_positions[idx, 1] = ry * SCALE
        init_thetas[idx] = best_thetas[idx]

    # Smooth optimization using scipy if available
    if HAS_SCIPY and time.time() < deadline - 10.0:
        # Define the objective: minimize the maximum L needed
        def compute_L(positions, thetas):
            """Compute the outer side length L for given positions and rotations."""
            max_L = 0.0
            for i in range(N):
                cx, cy = positions[i]
                theta = thetas[i]
                L_i = min_L_for_hexagon(cx, cy, theta)
                if L_i > max_L:
                    max_L = L_i
            return max_L

        # Non-overlap check using support functions
        def hex_support(cx, cy, theta, phi):
            """Support function of a unit hexagon at (cx,cy) with rotation theta in direction phi."""
            max_val = 0.0
            for k in range(6):
                angle = theta + k * math.pi / 3.0
                vx = cx + math.cos(angle)
                vy = cy + math.sin(angle)
                proj = vx * math.cos(phi) + vy * math.sin(phi)
                if proj > max_val:
                    max_val = proj
            return max_val

        def min_separation(theta1, theta2):
            """
            Compute the minimum center-to-center distance for two unit hexagons
            with rotations theta1 and theta2 to be non-overlapping.
            Uses the Minkowski sum approach: check 12 critical directions.
            """
            dtheta = (theta2 - theta1) % (2 * math.pi)
            if dtheta > math.pi:
                dtheta = 2 * math.pi - dtheta
            dtheta = abs(dtheta)

            # Critical directions: union of edge normals of both hexagons
            # Hexagon 1 normals: theta1 + k*pi/3 for k=0..5
            # Hexagon 2 normals: theta2 + k*pi/3 for k=0..5
            # We need to check these 12 directions (or 6 if aligned)
            min_sep = float('inf')

            for k in range(6):
                phi1 = theta1 + k * math.pi / 3.0
                phi2 = theta2 + k * math.pi / 3.0

                for phi in [phi1, phi2]:
                    # Support of hex1 in direction phi
                    h1 = 0.0
                    for j in range(6):
                        angle = theta1 + j * math.pi / 3.0
                        proj = math.cos(angle) * math.cos(phi) + math.sin(angle) * math.sin(phi)
                        if proj > h1:
                            h1 = proj

                    # Support of hex2 in direction -phi (i.e., phi + pi)
                    h2 = 0.0
                    for j in range(6):
                        angle = theta2 + j * math.pi / 3.0
                        proj = math.cos(angle) * math.cos(phi + math.pi) + math.sin(angle) * math.sin(phi + math.pi)
                        if proj > h2:
                            h2 = proj

                    sep = h1 + h2
                    if sep < min_sep:
                        min_sep = sep

            return min_sep

        def overlap_violation(positions, thetas):
            """Compute the maximum overlap violation between any pair of hexagons."""
            max_violation = 0.0
            for i in range(N):
                for j in range(i + 1, N):
                    dx = positions[j, 0] - positions[i, 0]
                    dy = positions[j, 1] - positions[i, 1]
                    dist = math.sqrt(dx * dx + dy * dy)

                    min_sep = min_separation(thetas[i], thetas[j])
                    violation = min_sep - dist
                    if violation > max_violation:
                        max_violation = violation
            return max_violation

        # Objective: minimize L + penalty for overlaps
        def objective(params):
            positions = params[:2 * N].reshape(N, 2)
            thetas = params[2 * N:]

            L = compute_L(positions, thetas)
            ov = overlap_violation(positions, thetas)

            # Penalize overlaps heavily
            penalty = 100.0 * max(0.0, ov)
            return L + penalty

        # Initial parameters
        x0 = np.concatenate([init_positions.flatten(), init_thetas])

        # Run optimization
        try:
            result = minimize(objective, x0, method='L-BFGS-B',
                             options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})

            if result.success or result.fun < best_L:
                opt_params = result.x
                opt_positions = opt_params[:2 * N].reshape(N, 2)
                opt_thetas = opt_params[2 * N:]

                # Verify no overlaps
                ov = overlap_violation(opt_positions, opt_thetas)
                if ov < 1e-6:
                    opt_L = compute_L(opt_positions, opt_thetas)
                    if opt_L < best_L:
                        best_L = opt_L
                        for idx in range(N):
                            init_positions[idx] = opt_positions[idx]
                            init_thetas[idx] = opt_thetas[idx]
        except Exception:
            pass

    # Final high-precision polish: refine thetas with very fine steps
    final_positions = init_positions.copy()
    final_thetas = init_thetas.copy()

    # Do one more pass of theta refinement for each hexagon
    for i in range(N):
        if time.time() > deadline - 5.0:
            break
        cx, cy = final_positions[i]
        best_theta_i = final_thetas[i]
        best_L_i = min_L_for_hexagon(cx, cy, best_theta_i)

        # Very fine search
        for t_deg in np.arange(math.degrees(best_theta_i) - 0.01, math.degrees(best_theta_i) + 0.011, 0.001):
            t = math.radians(t_deg)
            L_i = min_L_for_hexagon(cx, cy, t)
            if L_i < best_L_i:
                best_L_i = L_i
                best_theta_i = t

        final_thetas[i] = best_theta_i

    # Compute final L
    final_L = 0.0
    for i in range(N):
        L_i = min_L_for_hexagon(final_positions[i, 0], final_positions[i, 1], final_thetas[i])
        if L_i > final_L:
            final_L = L_i

    # Construct inner hexagons
    inner = []
    for i in range(N):
        inner.append([final_positions[i, 0], final_positions[i, 1], math.degrees(final_thetas[i])])

    return inner, (0.0, 0.0), final_L, 0.0
