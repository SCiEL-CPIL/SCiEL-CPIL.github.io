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

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    sqrt3 = math.sqrt(3.0)

    # Generate all 19 centers of the side-3 hexagon cluster
    all_centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                x = 1.5 * q
                y = sqrt3 * (r + q / 2.0)
                all_centers.append((x, y, q, r))

    # Normal directions for flat-top outer hexagon
    normal_angles = [math.radians(30 + 60 * j) for j in range(6)]
    normal_cos = np.array([math.cos(a) for a in normal_angles])
    normal_sin = np.array([math.sin(a) for a in normal_angles])

    def min_L_for_hexagon(cx, cy, theta):
        """Find the minimum L needed for a unit hexagon at (cx,cy) with rotation theta to fit in outer hex of side L."""
        vertex_angles = [math.radians(60 * k) + theta for k in range(6)]
        vx = cx + np.array([math.cos(a) for a in vertex_angles])
        vy = cy + np.array([math.sin(a) for a in vertex_angles])
        projs = vx[:, np.newaxis] * normal_cos[np.newaxis, :] + vy[:, np.newaxis] * normal_sin[np.newaxis, :]
        max_proj = np.max(projs)
        return max_proj / (sqrt3 / 2.0)

    def min_L_for_hexagon_fast(cx, cy, theta):
        """Vectorized version for speed."""
        vertex_angles = np.radians(60 * np.arange(6)) + theta
        vx = cx + np.cos(vertex_angles)
        vy = cy + np.sin(vertex_angles)
        projs = vx[:, np.newaxis] * normal_cos[np.newaxis, :] + vy[:, np.newaxis] * normal_sin[np.newaxis, :]
        return np.max(projs) / (sqrt3 / 2.0)

    def optimal_theta_for_hexagon(cx, cy):
        """Find rotation theta that minimizes the required L for this hexagon."""
        best_L = float('inf')
        best_theta = 0.0
        # Coarse search
        for t_deg in range(0, 61, 2):
            t = math.radians(t_deg)
            L = min_L_for_hexagon(cx, cy, t)
            if L < best_L:
                best_L = L
                best_theta = t
        # Fine search
        for step in [1.0, 0.1, 0.01, 0.001]:
            lo = max(0.0, best_theta - step * 5)
            hi = min(math.radians(60), best_theta + step * 5)
            for t_deg in np.arange(lo, hi + step, step):
                t = math.radians(t_deg)
                L = min_L_for_hexagon(cx, cy, t)
                if L < best_L:
                    best_L = L
                    best_theta = t
        return best_L, best_theta

    # Small scale to ensure strict non-overlap
    SCALE = 1.0000001

    best_L = float('inf')
    best_subset = None
    best_thetas = None
    best_cluster_angle = 0.0

    # Search over cluster rotation angles
    cluster_angles_to_try = []
    for a_deg in range(0, 61, 5):
        cluster_angles_to_try.append(math.radians(a_deg))
    for a_deg in [15, 30, 45, 52.5]:
        cluster_angles_to_try.append(math.radians(a_deg))
    cluster_angles_to_try = sorted(set(cluster_angles_to_try))

    for cluster_angle in cluster_angles_to_try:
        if time.time() > deadline:
            break

        rotated_centers = []
        for x, y, q, r in all_centers:
            rx = x * math.cos(cluster_angle) - y * math.sin(cluster_angle)
            ry = x * math.sin(cluster_angle) + y * math.cos(cluster_angle)
            rotated_centers.append((rx, ry, q, r))

        center_optimal = []
        for i in range(19):
            x, y, _, _ = rotated_centers[i]
            L_h, theta_h = optimal_theta_for_hexagon(x * SCALE, y * SCALE)
            center_optimal.append((L_h, theta_h))

        sorted_indices = sorted(range(19), key=lambda i: center_optimal[i][0])

        # Only need to check subsets where the max L could beat current best
        # Prune: if the 12th smallest L is >= best_L, no subset can be better
        # But we still need to enumerate to find the actual best subset
        for subset in combinations(range(19), 12):
            if time.time() > deadline:
                break
            L = max(center_optimal[i][0] for i in subset)
            if L < best_L:
                best_L = L
                best_subset = subset
                best_thetas = [center_optimal[i][1] for i in subset]
                best_cluster_angle = cluster_angle
                if best_L < 9.9:
                    break

    if best_subset is None:
        best_cluster_angle = 0.0
        rotated_centers = all_centers
        center_optimal = []
        for i in range(19):
            x, y, _, _ = rotated_centers[i]
            L_h, theta_h = optimal_theta_for_hexagon(x * SCALE, y * SCALE)
            center_optimal.append((L_h, theta_h))
        best_subset = tuple(range(12))
        best_thetas = [center_optimal[i][1] for i in best_subset]
        best_L = max(center_optimal[i][0] for i in best_subset)

    # Build initial positions
    init_positions = np.zeros((N, 2))
    init_thetas = np.zeros(N)
    for idx, i in enumerate(best_subset):
        x, y, _, _ = all_centers[i]
        rx = x * math.cos(best_cluster_angle) - y * math.sin(best_cluster_angle)
        ry = x * math.sin(best_cluster_angle) + y * math.cos(best_cluster_angle)
        init_positions[idx, 0] = rx * SCALE
        init_positions[idx, 1] = ry * SCALE
        init_thetas[idx] = best_thetas[idx]

    # Smooth optimization using scipy
    if HAS_SCIPY and time.time() < deadline - 10.0:
        # Precompute support function samples for faster min_separation
        # For hexagon with rotation theta, support in direction phi:
        # h(phi, theta) = max_k cos(phi - theta - k*pi/3)
        # We precompute for theta=0 and use relative rotation
        
        def min_separation(dtheta):
            """Compute minimum center-to-center distance for two unit hexagons with relative rotation dtheta."""
            dtheta = abs(dtheta) % (2 * math.pi)
            dtheta = min(dtheta, 2 * math.pi - dtheta)
            
            # Due to hexagonal symmetry, we only need phi in [0, pi/6]
            # and dtheta mod pi/6
            dtheta_mod = dtheta % (math.pi / 6)
            if dtheta_mod > math.pi / 12:
                dtheta_mod = math.pi / 6 - dtheta_mod
            
            min_sep = float('inf')
            # Sample directions - use finer grid for accuracy
            for phi_deg in range(0, 31, 1):  # 0 to 30 degrees, 1 degree steps
                phi = math.radians(phi_deg)
                h1 = max(math.cos(phi - math.radians(60 * k)) for k in range(6))
                h2 = max(math.cos(phi + math.pi - dtheta - math.radians(60 * k)) for k in range(6))
                sep = h1 + h2
                if sep < min_sep:
                    min_sep = sep
            return min_sep

        # Precompute min_separation for a range of relative rotations
        # dtheta will be in [0, pi/3] due to symmetry
        sep_table_size = 361  # 0 to 60 degrees in 0.174 degree steps
        sep_table = np.zeros(sep_table_size)
        for i in range(sep_table_size):
            dtheta = math.radians(i * 60.0 / (sep_table_size - 1))
            sep_table[i] = min_separation(dtheta)

        def get_min_sep(dtheta):
            """Interpolate from precomputed table."""
            dtheta = abs(dtheta) % (2 * math.pi)
            dtheta = min(dtheta, 2 * math.pi - dtheta)
            # Map to [0, 60 degrees]
            dtheta_deg = math.degrees(dtheta) % 60.0
            idx = dtheta_deg / 60.0 * (sep_table_size - 1)
            i0 = int(idx)
            i1 = min(i0 + 1, sep_table_size - 1)
            frac = idx - i0
            return sep_table[i0] * (1 - frac) + sep_table[i1] * frac

        def smooth_objective(params):
            L = params[-1]
            positions = params[:2 * N].reshape(N, 2)
            thetas = params[2 * N:3 * N]

            L_required = np.zeros(N)
            for i in range(N):
                L_required[i] = min_L_for_hexagon_fast(positions[i, 0], positions[i, 1], thetas[i])

            beta = 30.0
            max_lr = np.max(L_required)
            if max_lr > 0:
                soft_max = max_lr + math.log(np.sum(np.exp(beta * (L_required - max_lr)))) / beta
            else:
                soft_max = 0.0

            overlap_penalty = 0.0
            for i in range(N):
                for j in range(i + 1, N):
                    dx = positions[i, 0] - positions[j, 0]
                    dy = positions[i, 1] - positions[j, 1]
                    d = math.sqrt(dx * dx + dy * dy)
                    dtheta = abs(thetas[i] - thetas[j])
                    min_dist = get_min_sep(dtheta)
                    if d < min_dist:
                        deficit = min_dist - d
                        overlap_penalty += deficit * deficit * 50.0

            containment_penalty = 0.0
            for i in range(N):
                if L_required[i] > L:
                    deficit = L_required[i] - L
                    containment_penalty += deficit * deficit * 50.0

            return soft_max + overlap_penalty + containment_penalty

        L_init = best_L * 1.005
        init_params = np.concatenate([init_positions.flatten(), init_thetas, [L_init]])

        best_opt_L = float('inf')
        best_opt_params = None

        methods_to_try = ['Nelder-Mead', 'Powell']

        for method in methods_to_try:
            if time.time() > deadline - 5.0:
                break

            try:
                if method == 'Nelder-Mead':
                    result = minimize(
                        smooth_objective,
                        init_params,
                        method='Nelder-Mead',
                        options={'maxiter': 800, 'xatol': 1e-7, 'fatol': 1e-9}
                    )
                elif method == 'Powell':
                    result = minimize(
                        smooth_objective,
                        init_params,
                        method='Powell',
                        options={'maxiter': 400, 'xtol': 1e-7, 'ftol': 1e-9}
                    )

                if result.success or result.fun < smooth_objective(init_params):
                    opt_positions = result.x[:2 * N].reshape(N, 2)
                    opt_thetas = result.x[2 * N:3 * N]

                    actual_max_L = 0.0
                    for i in range(N):
                        actual_max_L = max(actual_max_L, min_L_for_hexagon_fast(opt_positions[i, 0], opt_positions[i, 1], opt_thetas[i]))

                    valid = True
                    for i in range(N):
                        for j in range(i + 1, N):
                            dx = opt_positions[i, 0] - opt_positions[j, 0]
                            dy = opt_positions[i, 1] - opt_positions[j, 1]
                            d = math.sqrt(dx * dx + dy * dy)
                            dtheta = abs(opt_thetas[i] - opt_thetas[j])
                            min_dist = get_min_sep(dtheta)
                            if d < min_dist - 1e-8:
                                valid = False
                                break
                        if not valid:
                            break

                    if valid and actual_max_L < best_opt_L:
                        best_opt_L = actual_max_L
                        best_opt_params = result.x.copy()

            except Exception:
                pass

        if best_opt_params is not None and best_opt_L < best_L:
            best_L = best_opt_L
            for i in range(N):
                init_positions[i, 0] = best_opt_params[2 * i]
                init_positions[i, 1] = best_opt_params[2 * i + 1]
                init_thetas[i] = best_opt_params[2 * N + i]

        # Final polish
        if HAS_SCIPY and time.time() < deadline - 3.0 and best_opt_params is not None:
            try:
                result = minimize(
                    smooth_objective,
                    best_opt_params,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-9, 'fatol': 1e-11}
                )

                if result.fun < smooth_objective(best_opt_params):
                    opt_positions = result.x[:2 * N].reshape(N, 2)
                    opt_thetas = result.x[2 * N:3 * N]

                    actual_max_L = 0.0
                    for i in range(N):
                        actual_max_L = max(actual_max_L, min_L_for_hexagon_fast(opt_positions[i, 0], opt_positions[i, 1], opt_thetas[i]))

                    valid = True
                    for i in range(N):
                        for j in range(i + 1, N):
                            dx = opt_positions[i, 0] - opt_positions[j, 0]
                            dy = opt_positions[i, 1] - opt_positions[j, 1]
                            d = math.sqrt(dx * dx + dy * dy)
                            dtheta = abs(opt_thetas[i] - opt_thetas[j])
                            min_dist = get_min_sep(dtheta)
                            if d < min_dist - 1e-9:
                                valid = False
                                break
                        if not valid:
                            break

                    if valid and actual_max_L < best_L:
                        best_L = actual_max_L
                        for i in range(N):
                            init_positions[i, 0] = opt_positions[i, 0]
                            init_positions[i, 1] = opt_positions[i, 1]
                            init_thetas[i] = opt_thetas[i]
            except Exception:
                pass

    inner = []
    for idx in range(N):
        inner.append([init_positions[idx, 0], init_positions[idx, 1], math.degrees(init_thetas[idx])])

    return inner, (0.0, 0.0), best_L, 0.0
# EVOLVE-BLOCK-END
