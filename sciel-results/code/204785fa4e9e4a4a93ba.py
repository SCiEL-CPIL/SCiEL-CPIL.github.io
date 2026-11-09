"""Combined approach: combinatorial grid search + continuous rotation refinement."""
import math
import time
import numpy as np
from itertools import combinations

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    sqrt3 = math.sqrt(3.0)

    # Generate all 19 centers of the side-3 hexagon cluster
    # Axial coordinates (q, r) with max(|q|, |r|, |q+r|) <= 2
    all_centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                # Convert axial to cartesian (flat-top hexagons)
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

        # Projection of each vertex onto each normal direction
        projs = vx[:, np.newaxis] * normal_cos[np.newaxis, :] + vy[:, np.newaxis] * normal_sin[np.newaxis, :]
        max_proj = np.max(projs)
        return max_proj / (sqrt3 / 2.0)

    def optimal_theta_for_hexagon(cx, cy):
        """Find rotation theta (radians) that minimizes the required L for this hexagon."""
        best_L = float('inf')
        best_theta = 0.0
        # Sample theta from 0 to 60 degrees (period is 60 degrees due to hexagonal symmetry)
        for t_deg in range(0, 61, 1):
            t = math.radians(t_deg)
            L = min_L_for_hexagon(cx, cy, t)
            if L < best_L:
                best_L = L
                best_theta = t
        # Refine around best
        for _ in range(2):
            step = 0.1 if _ == 0 else 0.01
            for t_deg in np.arange(best_theta - step*10, best_theta + step*10 + step, step):
                t = math.radians(t_deg)
                L = min_L_for_hexagon(cx, cy, t)
                if L < best_L:
                    best_L = L
                    best_theta = t
        return best_L, best_theta

    # Small scale to ensure strict non-overlap
    SCALE = 1.000001

    best_L = float('inf')
    best_subset = None
    best_thetas = None
    best_cluster_angle = 0.0

    # Search over cluster rotation angles
    cluster_angles_to_try = []
    for a_deg in range(0, 61, 3):
        cluster_angles_to_try.append(math.radians(a_deg))
    cluster_angles_to_try.append(math.radians(30))

    for cluster_angle in cluster_angles_to_try:
        if time.time() > deadline:
            break

        # Rotate all centers by cluster_angle
        rotated_centers = []
        for x, y, q, r in all_centers:
            rx = x * math.cos(cluster_angle) - y * math.sin(cluster_angle)
            ry = x * math.sin(cluster_angle) + y * math.cos(cluster_angle)
            rotated_centers.append((rx, ry, q, r))

        # Precompute optimal theta for each center (independent of subset)
        center_optimal = []
        for i in range(19):
            x, y, _, _ = rotated_centers[i]
            L_h, theta_h = optimal_theta_for_hexagon(x * SCALE, y * SCALE)
            center_optimal.append((L_h, theta_h))

        # Try all subsets of size 12 from 19 centers
        for subset in combinations(range(19), 12):
            if time.time() > deadline:
                break
            L = max(center_optimal[i][0] for i in subset)
            if L < best_L:
                best_L = L
                best_subset = subset
                best_thetas = [center_optimal[i][1] for i in subset]
                best_cluster_angle = cluster_angle

    if best_subset is None:
        # Fallback: use first 12 with no cluster rotation
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

    # Construct initial inner hexagons positions
    initial_positions = []
    for idx, i in enumerate(best_subset):
        x, y, _, _ = all_centers[i]
        rx = x * math.cos(best_cluster_angle) - y * math.sin(best_cluster_angle)
        ry = x * math.sin(best_cluster_angle) + y * math.cos(best_cluster_angle)
        initial_positions.append([rx * SCALE, ry * SCALE])

    # Continuous refinement using scipy to polish positions and rotations
    try:
        from scipy.optimize import minimize

        n_hex = len(best_subset)

        def objective(params):
            """Minimize the required outer hexagon side length."""
            positions = params[:2*n_hex].reshape(n_hex, 2)
            rotations = params[2*n_hex:]

            L = 0.0
            for i in range(n_hex):
                L = max(L, min_L_for_hexagon(positions[i, 0], positions[i, 1], rotations[i]))
            return L

        # Initial parameters
        x0 = np.zeros(3 * n_hex)
        for i in range(n_hex):
            x0[2*i] = initial_positions[i][0]
            x0[2*i+1] = initial_positions[i][1]
            x0[2*n_hex+i] = best_thetas[i]

        # Run optimization with time limit
        opt_start = time.time()
        result = minimize(
            objective, x0, method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
        )

        if time.time() < deadline - 5:
            # Second pass with tighter tolerance
            result = minimize(
                objective, result.x, method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )

        # Extract refined solution
        if result.success or result.fun < best_L:
            refined_positions = result.x[:2*n_hex].reshape(n_hex, 2)
            refined_rotations = result.x[2*n_hex:]
            refined_L = result.fun

            if refined_L < best_L:
                best_L = refined_L
                for i in range(n_hex):
                    initial_positions[i] = [refined_positions[i, 0], refined_positions[i, 1]]
                    best_thetas[i] = refined_rotations[i]

    except Exception:
        pass  # If optimization fails, keep the discrete solution

    # Construct final inner hexagons
    inner = []
    for idx in range(len(best_subset)):
        inner.append([initial_positions[idx][0], initial_positions[idx][1], math.degrees(best_thetas[idx])])

    return inner, (0.0, 0.0), best_L, 0.0
# EVOLVE-BLOCK-END
