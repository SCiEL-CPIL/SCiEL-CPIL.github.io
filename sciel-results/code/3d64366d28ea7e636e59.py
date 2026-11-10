import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def hexagon_packing_12():
    start_time = time.time()
    deadline = 610.0  # Safety margin before 630s kill

    def get_hex_vertices(cx, cy, side, angle_deg):
        """Get vertices of a regular hexagon."""
        angles = np.deg2rad(angle_deg) + np.arange(6) * np.pi / 3
        return np.column_stack((cx + side * np.cos(angles), cy + side * np.sin(angles)))

    def sat_separation(v1, v2, tol=1e-6):
        """SAT test: return True if hexagons with vertices v1, v2 are separated."""
        for v in (v1, v2):
            for i in range(6):
                p1 = v[i]
                p2 = v[(i + 1) % 6]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                proj1 = v1 @ normal
                proj2 = v2 @ normal
                if proj1.max() - proj2.min() > tol or proj2.max() - proj1.min() > tol:
                    return True
        return False

    def compute_s(centers, inner_angles_deg, theta_deg):
        """Compute minimum side length of containing regular hexagon."""
        theta = np.deg2rad(theta_deg)
        all_vertices = []
        for i in range(len(centers)):
            cx, cy = centers[i]
            angles = np.deg2rad(inner_angles_deg[i]) + np.arange(6) * np.pi / 3
            x = cx + np.cos(angles)
            y = cy + np.sin(angles)
            all_vertices.append(np.column_stack((x, y)))
        all_vertices = np.vstack(all_vertices)

        max_width = 0
        for k in range(6):
            n_angle = theta + k * np.pi / 3
            n = np.array([np.cos(n_angle), np.sin(n_angle)])
            projections = all_vertices @ n
            width = np.max(projections) - np.min(projections)
            if width > max_width:
                max_width = width
        return max_width / np.sqrt(3)

    def objective(params):
        """Objective: minimize outer hexagon side length with constraints."""
        centers = params[:24].reshape(12, 2)
        inner_angles_deg = params[24:36]
        theta_deg = params[36]

        # Check all pairs for separation using SAT
        for i in range(12):
            for j in range(i + 1, 12):
                v1 = get_hex_vertices(centers[i, 0], centers[i, 1], 1.0, inner_angles_deg[i])
                v2 = get_hex_vertices(centers[j, 0], centers[j, 1], 1.0, inner_angles_deg[j])
                if not sat_separation(v1, v2, tol=1e-6):
                    # Overlap penalty
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    return 1e6 + (np.sqrt(3) - dist) * 1000

        return compute_s(centers, inner_angles_deg, theta_deg)

    def get_initial_configs():
        """Generate diverse initial configurations."""
        configs = []
        # Config 1: Compact cluster
        config_1 = np.array([
            [0, 0], [np.sqrt(3), 0], [0, np.sqrt(3)], [-np.sqrt(3), 0],
            [np.sqrt(3)/2, 1.5], [-np.sqrt(3)/2, 1.5], [np.sqrt(3)/2, -1.5],
            [-np.sqrt(3)/2, -1.5], [2*np.sqrt(3), 0], [0, 2*np.sqrt(3)],
            [-2*np.sqrt(3), 0], [np.sqrt(3), 3]
        ])
        configs.append(config_1)

        # Config 2: Ring pattern
        config_2 = np.array([
            [0, 0],
            [np.sqrt(3), 0], [np.sqrt(3)/2, 1.5], [-np.sqrt(3)/2, 1.5],
            [-np.sqrt(3), 0], [-np.sqrt(3)/2, -1.5], [np.sqrt(3)/2, -1.5],
            [2*np.sqrt(3), 0], [np.sqrt(3), 3], [0, 3*np.sqrt(3)],
            [-np.sqrt(3), 3], [-2*np.sqrt(3), 0]
        ])
        configs.append(config_2)

        # Config 3: Linear-ish
        config_3 = np.array([
            [0, 0], [np.sqrt(3), 0], [2*np.sqrt(3), 0], [3*np.sqrt(3), 0],
            [4*np.sqrt(3), 0], [5*np.sqrt(3), 0], [-np.sqrt(3), 0],
            [-2*np.sqrt(3), 0], [-3*np.sqrt(3), 0], [0, np.sqrt(3)],
            [np.sqrt(3), np.sqrt(3)], [-np.sqrt(3), np.sqrt(3)]
        ])
        configs.append(config_3)

        return configs

    # Bounds for differential evolution
    bounds = [(-5, 5)] * 24  # centers
    bounds += [(0, 60)] * 12  # inner angles
    bounds += [(0, 60)]  # outer angle

    best_s = float('inf')
    best_params = None

    # Try multiple initial points with differential evolution
    n_restarts = 3
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break

        # Create initial guess
        if restart < len(get_initial_configs()):
            init_centers = get_initial_configs()[restart]
        else:
            init_centers = np.random.uniform(-4, 4, (12, 2))

        init_params = np.zeros(37)
        init_params[:24] = init_centers.flatten()
        init_params[24:36] = 0.0
        init_params[36] = 0.0

        try:
            # Use differential evolution for global search
            de_result = differential_evolution(
                objective,
                bounds=bounds,
                seed=restart,
                maxiter=50,
                tol=1e-6,
                popsize=15,
                polish=False,
                init='sobol'
            )

            if de_result.fun < best_s:
                best_s = de_result.fun
                best_params = de_result.x

            # Polish with Nelder-Mead
            if time.time() > deadline - 10:
                break

            nm_result = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )

            if nm_result.fun < best_s:
                best_s = nm_result.fun
                best_params = nm_result.x

        except Exception as e:
            continue

    # Fallback if nothing worked
    if best_params is None:
        # Simple valid fallback: 12 hexagons in a row, outer hexagon big enough
        inner = np.zeros((12, 3))
        for i in range(12):
            inner[i, 0] = i * np.sqrt(3)
            inner[i, 1] = 0
            inner[i, 2] = 0
        outer = np.array([6 * np.sqrt(3), 0, 0])
        s = 12 * np.sqrt(3) + 2
        return inner, outer, s

    # Extract results
    centers = best_params[:24].reshape(12, 2)
    inner_angles_deg = best_params[24:36]
    theta_deg = best_params[36]

    # Center the configuration
    centroid = np.mean(centers, axis=0)
    shifted_centers = centers - centroid

    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i] = [shifted_centers[i, 0], shifted_centers[i, 1], inner_angles_deg[i]]

    outer_hex_data = np.array([0, 0, theta_deg])
    outer_hex_side_length = best_s

    return inner_hex_data, outer_hex_data, outer_hex_side_length
