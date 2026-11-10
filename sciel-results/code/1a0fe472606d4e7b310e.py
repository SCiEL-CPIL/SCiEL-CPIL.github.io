import numpy as np
import time
from scipy.optimize import minimize

def hexagon_packing_12():
    start_time = time.time()
    deadline = 620.0  # Safety margin before 630s kill

    # Axial to cartesian for hexagonal grid
    def axial_to_cartesian(q, r):
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return x, y

    # Generate initial configurations (axial coordinates)
    def get_initial_configs():
        configs = []
        
        # Config 1: Star missing (2, -2)
        config_1 = [
            (0,0), 
            (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
            (2,0), (0,2), (-2,2), (-2,0), (0,-2)
        ]
        configs.append(config_1)
        
        # Config 2: Star missing (2, 0)
        config_2 = [
            (0,0), 
            (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
            (0,2), (-2,2), (-2,0), (0,-2), (2,-2)
        ]
        configs.append(config_2)
        
        # Config 3: Cap (5 consecutive in ring 2)
        config_3 = [
            (0,0), 
            (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
            (2,0), (1,1), (0,2), (-1,2), (-2,2)
        ]
        configs.append(config_3)
        
        # Config 4: Another variant
        config_4 = [
            (0,0), 
            (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
            (2,0), (1,1), (0,2), (-1,2), (0,-2)
        ]
        configs.append(config_4)
        
        return configs

    def compute_s(centers, theta_deg):
        """Compute the minimum side length of the containing hexagon."""
        theta = np.deg2rad(theta_deg)
        all_vertices = []
        for cx, cy in centers:
            angles = np.arange(6) * np.pi / 3
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

    def compute_s_fast(centers, theta):
        """Fast vectorized computation of containing hexagon side length."""
        # All vertices of all hexagons
        angles = np.arange(6) * np.pi / 3
        offsets = np.column_stack((np.cos(angles), np.sin(angles)))  # (6, 2)
        
        # centers: (12, 2), offsets: (6, 2)
        # vertices: (12, 6, 2)
        vertices = centers[:, np.newaxis, :] + offsets[np.newaxis, :, :]
        
        # Normals for the outer hexagon
        normal_angles = theta + np.arange(6) * np.pi / 3
        normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))  # (6, 2)
        
        # Project all vertices onto all normals: (12*6, 6)
        proj = vertices.reshape(-1, 2) @ normals.T  # (72, 6)
        
        # For each normal, get max and min projection
        max_proj = np.max(proj, axis=0)  # (6,)
        min_proj = np.min(proj, axis=0)  # (6,)
        widths = max_proj - min_proj  # (6,)
        
        max_width = np.max(widths)
        return max_width / np.sqrt(3)

    def objective(params):
        """Objective function for optimization."""
        centers = params[:24].reshape(12, 2)
        theta = params[24]
        
        # Check separation constraints
        for i in range(12):
            for j in range(i + 1, 12):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist_sq = dx * dx + dy * dy
                
                # Minimum distance squared for non-overlapping unit hexagons is 3
                if dist_sq < 3.0 - 1e-5:
                    return 1e6  # Large penalty for overlap
        
        return compute_s_fast(centers, theta)

    def optimize_config(config, maxiter=2000):
        """Optimize a single configuration using Nelder-Mead."""
        centers = np.array([axial_to_cartesian(q, r) for q, r in config])
        initial_params = np.zeros(25)
        initial_params[:24] = centers.flatten()
        initial_params[24] = 0.0  # Initial angle
        
        try:
            res = minimize(
                objective,
                initial_params,
                method='Nelder-Mead',
                options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            
            if res.success:
                return res.fun, res.x
            else:
                return res.fun, res.x
        except Exception as e:
            return None, None

    # Get initial configurations
    configs = get_initial_configs()
    
    best_s = float('inf')
    best_params = None
    
    for config in configs:
        if time.time() > deadline:
            break
            
        s, params = optimize_config(config)
        
        if s is not None and s < best_s:
            best_s = s
            best_params = params

    # If no config found, fallback to simple grid (should not happen)
    if best_params is None:
        # Fallback to seed-like solution
        inner_hex_data = np.array([
            [0, 0, 0], [-2.5, 0, 0], [2.5, 0, 0],
            [-1.25, 2.17, 0], [1.25, 2.17, 0], [-1.25, -2.17, 0],
            [1.25, -2.17, 0], [-3.75, 2.17, 0], [3.75, 2.17, 0],
            [-3.75, -2.17, 0], [3.75, -2.17, 0], [0, -4, 0]
        ])
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8
        return inner_hex_data, outer_hex_data, outer_hex_side_length

    # Extract results
    centers = best_params[:24].reshape(12, 2)
    theta = best_params[24]
    theta_deg = np.rad2deg(theta) % 360.0

    # Shift centers so their centroid is at origin
    centroid = np.mean(centers, axis=0)
    shifted_centers = centers - centroid

    inner_hex_data = np.zeros((12, 3))
    for i, (cx, cy) in enumerate(shifted_centers):
        inner_hex_data[i] = [cx, cy, 0]

    outer_hex_data = np.array([0, 0, theta_deg])
    outer_hex_side_length = best_s

    return inner_hex_data, outer_hex_data, outer_hex_side_length
