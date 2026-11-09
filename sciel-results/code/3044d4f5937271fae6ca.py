import numpy as np
import time
from scipy.optimize import minimize

def construct():
    N = 200
    start_time = time.time()
    deadline = 110.0  # Safety margin

    def compute_score(centers):
        if not np.all(np.isfinite(centers)):
            return -np.inf
        # Check bounds
        if np.any(centers < -0.5 - 1e-12) or np.any(centers > 0.5 + 1e-12):
            return -np.inf
        
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        # Wall distances
        wall_dists = np.minimum(
            0.5 - np.abs(centers[:, 0]), 0.5 - np.abs(centers[:, 1])
        )
        min_wall = np.min(wall_dists)
        
        return min(min_pair, min_wall)

    def energy_and_grad(x, p=4.0, eps=1e-6):
        """
        Minimize the sum of inverse powers of distances.
        This encourages spreading points apart, focusing on the closest ones.
        """
        centers = x.reshape(N, 2)
        
        # Pairwise
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Clip distances to avoid division by zero/inf
        dists_clipped = np.maximum(dists, eps)
        
        # Energy term: sum(d^-p)
        # We only need to sum i<j to avoid double counting, but for gradient 
        # symmetry it's easier to sum all i!=j and divide by 2, or just sum all.
        # Let's sum all i != j.
        energy_pairs = np.sum(np.power(dists_clipped, -p))
        
        # Wall distances
        wall_dists = np.minimum(
            0.5 - np.abs(centers[:, 0]), 0.5 - np.abs(centers[:, 1])
        )
        wall_dists_clipped = np.maximum(wall_dists, eps)
        energy_walls = np.sum(np.power(wall_dists_clipped, -p))
        
        total_energy = energy_pairs + energy_walls
        
        # Gradient
        grads = np.zeros_like(centers)
        
        # Pairwise gradient
        # d/dx_i sum_j d_ij^-p = sum_j -p * d_ij^(-p-1) * d(d_ij)/dx_i
        # d(d_ij)/dx_i = (x_i - x_j) / d_ij
        # So term is -p * d_ij^(-p-2) * (x_i - x_j)
        
        # dists_clipped has inf on diagonal, so power(-p-2) will be 0 there?
        # inf^-k is 0. So diagonal is safe.
        # But we need to be careful with the vector calculation.
        
        # Factor: -p * d_ij^(-p-2)
        # Note: d_ij = ||x_i - x_j||
        # grad_i += sum_{j!=i} -p * d_ij^(-p-2) * (x_i - x_j)
        
        # To do this efficiently:
        # weights = -p * dists_clipped^(-p-2)
        # Set diagonal to 0
        weights = -p * np.power(dists_clipped, -p - 2)
        np.fill_diagonal(weights, 0.0)
        
        # grads_i = sum_j weights_ij * (x_i - x_j)
        # This is equivalent to:
        # grads_i = sum_j weights_ij * x_i - sum_j weights_ij * x_j
        # grads_i = x_i * sum_j weights_ij - (weights @ x)_i
        
        sum_weights = np.sum(weights, axis=1)
        weighted_sum_x = weights @ centers
        
        grads += centers * sum_weights[:, np.newaxis] - weighted_sum_x
        
        # Wall gradient
        # w_i = min(0.5 - |x_i|, 0.5 - |y_i|)
        # This is the distance to the closest wall.
        # If x_i > 0, closest wall in x is right (0.5). dist = 0.5 - x_i. grad = -1.
        # If x_i < 0, closest wall in x is left (-0.5). dist = 0.5 + x_i. grad = 1.
        # Similarly for y.
        # The active constraint is the one with the minimum distance.
        
        # Calculate distances to 4 walls for each point
        d_left = 0.5 + centers[:, 0]
        d_right = 0.5 - centers[:, 0]
        d_bottom = 0.5 + centers[:, 1]
        d_top = 0.5 - centers[:, 1]
        
        # We need the gradient of min(d_left, d_right, d_bottom, d_top)^-p
        # Let w = min(...). d(w^-p)/dx = -p w^(-p-1) * dw/dx
        # dw/dx is the gradient of the active wall distance.
        
        # For each point, find the active walls.
        # Since it's a min, only the closest wall(s) contribute.
        # If there's a tie, sum the gradients.
        
        for i in range(N):
            w_val = wall_dists_clipped[i]
            # Coefficient for gradient of energy wrt wall distance
            coeff = -p * np.power(w_val, -p - 1)
            
            # Determine active walls
            gx = 0.0
            gy = 0.0
            
            # Check if left is active
            if np.isclose(d_left[i], w_val, rtol=1e-9, atol=1e-12):
                gx += 1.0  # d(0.5+x)/dx = 1
            if np.isclose(d_right[i], w_val, rtol=1e-9, atol=1e-12):
                gx += -1.0 # d(0.5-x)/dx = -1
            if np.isclose(d_bottom[i], w_val, rtol=1e-9, atol=1e-12):
                gy += 1.0
            if np.isclose(d_top[i], w_val, rtol=1e-9, atol=1e-12):
                gy += -1.0
            
            grads[i, 0] += coeff * gx
            grads[i, 1] += coeff * gy

        return total_energy, grads.reshape(-1)

    def hex_init(N):
        # Approximate radius for 200 circles in square ~ 0.036
        # Hex packing density is pi/sqrt(12) ~ 0.9069
        # Area per circle ~ 1/200 = 0.005
        # r^2 * pi * 0.9069 ~ 0.005 => r^2 ~ 0.00175 => r ~ 0.042
        # But boundary effects reduce this. Let's use 0.035 for init spacing.
        r_target = 0.035
        dx = 2 * r_target
        dy = np.sqrt(3) * r_target
        
        points = []
        y = -0.5 + r_target
        row = 0
        while y <= 0.5 - r_target + 1e-6:
            x_offset = 0 if row % 2 == 0 else dx / 2
            x = -0.5 + r_target + x_offset
            while x <= 0.5 - r_target + 1e-6:
                points.append((x, y))
                x += dx
            y += dy
            row += 1
            
        points = np.array(points)
        
        if len(points) >= N:
            # Select N points closest to center to maximize symmetry/packing
            dists_to_center = np.linalg.norm(points, axis=1)
            sorted_indices = np.argsort(dists_to_center)
            return points[sorted_indices[:N]]
        else:
            # If not enough, add random points in the valid region
            remaining = N - len(points)
            # Valid region is slightly inside
            random_points = np.random.uniform(-0.45, 0.45, (remaining, 2))
            return np.vstack([points, random_points])

    def grid_init(N):
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]

    best_centers = None
    best_score = -np.inf
    
    # Try a few initializations
    inits = [
        hex_init(N),
        grid_init(N)
    ]
    
    for init in inits:
        if time.time() - start_time > deadline:
            break
            
        x0 = init.flatten()
        
        # Optimization
        try:
            # Use L-BFGS-B with the energy objective
            # Bounds: [-0.5, 0.5] for all coordinates
            bounds = [(-0.5, 0.5)] * (2 * N)
            
            result = minimize(
                energy_and_grad, 
                x0, 
                method='L-BFGS-B', 
                jac=True,
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            
            if time.time() - start_time < deadline:
                new_centers = result.x.reshape(N, 2)
                # Clip to ensure strict validity
                new_centers = np.clip(new_centers, -0.5, 0.5)
                new_score = compute_score(new_centers)
                
                if new_score > best_score:
                    best_score = new_score
                    best_centers = new_centers.copy()
        except Exception as e:
            pass

    # If no valid result, fallback to grid
    if best_centers is None:
        best_centers = grid_init(N)
        best_score = compute_score(best_centers)
        
    return best_centers
