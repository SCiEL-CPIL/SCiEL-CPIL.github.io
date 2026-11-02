import numpy as np
import time
from scipy.optimize import minimize

N = 50

def construct():
    start_time = time.time()
    time_limit = 115.0

    def get_initial_guess():
        # Hexagonal grid
        # We want to pack N points.
        # Approximate number of rows: sqrt(N * 2 / sqrt(3))
        rows = int(np.ceil(np.sqrt(N * 2.0 / np.sqrt(3.0))))
        cols = int(np.ceil(N / rows))
        
        points = []
        for r in range(rows):
            for c in range(cols):
                # Hex spacing
                x = c * 1.0
                y = r * np.sqrt(3.0) / 2.0
                if r % 2 == 1:
                    x += 0.5
                points.append([x, y])
        
        # Take first N points
        points = points[:N]
        
        if len(points) < N:
            # Fallback: square grid
            side = int(np.ceil(np.sqrt(N)))
            for r in range(side):
                for c in range(side):
                    points.append([c, r])
                    if len(points) == N:
                        break
                if len(points) == N:
                    break
        
        points = np.array(points)
        
        # Normalize to fit in [-0.49, 0.49]
        min_x, max_x = points[:, 0].min(), points[:, 0].max()
        min_y, max_y = points[:, 1].min(), points[:, 1].max()
        
        scale_x = 0.98 / (max_x - min_x + 1e-6)
        scale_y = 0.98 / (max_y - min_y + 1e-6)
        scale = min(scale_x, scale_y)
        
        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0
        
        points[:, 0] = (points[:, 0] - center_x) * scale
        points[:, 1] = (points[:, 1] - center_y) * scale
        
        # Add small noise to break symmetry
        np.random.seed(42)
        points += np.random.uniform(-0.01, 0.01, size=points.shape)
        
        # Clip to bounds
        points[:, 0] = np.clip(points[:, 0], -0.499, 0.499)
        points[:, 1] = np.clip(points[:, 1], -0.499, 0.499)
        
        return points.flatten()

    def objective(x):
        centers = x.reshape(N, 2)
        
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=2)
        
        # Avoid division by zero
        # We want to minimize sum(1/d^4) = sum(1/dists_sq^2)
        # Clip dists_sq to avoid 0
        dists_sq = np.clip(dists_sq, 1e-12, None)
        
        # Exclude diagonal
        np.fill_diagonal(dists_sq, np.inf)
        
        # Sum of repulsions
        # 1/d^4 is (1/d^2)^2
        repulsion = np.sum(1.0 / (dists_sq * dists_sq))
        
        return repulsion

    x0 = get_initial_guess()
    
    # Bounds
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # Optimize
    # Use L-BFGS-B
    # To improve convergence, we might need to adjust options
    res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
    
    x_opt = res.x
    
    # Check time
    if (time.time() - start_time) < time_limit:
         # Try a second pass with a different initial guess or perturbation?
         # Or just return. The first pass is usually decent.
         pass
         
    return x_opt.reshape(N, 2)
