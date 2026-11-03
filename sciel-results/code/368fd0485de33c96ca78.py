import numpy as np
from scipy.optimize import minimize

N = 26

def construct():
    def get_radii(centers):
        # centers: (N, 2)
        n = centers.shape[0]
        radii = np.zeros(n)
        for i in range(n):
            # Distance to boundaries
            xb = centers[i, 0]
            yb = centers[i, 1]
            dist_b = min(xb, 1.0 - xb, yb, 1.0 - yb)
            
            # Distance to other centers
            # Vectorized distance calculation
            dists = np.linalg.norm(centers - centers[i], axis=1)
            dists[i] = np.inf
            min_dist_c = np.min(dists)
            
            radii[i] = 0.5 * min(dist_b, min_dist_c)
        return radii

    def objective(x):
        centers = x.reshape(-1, 2)
        radii = get_radii(centers)
        return -np.sum(radii)

    bounds = [(0, 1)] * (2 * N)
    
    best_val = np.inf
    best_x = None
    
    # Random restarts
    np.random.seed(42)
    
    num_restarts = 150
    
    for k in range(num_restarts):
        if k < 20:
            # Grid-like initialization with jitter
            side = 5
            # Create 5x5 grid points
            x_coords = np.linspace(0.1, 0.9, 5)
            y_coords = np.linspace(0.1, 0.9, 5)
            xx, yy = np.meshgrid(x_coords, y_coords)
            grid_points = np.column_stack((xx.ravel(), yy.ravel()))
            
            # Add one extra point in the center or random
            if k % 2 == 0:
                extra = np.array([0.5, 0.5])
            else:
                extra = np.random.rand(2)
            
            x0_grid = np.vstack([grid_points, extra])
            # Add small jitter to break symmetry
            x0_grid += np.random.normal(0, 0.01, x0_grid.shape)
            # Clip to bounds
            x0_grid = np.clip(x0_grid, 0.01, 0.99)
            
            # Shuffle
            np.random.shuffle(x0_grid)
            x0 = x0_grid.ravel()
        else:
            x0 = np.random.rand(2 * N)
            
        # Run L-BFGS-B
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-9})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            continue

    if best_x is None:
        # Fallback to seed if something went wrong
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    centers = best_x.reshape(-1, 2)
    radii = get_radii(centers)
    
    # Construct output array
    result = np.zeros((N, 3))
    result[:, 0] = centers[:, 0]
    result[:, 1] = centers[:, 1]
    result[:, 2] = radii
    
    return result
