"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    """
    Pack N equal circles inside the unit circle to maximize the common radius.
    
    Uses a multi-start local optimization approach:
    1. Initialize points using a hexagonal lattice pattern
    2. Optimize the radius and positions using scipy.optimize.minimize
    3. Try multiple initial configurations and keep the best result
    
    The objective is to maximize the minimum of:
    - Half the minimum distance between any two circle centers
    - The minimum distance from any center to the container boundary
    """
    
    def compute_radius(centers):
        """Compute the feasible radius for given centers."""
        # Minimum distance to boundary
        dist_to_boundary = 1.0 - np.linalg.norm(centers, axis=1)
        min_boundary = np.min(dist_to_boundary)
        
        # Minimum pairwise distance / 2
        if len(centers) < 2:
            min_pair = np.inf
        else:
            # Compute all pairwise distances
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            # Zero out diagonal
            np.fill_diagonal(dists, np.inf)
            min_pair = np.min(dists) / 2.0
        
        return min(min_boundary, min_pair)
    
    def objective(params, n):
        """Objective function to minimize (negative of the radius)."""
        r = params[0]
        centers = params[1:].reshape(n, 2)
        
        # Penalty for centers outside the container
        dists_to_origin = np.linalg.norm(centers, axis=1)
        boundary_violation = np.sum(np.maximum(0, dists_to_origin - (1.0 - r)) ** 2)
        
        # Penalty for overlapping circles
        if n >= 2:
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dists, np.inf)
            overlap_violation = np.sum(np.maximum(0, 2 * r - dists) ** 2)
        else:
            overlap_violation = 0
        
        # We want to maximize r, so minimize -r with penalties
        return -r + 100.0 * (boundary_violation + overlap_violation)
    
    def make_initial_hexagonal(n, scale=0.8):
        """Create initial configuration using hexagonal lattice."""
        points = []
        # Generate hexagonal lattice points
        for j in range(-10, 11):
            for i in range(-10, 11):
                x = i * scale + 0.5 * scale * (j % 2)
                y = j * scale * np.sqrt(3) / 2
                if np.sqrt(x * x + y * y) < 1.0:
                    points.append([x, y])
        
        points = np.array(points)
        
        # If we have more than n points, select n closest to center
        if len(points) > n:
            dists = np.linalg.norm(points, axis=1)
            indices = np.argsort(dists)[:n]
            points = points[indices]
        elif len(points) < n:
            # Add random points if not enough
            while len(points) < n:
                angle = np.random.uniform(0, 2 * np.pi)
                radius = np.random.uniform(0, 0.9)
                points = np.vstack([points, [radius * np.cos(angle), radius * np.sin(angle)]])
        
        return points
    
    def make_initial_ring(n, rings=5):
        """Create initial configuration using concentric rings."""
        points = [[0.0, 0.0]]
        remaining = n - 1
        for ring in range(1, rings + 1):
            if remaining <= 0:
                break
            # Number of points on this ring
            n_on_ring = min(remaining, 6 * ring)
            radius = ring * 0.15
            for i in range(n_on_ring):
                angle = 2 * np.pi * i / n_on_ring
                points.append([radius * np.cos(angle), radius * np.sin(angle)])
            remaining -= n_on_ring
        
        # If still not enough, add more
        while len(points) < n:
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(0, 0.9)
            points.append([radius * np.cos(angle), radius * np.sin(angle)])
        
        return np.array(points[:n])
    
    def make_initial_random(n, seed=42):
        """Create random initial configuration."""
        np.random.seed(seed)
        points = []
        for _ in range(n):
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(0, 0.95)
            points.append([radius * np.cos(angle), radius * np.sin(angle)])
        return np.array(points)
    
    def optimize_from_initial(centers, n):
        """Optimize starting from given initial centers."""
        # Estimate initial radius
        r_init = compute_radius(centers)
        r_init = max(r_init, 0.01)
        
        # Parameters: [r, x1, y1, x2, y2, ...]
        params = np.concatenate([[r_init], centers.flatten()])
        
        # Bounds: r in [0.01, 0.15], coordinates in [-1, 1]
        bounds = [(0.01, 0.15)] + [(-1.0, 1.0)] * (2 * n)
        
        # Try multiple optimization methods
        best_result = None
        best_r = -np.inf
        
        for method in ['L-BFGS-B', 'Nelder-Mead']:
            try:
                result = minimize(objective, params, args=(n,), method=method,
                                  bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12})
                
                if result.success or result.fun < best_r:
                    # Extract the radius
                    r_opt = result.x[0]
                    centers_opt = result.x[1:].reshape(n, 2)
                    
                    # Verify the solution is valid
                    actual_r = compute_radius(centers_opt)
                    if actual_r > best_r:
                        best_r = actual_r
                        best_result = centers_opt
            except Exception:
                pass
        
        return best_result, best_r
    
    # Try multiple initial configurations
    best_centers = None
    best_r = -np.inf
    
    initial_configs = []
    
    # Hexagonal with different scales
    for scale in [0.1, 0.12, 0.15, 0.2, 0.25]:
        initial_configs.append(make_initial_hexagonal(N, scale=scale))
    
    # Ring configurations
    initial_configs.append(make_initial_ring(N, rings=4))
    initial_configs.append(make_initial_ring(N, rings=6))
    
    # Random configurations with different seeds
    for seed in [42, 123, 456, 789, 1000]:
        initial_configs.append(make_initial_random(N, seed=seed))
    
    # Optimize from each initial configuration
    for i, centers in enumerate(initial_configs):
        centers_opt, r_opt = optimize_from_initial(centers, N)
        if centers_opt is not None and r_opt > best_r:
            best_r = r_opt
            best_centers = centers_opt
    
    # If optimization failed, fall back to hexagonal lattice
    if best_centers is None:
        best_centers = make_initial_hexagonal(N, scale=0.15)
        best_r = compute_radius(best_centers)
    
    return best_centers
# EVOLVE-BLOCK-END
