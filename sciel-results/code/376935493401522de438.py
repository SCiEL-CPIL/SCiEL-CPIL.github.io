"""Packing 200 equal circles in a unit square via gradient-based optimization."""
import numpy as np
from scipy.optimize import minimize
import time

N = 200

# EVOLVE-BLOCK-START

def _compute_radius(centres):
    """Compute the maximum common radius for the given centres."""
    # Distance between all pairs
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    
    # Distance to walls
    wall_dists = np.minimum(np.minimum(0.5 - centres[:, 0], 0.5 + centres[:, 0]),
                            np.minimum(0.5 - centres[:, 1], 0.5 + centres[:, 1]))
    min_wall_dist = np.min(wall_dists)
    
    return min(min_pair_dist / 2.0, min_wall_dist)

def _initial_hexagonal_layout(n):
    """Create an initial hexagonal-like layout for n circles."""
    # Approximate hexagonal packing
    # For a hexagonal arrangement, we can estimate rows
    # In hex packing, each row has either floor or ceil(n / rows) circles
    # Let's try different numbers of rows
    
    best_centres = None
    best_radius = 0
    
    # Try various row counts
    for rows in range(int(np.sqrt(n)) - 2, int(np.sqrt(n)) + 5):
        if rows <= 0:
            continue
        cols_per_row = np.ceil(n / rows).astype(int)
        
        # Calculate spacing
        # In hexagonal packing, horizontal spacing = 2r, vertical spacing = 2r * sqrt(3)/2
        # For a unit square, we need to fit rows * 2r * sqrt(3)/2 <= 1 and cols * 2r <= 1
        # So r <= 1/(2*cols) and r <= 1/(rows * sqrt(3))
        
        # Let's place them in a grid first, then perturb
        centres = []
        for i in range(rows):
            for j in range(cols_per_row):
                if len(centres) >= n:
                    break
                x = (j + 0.5) / cols_per_row - 0.5
                y = (i + 0.5) / rows - 0.5
                # Add hexagonal offset
                if i % 2 == 1:
                    x += 0.5 / cols_per_row
                centres.append([x, y])
        
        if len(centres) == n:
            centres = np.array(centres)
            r = _compute_radius(centres)
            if r > best_radius:
                best_radius = r
                best_centres = centres
    
    if best_centres is None:
        # Fallback to square grid
        m = int(np.ceil(np.sqrt(n)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centres = grid[:n]
    
    return best_centres

def _refine_with_scipy(centres, max_time=100):
    """Refine the layout using scipy optimization."""
    start_time = time.time()
    deadline = start_time + max_time
    
    def objective(x):
        """Negative radius to maximize."""
        centres = x.reshape(-1, 2)
        # Check if valid
        if np.any(np.isnan(centres)) or np.any(np.isinf(centres)):
            return 1e10
        r = _compute_radius(centres)
        return -r
    
    # Use L-BFGS-B with bounds
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    best_centres = centres.copy()
    best_radius = _compute_radius(centres)
    
    # Try multiple random perturbations and optimizations
    np.random.seed(42)
    
    for attempt in range(10):
        if time.time() > deadline - 5:
            break
        
        # Perturb the initial layout
        perturbation = np.random.randn(2 * N) * 0.01
        x0 = centres.flatten() + perturbation
        
        # Clip to valid range
        x0 = np.clip(x0, -0.5, 0.5)
        
        # Optimize
        try:
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if result.success or result.fun < -best_radius:
                new_centres = result.x.reshape(-1, 2)
                new_radius = -result.fun
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_centres = new_centres.copy()
        except Exception:
            continue
        
        if time.time() > deadline - 10:
            break
    
    return best_centres, best_radius

def construct():
    """Construct the optimal packing of 200 circles in a unit square."""
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # Generate initial layout
    centres = _initial_hexagonal_layout(N)
    
    # Refine with scipy
    centres, radius = _refine_with_scipy(centres, max_time=120)
    
    # Ensure centres are within bounds
    centres = np.clip(centres, -0.5, 0.5)
    
    return centres

# EVOLVE-BLOCK-END
