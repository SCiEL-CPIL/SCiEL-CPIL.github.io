import numpy as np
import time
import random
from scipy.optimize import minimize

N = 50
CUBE_HALF = 0.5
EPS = 1e-8

def objective(x):
    # x is a flattened array of shape (N, 3)
    c = x.reshape(N, 3)
    
    # Calculate pair distances
    # c_i - c_j
    # To avoid O(N^2) memory explosion for large N, we do it in a loop or vectorized block.
    # For N=50, 50x50 matrix is fine.
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Mask diagonal (self distance)
    np.fill_diagonal(dist_sq, np.inf)
    
    # Pairwise repulsion: sum of 1/dist^4
    # Use dist_sq to avoid sqrt
    # (1/sqrt(d^2))^4 = 1/d^4 = 1/(d^2)^2
    # Add EPS to avoid division by zero
    pair_energy = np.sum(1.0 / (dist_sq + EPS)**2)
    
    # Wall repulsion: distance to nearest wall
    # d_wall = 0.5 - |x|
    # We want to maximize min d_wall, so minimize sum of 1/d_wall^4
    # d_wall_k for each coordinate k
    abs_c = np.abs(c)
    d_wall = CUBE_HALF - abs_c
    # d_wall can be negative if outside, but bounds prevent this.
    # If d_wall is small, energy is high.
    wall_energy = np.sum(1.0 / (d_wall + EPS)**4)
    
    return pair_energy + wall_energy

def gradient(x):
    # Numerical gradient is slow. Analytical is better.
    # But for robustness and simplicity in a short script, let's rely on L-BFGS-B's numerical gradient
    # or provide an analytical one if easy.
    # Given the complexity of the wall term (abs), numerical is safer.
    # However, L-BFGS-B default is numerical. It's okay for N=50.
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_score = 0.0
    best_x = None
    
    # Bounds for L-BFGS-B
    bounds = [(-CUBE_HALF, CUBE_HALF)] * (N * 3)
    
    # Strategies for initialization
    # 1. Random
    # 2. Jittered Grid
    # 3. Lattice (FCC/BCC) subset
    
    def evaluate_config(x_init):
        nonlocal best_score, best_x
        if time.time() > deadline:
            return
        
        try:
            res = minimize(objective, x_init, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.success:
                c = res.x.reshape(N, 3)
                # Calculate actual radius
                # Min pairwise dist / 2
                diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
                dist_sq = np.sum(diff * diff, axis=2)
                np.fill_diagonal(dist_sq, np.inf)
                min_pair_dist = np.sqrt(np.min(dist_sq))
                
                # Min wall dist
                abs_c = np.abs(c)
                min_wall_dist = np.min(CUBE_HALF - np.max(abs_c, axis=1))
                
                r = min(min_pair_dist / 2.0, min_wall_dist)
                
                if r > best_score:
                    best_score = r
                    best_x = res.x.copy()
        except Exception:
            pass

    # 1. Jittered Grid Start
    # 50 spheres. 4x4x4 = 64. We take 50.
    # Or 3x3x3 = 27, too small.
    # Let's use a 4x4x4 grid and take the first 50, or random 50 from a finer grid.
    # Better: Random points in the cube.
    
    random.seed(42)
    np.random.seed(42)
    
    # Try several random starts
    num_starts = 20
    for i in range(num_starts):
        if time.time() > deadline:
            break
        
        # Random uniform
        x_init = np.random.uniform(-CUBE_HALF, CUBE_HALF, (N, 3)).flatten()
        evaluate_config(x_init)
        
        # Jittered grid
        if i % 2 == 0:
            # Create a 4x4x4 grid
            m = 4
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
            # Shuffle and take N
            indices = np.random.choice(len(grid), N, replace=False)
            x_init = grid[indices].flatten()
            # Add small jitter
            x_init += np.random.uniform(-0.01, 0.01, x_init.shape)
            evaluate_config(x_init)

    # If we have time, try a more targeted optimization on the best found so far
    if best_x is not None and time.time() < deadline - 10.0:
        # Refine the best solution with tighter tolerances
        try:
            res = minimize(objective, best_x, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.success:
                c = res.x.reshape(N, 3)
                diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
                dist_sq = np.sum(diff * diff, axis=2)
                np.fill_diagonal(dist_sq, np.inf)
                min_pair_dist = np.sqrt(np.min(dist_sq))
                abs_c = np.abs(c)
                min_wall_dist = np.min(CUBE_HALF - np.max(abs_c, axis=1))
                r = min(min_pair_dist / 2.0, min_wall_dist)
                if r > best_score:
                    best_score = r
                    best_x = res.x.copy()
        except Exception:
            pass

    # Final check and return
    if best_x is None:
        # Fallback to seed program behavior
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]
        
    return best_x.reshape(N, 3)

# EVOLVE-BLOCK-START
# The code above is the implementation.
# Ensure the function name is construct and it returns an array.
# EVOLVE-BLOCK-END
