"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import sys

def _score(positions):
    """Compute the minimum radius given positions in [-0.5, 0.5]^2."""
    n = len(positions)
    if n == 0:
        return 0.0
    
    # Ensure positions are within bounds for valid score calculation
    # If any position is out of bounds, the radius is effectively 0 or invalid
    # The problem states centers must be inside.
    
    # Distance to boundary
    # min_i min_k (1/2 - |c_ik|)
    # This is equivalent to min( min(0.5 - |x|), min(0.5 - |y|) )
    min_boundary = np.min(np.minimum(
        0.5 - np.abs(positions[:, 0]),
        0.5 - np.abs(positions[:, 1])
    ))
    
    if min_boundary <= 0:
        return 0.0

    # Pairwise distances
    # min_{i<j} |c_i - c_j| / 2
    # We can compute this efficiently using broadcasting for N=200
    # diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    # dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    # np.fill_diagonal(dist, np.inf)
    # min_pair = np.min(dist)
    
    # For N=200, explicit loop or vectorized is fine. Vectorized is faster in Python/NumPy.
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    
    return min(min_boundary, min_pair / 2.0)

def _objective(pos_flat):
    """Negative score for minimization."""
    pos = pos_flat.reshape(-1, 2)
    # Add small penalty if out of bounds to help optimizer stay in range
    # L-BFGS-B handles bounds, but this adds a gradient signal if it drifts
    out_of_bounds = np.any(pos > 0.5) or np.any(pos < -0.5)
    if out_of_bounds:
        return 1e6
    return -_score(pos)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin from 150s limit

    best_score = 0.0
    best_positions = None

    # Initial grid
    # Square grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    initial_pos = grid[:N].copy()
    
    # Evaluate initial
    curr_score = _score(initial_pos)
    if curr_score > best_score:
        best_score = curr_score
        best_positions = initial_pos.copy()

    # Define bounds: each coordinate in [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # Optimization settings
    # L-BFGS-B is good for box-constrained problems
    # We try multiple restarts if time permits, but first try the grid
    
    try:
        # First optimization pass from grid
        res = minimize(
            _objective, 
            initial_pos.ravel(), 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 1000, 'maxfun': 10000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        new_pos = res.x.reshape(-1, 2)
        new_score = _score(new_pos)
        if new_score > best_score:
            best_score = new_score
            best_positions = new_pos.copy()
            
    except Exception:
        pass

    # Check time
    if time.time() > deadline:
        return best_positions

    # Try a second pass from the best found so far, or a different initialization
    # A hexagonal grid might be a better starting point for the optimizer
    # Let's generate a hexagonal grid
    
    # Hexagonal grid generation
    # Rows of circles, alternating offsets
    # This is more complex to fit exactly N, so we stick to the result of the first optimization
    # and try to refine it further if time allows.
    
    if best_positions is not None:
        try:
            # Re-optimize from the best found with potentially different settings
            # or just run again to ensure convergence
            res2 = minimize(
                _objective, 
                best_positions.ravel(), 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 1000, 'maxfun': 10000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            new_pos2 = res2.x.reshape(-1, 2)
            new_score2 = _score(new_pos2)
            if new_score2 > best_score:
                best_score = new_score2
                best_positions = new_pos2.copy()
        except Exception:
            pass

    # If time remains, try a random restart
    if time.time() < deadline - 20:
        np.random.seed(42)
        random_pos = np.random.uniform(-0.4, 0.4, (N, 2))
        curr_score_rand = _score(random_pos)
        if curr_score_rand > best_score:
            best_score = curr_score_rand
            best_positions = random_pos.copy()
        
        try:
            res3 = minimize(
                _objective, 
                random_pos.ravel(), 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 500, 'maxfun': 5000}
            )
            
            new_pos3 = res3.x.reshape(-1, 2)
            new_score3 = _score(new_pos3)
            if new_score3 > best_score:
                best_score = new_score3
                best_positions = new_pos3.copy()
        except Exception:
            pass

    return best_positions
# EVOLVE-BLOCK-END
