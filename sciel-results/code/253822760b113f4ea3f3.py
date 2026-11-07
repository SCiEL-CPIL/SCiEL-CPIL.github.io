import numpy as np
import time
from scipy.special import erf

# Global timer for deadline guard
_start_time = None

def acquisition(mu, sigma, f_best):
    """
    Grid-based local search acquisition function.
    
    Maps the 2000 candidate indices to a 2D grid and applies multi-start
    local optimization to identify the point with the lowest posterior mean,
    bypassing the stochastic exploration-exploitation trade-off of scalar
    acquisition functions to exploit the dense sampling.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Posterior mean at candidate points
    sigma : array of shape (2000,)
        Posterior standard deviation at candidate points
    f_best : float
        Best (lowest) observed value so far
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores, higher is better
    """
    global _start_time
    
    # Initialize timer
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: 330s limit
    elapsed = time.time() - _start_time
    if elapsed > 320:
        # Return a simple exploitation score as fallback
        return -mu
    
    n = len(mu)
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # Map 2000 candidates to a 45x45 grid (2025 cells, close to 2000)
    grid_size = 45
    # Assign each candidate to a grid cell
    # We'll use a space-filling mapping: index -> (row, col)
    rows = np.arange(n) // grid_size
    cols = np.arange(n) % grid_size
    
    # Create a 2D array of mu values on the grid
    mu_grid = np.full((grid_size, grid_size), np.inf)
    sigma_grid = np.full((grid_size, grid_size), np.inf)
    
    for i in range(n):
        r, c = rows[i], cols[i]
        if r < grid_size and c < grid_size:
            mu_grid[r, c] = mu[i]
            sigma_grid[r, c] = sigma[i]
    
    # Multi-start local search: find local minima in mu_grid
    # Start from several points and do gradient descent on the grid
    scores = np.zeros(n)
    
    # Find candidate starting points: local minima in the grid
    # A cell is a local minimum if it's less than or equal to its 8 neighbors
    local_minima = []
    for r in range(grid_size):
        for c in range(grid_size):
            if np.isinf(mu_grid[r, c]):
                continue
            is_min = True
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < grid_size and 0 <= nc < grid_size:
                        if not np.isinf(mu_grid[nr, nc]) and mu_grid[nr, nc] < mu_grid[r, c]:
                            is_min = False
                            break
                if not is_min:
                    break
            if is_min:
                local_minima.append((r, c))
    
    # If no local minima found, use the global minimum
    if not local_minima:
        flat_idx = np.argmin(mu_grid)
        r, c = divmod(flat_idx, grid_size)
        local_minima = [(r, c)]
    
    # For each local minimum, perform local search and assign high scores
    # to nearby candidates
    for start_r, start_c in local_minima[:20]:  # Limit to top 20 starts
        # Local search: hill climbing on the grid
        curr_r, curr_c = start_r, start_c
        best_val = mu_grid[curr_r, curr_c]
        
        # Iterate: move to the best neighbor
        for _ in range(10):
            improved = False
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = curr_r + dr, curr_c + dc
                    if 0 <= nr < grid_size and 0 <= nc < grid_size:
                        if not np.isinf(mu_grid[nr, nc]) and mu_grid[nr, nc] < best_val:
                            best_val = mu_grid[nr, nc]
                            curr_r, curr_c = nr, nc
                            improved = True
            if not improved:
                break
        
        # Assign scores to candidates near this local minimum
        # Use a Gaussian kernel centered at the found minimum
        for i in range(n):
            r_i, c_i = rows[i], cols[i]
            if r_i >= grid_size or c_i >= grid_size:
                continue
            dist_sq = (r_i - curr_r)**2 + (c_i - curr_c)**2
            # Score based on proximity to the local minimum
            # Lower mu and closer distance = higher score
            base_score = -mu[i]
            proximity_bonus = np.exp(-dist_sq / 4.0) * 10.0
            scores[i] = max(scores[i], base_score + proximity_bonus)
    
    # Also add a small exploration component based on sigma
    # to avoid complete exploitation
    exploration_weight = 0.1
    scores += exploration_weight * sigma_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
