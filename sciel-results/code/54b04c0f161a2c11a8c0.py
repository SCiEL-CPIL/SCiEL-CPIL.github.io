import numpy as np

def heuristics(demands, capacity):
    n = len(demands)
    d_min = np.min(demands)
    
    # Vectorized computation
    # demands is 1D array
    # We need H[i, j]
    
    d_i = demands[:, None]
    d_j = demands[None, :]
    
    sums = d_i + d_j
    slack = capacity - sums
    
    # Initialize with small value (invalid)
    H = np.full((n, n), 1e-9, dtype=np.float64)
    
    # Valid mask: sums <= capacity
    valid = sums <= capacity
    
    # Calculate fit score for valid entries
    # Base fit: (d_i + d_j) / capacity
    base_fit = np.where(valid, sums / capacity, 0)
    
    # Slack penalty factor
    # sigma = 10
    sigma = 10.0
    slack_sq = np.where(valid, slack * slack, 0)
    fit_factor = np.where(valid, 1.0 / (1.0 + slack_sq / (sigma * sigma)), 0)
    
    # Dead slack penalty
    # If 0 < slack < d_min, penalty = 0.1
    # Else penalty = 1.0
    dead_slack = (slack > 0) & (slack < d_min)
    penalty = np.where(dead_slack, 0.1, 1.0)
    # Apply penalty only where valid
    penalty = np.where(valid, penalty, 1.0)
    
    H = np.where(valid, base_fit * fit_factor * penalty, 1e-9)
    
    # Ensure diagonal or self-interactions are handled if needed?
    # The task says "placing item j right after item i". 
    # Usually i != j in terms of *instances* of items, but here items are distinct indices.
    # If i == j, it means placing item i after item i? No, items are placed once.
    # The ACO loop picks "a fitting item". It maintains a list of unplaced items.
    # So i and j are distinct indices in the actual execution.
    # However, the matrix is N x N.
    # If i == j, it's not a valid move in the actual algorithm (can't place same item twice).
    # But the heuristic matrix is static. It doesn't know which items are placed.
    # The ACO algorithm will mask out already placed items.
    # So H[i, i] doesn't really matter, but let's keep it consistent.
    
    return H
