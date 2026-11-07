import numpy as np
import time

def heuristics(demands, capacity):
    start_time = time.time()
    deadline = start_time + 300.0  # safety margin before 330s limit
    
    demands = np.asarray(demands, dtype=float)
    n = len(demands)
    
    # Calculate pair sums
    sums = demands[:, None] + demands[None, :]
    
    # Identify valid pairs (sum <= capacity)
    valid = sums <= capacity
    
    # Calculate residual (gap) for valid pairs
    residuals = capacity - sums
    
    # Initialize heuristic with a very small value
    eta = np.full((n, n), 1e-9, dtype=float)
    
    # For each valid pair, compute how many other items fit in the residual
    # This rewards pairs that leave a "productive" residual
    # We'll compute this efficiently using sorting
    
    sorted_demands = np.sort(demands)
    
    # For each pair (i,j), count items k != i,j with d_k <= residual[i,j]
    # Using searchsorted on sorted_demands
    # But we need to exclude i and j themselves
    
    # Efficient approach: for each unique residual value, use searchsorted
    # Then adjust for the two excluded items
    
    # Compute base count: number of items with d <= residual
    # residuals[valid] has the residual values for valid pairs
    res_valid = residuals[valid]
    
    # Number of items with d <= r for each r
    # searchsorted gives insertion point, so count = searchsorted(sorted_demands, r, side='right')
    base_counts = np.searchsorted(sorted_demands, res_valid, side='right')
    
    # Now subtract 1 for each of i and j if their demand <= residual
    # We need the indices of valid pairs
    valid_indices = np.argwhere(valid)
    i_idx = valid_indices[:, 0]
    j_idx = valid_indices[:, 1]
    
    # For item i: if demands[i] <= residual, subtract 1
    # For item j: if demands[j] <= residual, subtract 1
    count_i = (demands[i_idx] <= res_valid).astype(int)
    count_j = (demands[j_idx] <= res_valid).astype(int)
    
    fitting_count = base_counts - count_i - count_j
    fitting_count = np.maximum(fitting_count, 0)
    
    # Normalize fitting count
    fit_norm = fitting_count / max(n - 2, 1)
    
    # Heuristic: combine tight-fill reward and productive-residual reward
    # Tight fill: exp(-alpha * residual) - rewards small residuals
    # Productive: fit_norm - rewards residuals that fit many items
    
    # Use a sharper exponential to strongly prefer tight packs
    alpha = 0.15
    tight_reward = np.exp(-alpha * res_valid)
    
    # Combine: primary weight on tight fill, secondary on productivity
    # When residual is very small, tight_reward is high, fit_norm is low (few items fit)
    # When residual is medium, tight_reward is moderate, fit_norm is high
    # We want to reward both extremes of "good" packing
    
    # Use a weighted combination with emphasis on tight fill
    weights_tight = 0.7
    weights_prod = 0.3
    scores = weights_tight * tight_reward + weights_prod * fit_norm
    
    # Ensure scores are positive
    scores = np.maximum(scores, 1e-9)
    
    # Assign to valid positions
    eta[valid] = scores
    
    # Diagonal: set to 1e-9 (item can't be placed after itself)
    np.fill_diagonal(eta, 1e-9)
    
    # Check deadline
    if time.time() > deadline:
        # Return what we have - should be fine since computation is fast
        pass
    
    return eta
