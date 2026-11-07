import numpy as np
import time

def heuristics(demands, capacity):
    """
    Computes the heuristic matrix for the ACO bin packing problem.
    
    Args:
        demands: array of item sizes (floats).
        capacity: bin capacity (float).
        
    Returns:
        matrix: n x n matrix where entry (i, j) is the attractiveness of 
                placing item j after item i.
    """
    start_time = time.time()
    deadline = start_time + 325.0  # 5 second safety margin before 330s limit
    
    n = len(demands)
    demands_arr = np.array(demands, dtype=np.float64)
    
    # Initialize with a very small value for infeasible transitions
    H = np.full((n, n), 1e-9, dtype=np.float64)
    
    # Compute pairwise sums
    pair_sums = demands_arr[:, None] + demands_arr[None, :]
    
    # Feasibility mask: items i and j must fit together
    fit_mask = pair_sums <= capacity
    
    # Get indices of feasible pairs
    rows, cols = np.where(fit_mask)
    
    if len(rows) > 0:
        # Calculate waste for feasible pairs
        waste = capacity - pair_sums[rows, cols]
        
        # Heuristic strategy:
        # Prioritize placing larger items (d_j) to avoid stranding them,
        # while rewarding tight fits (low waste).
        # H[i, j] = d_j / (1 + waste)
        
        values = demands_arr[cols] / (1.0 + waste)
        
        # Ensure values are positive and finite
        values[~np.isfinite(values)] = 1e-9
        values[values <= 0] = 1e-9
        
        H[rows, cols] = values
    
    # Final safety check
    H[~np.isfinite(H)] = 1e-9
    H[H <= 0] = 1e-9
    
    # Check deadline (though computation is O(N^2) and N=120, it's very fast)
    if time.time() > deadline:
         # Return best result found so far (which is H)
         pass
         
    return H
