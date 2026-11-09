import numpy as np

def heuristics(demands, capacity):
    n = len(demands)
    # Sort demands for potential optimization, but we need original indices for matrix
    # Actually, we just need the list of values
    d = np.array(demands)
    
    # Calculate all pairwise sums
    pair_sums = d[:, None] + d[None, :]
    residuals = capacity - pair_sums
    
    # Initialize heuristic matrix
    H = np.full((n, n), 1e-9)
    
    # Valid pairs are those with residual >= 0
    valid_mask = residuals >= 0
    
    # For valid pairs, calculate N(R) = count of items <= R
    # We can do this by iterating over unique residuals or just computing for each valid pair
    # Given N=120, N^2 is small.
    
    # To speed up N(R), we can sort d and use searchsorted
    d_sorted = np.sort(d)
    
    # Get indices of valid pairs
    valid_indices = np.argwhere(valid_mask)
    
    for i, j in valid_indices:
        r = residuals[i, j]
        # Count how many items in d are <= r
        # searchsorted returns index where r would be inserted to keep sorted order
        # so it's the count of elements <= r if we use 'right'
        count = np.searchsorted(d_sorted, r, side='right')
        
        # Heuristic: Count + 1/(R+1)
        # If count is 0, it relies on 1/(R+1)
        # If count > 0, it relies on count
        H[i, j] = count + 1.0 / (r + 1.0)
        
    return H
