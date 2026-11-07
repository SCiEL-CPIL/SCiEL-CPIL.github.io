import numpy as np

def heuristics(demands, capacity):
    n = len(demands)
    # Create pair sum matrix
    pair_sum = demands[:, None] + demands[None, :]
    
    # Calculate remaining space (gap)
    gap = capacity - pair_sum
    
    # Initialize heuristic matrix with a very small value
    H = np.full((n, n), 1e-9)
    
    # Valid pairs are those where gap >= 0
    valid = gap >= 0
    
    if not np.any(valid):
        return H

    # New Heuristic: Best Fit (Inverse Gap)
    # Favor items that result in the smallest remaining gap
    # This encourages tight packing and is a distinct approach from size-weighted heuristics
    
    # Calculate raw values for valid entries
    # H_ij = 1 / (1 + gap_ij)
    raw_values = 1.0 / (1.0 + gap)
    
    # Extract valid raw values
    valid_raw = raw_values[valid]
    
    # Normalize to [0, 1] to keep scale consistent with typical ACO heuristics
    max_val = np.max(valid_raw)
    if max_val > 0:
        normalized_valid = valid_raw / max_val
    else:
        normalized_valid = valid_raw
        
    H[valid] = normalized_valid
    
    return H
