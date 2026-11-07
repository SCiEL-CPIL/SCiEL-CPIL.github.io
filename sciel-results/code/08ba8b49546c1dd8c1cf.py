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

    # New Heuristic: Item-Size Weighted Tightness
    # Favor placing larger items (d_j) in tighter fits (smaller gap)
    d_j = demands[None, :]
    
    # Calculate raw values for valid entries
    # H_ij = d_j / (1 + gap_ij)
    raw_values = d_j / (1.0 + gap)
    
    # Extract valid raw values
    valid_raw = raw_values[valid]
    
    # Normalize to [0, 1] to keep scale consistent with typical ACO heuristics
    # This ensures the heuristic doesn't overpower pheromone excessively
    max_val = np.max(valid_raw)
    if max_val > 0:
        normalized_valid = valid_raw / max_val
    else:
        normalized_valid = valid_raw
        
    H[valid] = normalized_valid
    
    return H
