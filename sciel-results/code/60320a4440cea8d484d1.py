import numpy as np

def heuristics(demands, capacity):
    n = len(demands)
    d = np.array(demands, dtype=float)
    
    # Compute sum of all pairs
    pair_sum = d[:, None] + d[None, :]
    
    # Calculate waste (remaining space in bin if only i and j are in it)
    waste = capacity - pair_sum
    
    # Determine valid pairs (those that fit)
    # Use a small tolerance for floating point errors
    valid = pair_sum <= capacity + 1e-9
    
    # Initialize heuristic matrix
    h = np.zeros((n, n), dtype=float)
    
    # For valid pairs, compute attractiveness
    # Using inverse squared waste: 1 / (waste^2 + 1)
    # This creates a steeper potential field, strongly repelling ants
    # from leaving large empty spaces and rewarding tight fits.
    waste_clipped = np.maximum(waste, 0.0)
    h[valid] = 1.0 / (waste_clipped[valid] ** 2 + 1.0)
    
    # The prompt states: "Heuristic entries that are not positive become 1e-9"
    # My h[~valid] is 0.0, which is not positive, so it will become 1e-9.
    # My h[valid] is > 0, so it stays.
    
    return h
