import numpy as np

def heuristics(demands, capacity):
    demands = np.asarray(demands, dtype=float)
    n = demands.shape[0]

    # Calculate sums of all pairs
    sums = demands[:, None] + demands[None, :]
    
    # Determine which pairs fit in the bin
    fits = sums <= capacity
    
    # Calculate the gap (waste) for each pair
    # For fitting pairs, gap >= 0
    gap = capacity - sums
    
    # Heuristic: Inverse of the gap (plus a small epsilon to avoid division by zero)
    # This strongly favors pairs that fill the bin completely (gap close to 0).
    # Pairs with larger gaps get much lower attractiveness.
    # We use a small epsilon to handle the case where gap is 0.
    epsilon = 1e-5
    with np.errstate(divide='ignore', invalid='ignore'):
        h_val = 1.0 / (gap + epsilon)
        
    # For non-fitting pairs, gap < 0, so h_val would be negative or large positive if we weren't careful.
    # But we mask them out.
    # For fitting pairs, gap >= 0, so h_val is positive.
    
    # Initialize attractiveness with a very small value for invalid pairs
    attractiveness = np.full((n, n), 1e-9)
    
    # Assign heuristic values only for fitting pairs
    attractiveness = np.where(fits, h_val, 1e-9)
    
    # Ensure all values are positive and finite
    # If any value is not finite or <= 0, replace with 1e-9
    attractiveness = np.where(np.isfinite(attractiveness) & (attractiveness > 0), attractiveness, 1e-9)
    
    return attractiveness
