import numpy as np

def heuristics(demands, capacity):
    demands = np.asarray(demands, dtype=float)
    n = len(demands)
    
    # Calculate pair sums
    sums = demands[:, None] + demands[None, :]
    
    # Identify valid pairs (sum <= capacity)
    valid = sums <= capacity
    
    # Calculate gap for valid pairs
    gaps = capacity - sums
    
    # Initialize heuristic with a very small value
    eta = np.full((n, n), 1e-9, dtype=float)
    
    # Extract indices for valid pairs
    i_idx, j_idx = np.where(valid)
    if len(i_idx) == 0:
        return eta
        
    d_j = demands[j_idx]
    waste = gaps[valid]
    
    # Determine a scaling factor for waste.
    # Using the minimum demand as a natural scale for "useful" vs "dead" space.
    d_min = np.min(demands)
    
    # Scaled waste
    # If waste is 0, scaled is 0.
    # If waste is d_min, scaled is 1.
    # This flattens the penalty curve compared to raw waste^2.
    # Raw: w=10, w^2=100.
    # Scaled: w=10, (10/20)^2 = 0.25.
    # This allows the ant to consider pairs with small non-zero waste more seriously.
    
    scaled_waste = waste / d_min
    
    # Heuristic formula
    # eta = d_j / (1 + scaled_waste^2)
    scores = d_j / (1.0 + scaled_waste * scaled_waste)
    
    # Assign to matrix
    eta[valid] = scores
    
    # Diagonal: set to 1e-9 (cannot place item after itself in this context usually, or just invalid)
    np.fill_diagonal(eta, 1e-9)
    
    # Ensure no non-finite values
    eta[~np.isfinite(eta)] = 1e-9
    
    return eta
