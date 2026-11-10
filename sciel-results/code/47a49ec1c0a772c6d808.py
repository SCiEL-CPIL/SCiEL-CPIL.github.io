import numpy as np

def heuristics(demands, capacity):
    n = len(demands)
    d = demands.astype(float)
    
    # Pair sums
    pair = d[:, None] + d[None, :]
    fits = pair <= capacity
    
    # Remaining space after placing i
    rem = capacity - d[:, None]
    
    # Waste if we place j after i
    # waste = capacity - d_i - d_j
    # We want to minimize waste relative to rem
    # ratio_waste = (capacity - d_i - d_j) / rem
    #             = (rem - d_j) / rem
    #             = 1 - d_j/rem
    
    # Only valid where fits
    # If fits, d_j <= rem, so ratio_waste >= 0
    
    # Calculate ratio_waste
    # Handle division by zero (though rem > 0 always)
    ratio_waste = np.where(fits, 1.0 - d[None, :] / rem, 0.0)
    
    # Heuristic: exp(-alpha * ratio_waste)
    alpha = 2.0
    eta = np.where(fits, np.exp(-alpha * ratio_waste), 0.0)
    
    # Ensure positive
    eta = np.maximum(eta, 1e-9)
    
    return eta
