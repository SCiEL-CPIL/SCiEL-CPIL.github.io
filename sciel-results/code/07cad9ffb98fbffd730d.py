import numpy as np

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
    n = len(demands)
    # Vectorized calculation of pair sums
    pair_sums = demands[:, None] + demands[None, :]
    
    # Initialize heuristic matrix with a very small positive value for non-fitting pairs
    # The problem states non-positive become 1e-9, but we should return positive values.
    # 1e-9 is a safe small positive number.
    H = np.full((n, n), 1e-9)
    
    # Mask for fitting pairs
    fit_mask = pair_sums <= capacity
    
    # Calculate waste for fitting pairs
    # waste = capacity - pair_sums
    # Heuristic = 1 / (1 + waste)
    # This gives high values for small waste (tight fits) and low values for large waste.
    
    # We only calculate for fitting pairs to avoid division issues or negative values
    # Note: pair_sums can be equal to capacity, waste=0, H=1.
    # If pair_sums < capacity, waste > 0, H < 1.
    
    waste = capacity - pair_sums
    # Ensure waste is non-negative for the calculation, though fit_mask guarantees it
    # We use np.where to apply the formula only where fit_mask is True
    
    # To avoid division by zero or large numbers if waste is huge (though max waste is C-2*20=110),
    # 1/(1+110) is small.
    
    H[fit_mask] = 1.0 / (1.0 + waste[fit_mask])
    
    return H
