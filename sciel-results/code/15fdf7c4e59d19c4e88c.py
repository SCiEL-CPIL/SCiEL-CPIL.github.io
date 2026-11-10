import numpy as np

def heuristics(demands, capacity):
    """
    Computes the heuristic matrix for the ACO bin packing task.
    
    Args:
        demands: list or array of item sizes (floats).
        capacity: bin capacity (float).
        
    Returns:
        numpy array of shape (n, n) with heuristic values.
    """
    n = len(demands)
    d = np.array(demands, dtype=np.float64)
    
    # Calculate pair sums
    pair_sums = d[:, None] + d[None, :]
    
    # Calculate waste
    waste = capacity - pair_sums
    
    # Initialize heuristic matrix with 1e-9 (will be used for non-fitting pairs)
    H = np.full((n, n), 1e-9, dtype=np.float64)
    
    # Mask for items that fit together
    fits = pair_sums <= capacity
    
    if np.any(fits):
        # Add a small epsilon to avoid division by zero for perfect fits
        # This ensures perfect fits get the highest score
        eps = 1e-6
        valid_waste = waste[fits] + eps
        
        # Heuristic: Reward tight fits AND large items
        # (d_i + d_j) / (waste)^2
        # Large items that fit tightly are highly attractive.
        # This aligns with the intuition that large items should be placed
        # early to leave space that small items can fill.
        
        # Extract the pair sums for the valid fits
        valid_sums = pair_sums[fits]
        
        # Calculate scores
        scores = valid_sums / (valid_waste ** 2)
        
        # Assign back to the matrix
        H[fits] = scores
        
        # Ensure all values are finite and positive
        # Replace any non-finite values with a small positive number
        H[~np.isfinite(H)] = 1e-9
        H[H <= 0] = 1e-9

    return H
