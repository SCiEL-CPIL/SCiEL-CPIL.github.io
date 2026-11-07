"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(demands, capacity):
    """
    Computes a heuristic matrix for the ant colony optimizer.
    
    Strategy:
    Reward tight packing. The heuristic eta[i, j] is high if placing item j 
    after item i results in a bin with very little remaining space.
    This guides ants to fill bins efficiently, reducing the total number of bins.
    
    eta[i, j] = 1 / (remaining_space + 1) if demands[i] + demands[j] <= capacity
    eta[i, j] = 1e-9 otherwise
    """
    # Track time to ensure we don't hit the 300s limit, though this is very fast.
    start_time = time.time()
    
    d = np.asarray(demands, dtype=np.float64)
    n = len(d)
    
    # Calculate pair sums
    pair_sums = d[:, None] + d[None, :]
    
    # Determine which pairs fit in the bin
    fits = pair_sums <= capacity
    
    # Calculate remaining space for fitting pairs
    # For non-fitting pairs, we set remaining to a large value to make eta small,
    # but we will mask them out anyway.
    remaining = capacity - pair_sums
    
    # Heuristic: Inverse of (remaining + 1)
    # +1 prevents division by zero if a perfect fit occurs (remaining=0).
    eta = np.zeros((n, n), dtype=np.float64)
    
    # Apply formula only where fits is True
    # remaining + 1.0 ensures denominator is >= 1.0
    eta[fits] = 1.0 / (remaining[fits] + 1.0)
    
    # Ensure non-fitting entries are effectively 0 (1e-9 as per task spec for non-positive)
    eta[~fits] = 1e-9
    
    # Sanity check for finite values
    if not np.all(np.isfinite(eta)):
        eta = np.nan_to_num(eta, nan=1e-9, posinf=1e-9, neginf=1e-9)
        
    return eta
# EVOLVE-BLOCK-END
