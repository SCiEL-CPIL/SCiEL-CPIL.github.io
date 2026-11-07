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
    remaining = capacity - pair_sums
    
    eta = np.full((n, n), 1e-9, dtype=np.float64)
    
    if np.any(fits):
        # Heuristic: 
        # We want to reward placing larger items (d_j) into tight fits.
        # The term (remaining + 1)^-2 creates a sharp preference for minimal waste.
        # d_j ensures that larger items are prioritized for placement if they fit tightly.
        # This is a "value per waste" metric, squared to emphasize tightness.
        
        # Avoid division by zero (though remaining >= 0, so remaining+1 >= 1)
        # d[None, :] broadcasts item j's size to all rows i.
        # We only apply this where fits is True.
        
        # To avoid warnings, we compute on the masked array or use where
        # We use np.where to be safe, though indexing is faster.
        
        # Let's use indexing for speed and clarity
        # d_j is the column index, so we need d[None, :] to broadcast across rows
        # remaining is (n, n)
        
        # Compute values for fitting pairs
        # Note: remaining[fits] gives a 1D array of remaining spaces for valid pairs.
        # We need the corresponding d_j values.
        # It's easier to compute the full matrix and then mask, or use advanced indexing.
        
        # Full matrix calculation (fast in numpy)
        # denom = (remaining + 1.0) ** 2
        # values = d[None, :] / denom
        
        # This creates an (n, n) array.
        denom = (remaining + 1.0) ** 2
        # Prevent division by zero just in case, though remaining >= 0
        # np.divide handles this well
        values = np.divide(d[None, :], denom, out=np.zeros_like(denom), where=denom>0)
        
        # Apply mask: only keep values where fits is True
        eta = np.where(fits, values, 1e-9)
    
    # Sanity check for finite values
    if not np.all(np.isfinite(eta)):
        eta = np.nan_to_num(eta, nan=1e-9, posinf=1e-9, neginf=1e-9)
        
    return eta
# EVOLVE-BLOCK-END
