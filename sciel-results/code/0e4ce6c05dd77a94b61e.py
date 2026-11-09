"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(demands, capacity):
    """
    Computes the heuristic matrix for the ant colony optimization.
    Combines the structural idea of favoring large-large pairs (product)
    with a refinement based on how well the pair fills the bin (sum/capacity).
    
    The heuristic encourages:
    1. Pairing items that fit together.
    2. Among fitting pairs, preferring those with larger product (large items).
    3. Among those, preferring those that fill the bin more completely (higher sum).
    
    We use a combined score: product * (sum / capacity) for fitting pairs.
    This is equivalent to (d_i * d_j * (d_i + d_j)) / capacity.
    For non-fitting pairs, we use 1e-9.
    """
    start_time = time.time()
    
    # Convert to numpy array for vectorized operations
    d = np.asarray(demands, dtype=np.float64)
    n = len(d)
    
    if n == 0:
        return np.empty((0, 0))
        
    # Calculate sums to check if items fit together
    sums = d[:, None] + d[None, :]
    
    # Calculate products to measure "compatibility" (favoring large-large pairs)
    products = d[:, None] * d[None, :]
    
    # Condition: items must fit together
    fits = sums <= capacity
    
    # Combined heuristic:
    # For fitting pairs: product * (sum / capacity)
    # This emphasizes both large items and good bin utilization.
    # For non-fitting pairs: 1e-9
    
    # We can compute the combined score for all pairs, then mask with fits
    # score = products * (sums / capacity)
    # But we need to be careful: for non-fitting pairs, we want 1e-9, not the computed value.
    
    # Compute the raw score for all pairs
    raw_scores = products * (sums / capacity)
    
    # Apply the fit condition
    H = np.where(fits, raw_scores, 1e-9)
    
    # Ensure no non-finite values
    if not np.all(np.isfinite(H)):
        # If any non-finite values exist, replace them with 1e-9
        H = np.where(np.isfinite(H), H, 1e-9)
    
    # Ensure shape is correct
    if H.shape != (n, n):
        raise ValueError("Wrong shape")
    
    # Time guard
    elapsed = time.time() - start_time
    if elapsed > 325:
        # Safety margin: if we're close to the limit, return immediately
        # (though this computation is O(n^2) with n=120, it's instant)
        pass
    
    return H
# EVOLVE-BLOCK-END
