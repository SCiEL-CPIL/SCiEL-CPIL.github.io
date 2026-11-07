"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(demands, capacity):
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    # Convert to numpy array for vectorized operations
    W = np.array(demands, dtype=float)
    n = len(W)
    
    # Calculate sum of all pairs
    pair_sums = W[:, None] + W[None, :]
    
    # Calculate remaining capacity (gap) for each pair
    # If pair_sum > capacity, gap is negative (invalid)
    gaps = capacity - pair_sums
    
    # Initialize heuristic matrix
    # We want high values for small positive gaps (tight fit)
    # For invalid pairs (gap <= 0), we use a very small value
    # For valid pairs, we use 1 / (gap + epsilon)
    # This creates a strong preference for items that fill the bin completely
    
    epsilon = 1e-9
    # np.divide handles the division, we use where to mask invalid cases
    # We only want to divide where gap > 0
    # If gap is 0, 1/epsilon is huge. This is good, it's the best move.
    
    # To avoid warnings and ensure finite values:
    # We can compute the inverse gap safely
    inv_gaps = np.full_like(pair_sums, 1e-9)
    
    # Mask for valid pairs (gap > 0)
    valid_mask = gaps > 0
    
    # Compute 1 / (gap + epsilon) for valid pairs
    # We add epsilon to avoid division by zero if gap is exactly 0, 
    # but since valid_mask is gap > 0, gap is at least some small positive number 
    # (unless float precision makes it 0 for exact fit).
    # If exact fit, gap=0. valid_mask is False. 
    # We should treat exact fit as the best possible.
    # So, let's include gap == 0 in the "good" category.
    
    # Revised mask: gap >= 0
    fit_mask = gaps >= 0
    
    # For fit_mask, compute attractiveness
    # If gap is 0, attractiveness is very high.
    # If gap is large, attractiveness is low.
    
    # Using np.where with a safe division
    # We can't just do 1/gaps because of zeros.
    # We can do:
    # safe_gaps = np.where(fit_mask, gaps + epsilon, 1.0)
    # values = 1.0 / safe_gaps
    # result = np.where(fit_mask, values, 1e-9)
    
    safe_gaps = np.where(fit_mask, gaps + epsilon, 1.0)
    attractiveness = 1.0 / safe_gaps
    
    # Ensure invalid pairs are 1e-9
    H = np.where(fit_mask, attractiveness, 1e-9)
    
    # Check deadline (unlikely to be hit, but required by directive)
    if time.time() > deadline:
        # Fallback to simple size-based heuristic if something went wrong
        return np.where(fit_mask, W[None, :], 1e-9)
        
    return H
# EVOLVE-BLOCK-END
