import numpy as np
import time

def heuristics(demands, capacity):
    """
    Computes the heuristic matrix for the ACO bin packing algorithm.
    
    Combines:
    1. Best-fit tightness (inverse gap)
    2. Item size preference (larger items preferred to avoid stranding)
    3. Fill-ratio bonus (rewards pairs that fill bins well)
    4. Complementarity signal: scarcity of valid partners (inverse degree)
       Items with fewer compatible partners are more "urgent" to place,
       so placing them earlier (when more options exist) is better.
    """
    W = np.array(demands, dtype=float)
    n = len(W)
    
    # Calculate pair sums
    pair_sums = W[:, None] + W[None, :]
    
    # Calculate gaps
    gaps = capacity - pair_sums
    
    # Identify valid pairs (item j fits after item i)
    fits = gaps >= 0
    
    # Initialize heuristic matrix
    H = np.full((n, n), 1e-9)
    
    if np.any(fits):
        eps = 1e-9
        safe_gaps = np.where(fits, gaps + eps, 1.0)
        
        # Term 1: Best-fit tightness - inverse gap
        # Smaller gap = more attractive
        tightness = np.where(fits, 1.0 / safe_gaps, 1e-9)
        
        # Term 2: Item size - prefer larger items to place
        # W[None, :] gives the size of item j (the one being placed)
        item_size = W[None, :]
        
        # Term 3: Fill ratio bonus
        # Rewards pairs that fill a high fraction of capacity
        fill_ratio = np.where(fits, pair_sums / capacity, 0.0)
        fill_bonus = 1.0 + 0.3 * (fill_ratio ** 2)
        
        # Term 4: Complementarity / scarcity signal
        # For each item j, count how many other items can fit in a bin with it
        # (i.e., how many i satisfy W[i] + W[j] <= capacity)
        # Items with fewer compatible partners are scarcer and should be placed
        # with higher priority (when more choices exist)
        
        # degree[j] = number of items i such that W[i] + W[j] <= capacity
        # This is the column sum of the fits matrix
        degree = np.sum(fits, axis=0)  # shape (n,)
        
        # Inverse degree: items with fewer partners get higher score
        # Add small offset to avoid division by zero
        inverse_degree = 1.0 / (degree + 1.0)  # shape (n,)
        
        # Broadcast to (n, n) - same value for all i, varies by j
        scarcity_bonus = 1.0 + 0.5 * inverse_degree[None, :]
        
        # Combine all terms multiplicatively
        # Primary signal: tightness (strongest)
        # Secondary: item size, fill bonus, scarcity
        base = tightness * item_size * fill_bonus * scarcity_bonus
        
        # Apply only where fits
        H = np.where(fits, base, 1e-9)
    
    # Final safety checks
    H = np.where(np.isfinite(H), H, 1e-9)
    H = np.where(H > 0, H, 1e-9)
    
    return H
