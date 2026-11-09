"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(demands, capacity):
    """
    Computes the heuristic matrix for the Ant Colony Optimization.
    Heuristic value h[i][j] represents the attractiveness of placing item j
    after item i in the same bin.
    
    We use an 'inverse waste' heuristic:
    If item i and item j fit in a bin, the attractiveness is proportional to
    1 / (remaining_space). This strongly favors pairs that fill the bin tightly.
    """
    # Start time for deadline guard (though this function is fast)
    start_time = time.time()
    
    d = np.array(demands, dtype=float)
    n = len(d)
    
    # Compute pair sums
    pair_sums = d[:, None] + d[None, :]
    
    # Compute remaining space in the bin if i and j are placed together
    remaining = capacity - pair_sums
    
    # Initialize heuristic matrix with the fallback value
    # The problem states: "Heuristic entries that are not positive become 1e-9"
    # We will set invalid pairs (remaining < 0) to 1e-9.
    h = np.full((n, n), 1e-9)
    
    # Valid pairs are those where remaining space is non-negative
    valid_mask = remaining >= 0
    
    # For valid pairs, calculate heuristic
    # We add a small epsilon to avoid division by zero for perfect fits (remaining=0)
    # and to ensure values are positive.
    eps = 1e-9
    valid_remaining = remaining[valid_mask] + eps
    
    # Calculate 1 / remaining
    # This results in very high values for tight fits and low values for loose fits.
    h[valid_mask] = 1.0 / valid_remaining
    
    # Simple deadline guard check
    # If this function were part of a longer computation, we would check time here.
    # Since it's O(N^2) and N=120, it's negligible.
    if time.time() - start_time > 290:
         # In a real scenario, we might return a simpler heuristic or raise an error.
         # But since it's so fast, this is just a formality.
         pass

    return h
# EVOLVE-BLOCK-END
