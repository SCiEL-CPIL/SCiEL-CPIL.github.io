"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(demands, capacity):
    """
    Computes the heuristic matrix for the ACO bin packing problem.
    
    The heuristic eta[i, j] represents the attractiveness of placing item j
    immediately after item i in the same bin.
    
    Strategy:
    - We favor pairs (i, j) that have a combined size close to the bin capacity (tight fit).
    - Additionally, we favor placing larger items (j) earlier or in tighter slots,
      as larger items are harder to place later (aligning with First Fit Decreasing logic).
    - The heuristic is defined as demand[j] / (remaining_space + 1) for valid pairs,
      where remaining_space = capacity - (demand[i] + demand[j]).
    - Invalid pairs (where demand[i] + demand[j] > capacity) or diagonal entries
      are set to a very small value (1e-9) to make them effectively unselectable.
    """
    start_time = time.time()
    
    # Convert to numpy array for vectorized operations
    d = np.asarray(demands, dtype=np.float64)
    n = len(d)
    
    # Compute the sum of demands for all pairs (i, j)
    # pair_sums[i, j] = d[i] + d[j]
    pair_sums = d[:, None] + d[None, :]
    
    # Compute the gap (remaining capacity) for all pairs
    # gap[i, j] = capacity - pair_sums[i, j]
    gap = capacity - pair_sums
    
    # Identify valid pairs:
    # 1. The pair must fit in the bin: gap >= 0
    # 2. It must not be the same item (i != j)
    valid_mask = (gap >= 0)
    # Ensure diagonal is excluded from valid calculation to be strict
    np.fill_diagonal(valid_mask, False)
    
    # Calculate candidate heuristic values
    # We want to favor:
    # 1. Small gaps (tight fit): 1 / (gap + 1)
    # 2. Large items j: d[j]
    # Combined: d[j] / (gap + 1)
    
    # To avoid division by zero or issues, gap >= 0, so gap + 1 >= 1.
    # We create a matrix of d[j] to multiply.
    # d[None, :] broadcasts to shape (n, n) where each row is the same d vector.
    d_j_matrix = d[None, :]
    
    # Compute the raw heuristic values for all positions
    # For invalid positions, we will mask them out later.
    # We use gap + 1 as denominator.
    # For safety, if gap is negative (invalid), we still compute, but mask later.
    # However, to avoid inf/nan, we ensure denominator is at least 1? 
    # gap can be negative. e.g. -10. gap+1 = -9. 
    # d[j] / -9 is negative. 
    # We will mask these out.
    
    raw_H = d_j_matrix / (gap + 1.0)
    
    # Apply the mask: keep calculated values for valid pairs, 1e-9 for others
    H = np.where(valid_mask, raw_H, 1e-9)
    
    # Ensure no non-finite values exist (defensive check)
    H = np.where(np.isfinite(H), H, 1e-9)
    
    # Check deadline guard
    elapsed = time.time() - start_time
    if elapsed > 290:
        return H
        
    return H
# EVOLVE-BLOCK-END
