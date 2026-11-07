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
    - We favor pairs (i, j) that leave a residual space (gap) too small for any
      other item to fit. This creates "dead gaps" that prevent further packing
      into that bin, effectively closing it sooner and reducing total bins.
    - For each pair (i, j), we count how many items k (k != i, k != j) have
      demand[k] <= gap[i, j].
    - The heuristic is inversely proportional to this count (plus a small offset
      to avoid division by zero).
    - If the count is 0 (dead gap), the heuristic is very high.
    - If the count is large (lots of room), the heuristic is low.
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
    
    # For each pair (i, j), count how many items k (k != i, k != j) fit in the gap
    # fit_count[i, j] = number of k such that k != i, k != j, and d[k] <= gap[i, j]
    
    # We can compute this efficiently using broadcasting
    # d_flat = d.reshape(1, 1, n)
    # gap_expanded = gap[:, :, None]
    # k_indices = np.arange(n).reshape(1, 1, n)
    # i_indices = np.arange(n).reshape(n, 1, 1)
    # j_indices = np.arange(n).reshape(1, n, 1)
    
    # But this creates an n x n x n array which might be large for n=120 (1.7M entries, fine)
    # Let's try a more memory-efficient approach if possible, but n=120 is small.
    
    # Create masks for k != i and k != j
    # i_mask: shape (n, n, 1) -> broadcast to (n, n, n) where k != i
    # j_mask: shape (n, n, 1) -> broadcast to (n, n, n) where k != j
    
    i_idx = np.arange(n).reshape(n, 1, 1)
    j_idx = np.arange(n).reshape(1, n, 1)
    k_idx = np.arange(n).reshape(1, 1, n)
    
    # d_k: shape (1, 1, n)
    d_k = d.reshape(1, 1, n)
    
    # gap_ij: shape (n, n, 1)
    gap_ij = gap[:, :, None]
    
    # Condition: d[k] <= gap[i, j]
    fits = d_k <= gap_ij  # shape (n, n, n)
    
    # Exclude k == i and k == j
    # k != i: k_idx != i_idx
    # k != j: k_idx != j_idx
    not_i = k_idx != i_idx
    not_j = k_idx != j_idx
    
    # Count fits where k != i and k != j
    # fits * not_i * not_j
    count = np.sum(fits & not_i & not_j, axis=2)  # shape (n, n)
    
    # Heuristic: inversely proportional to count
    # Use 1 / (count + 1) to avoid division by zero and ensure positive values
    # A higher value means fewer items fit in the gap, which is more attractive
    candidate_H = 1.0 / (count + 1.0)
    
    # Apply the mask: keep calculated values for valid pairs, 1e-9 for others
    H = np.where(valid_mask, candidate_H, 1e-9)
    
    # Ensure no non-finite values exist (defensive check)
    H = np.where(np.isfinite(H), H, 1e-9)
    
    # Check deadline guard
    elapsed = time.time() - start_time
    if elapsed > 290:
        return H
        
    return H
# EVOLVE-BLOCK-END
