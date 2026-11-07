"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    """
    Computes the heuristic matrix for the ACO bin packing algorithm.
    
    The heuristic eta[i, j] represents the attractiveness of placing item j
    immediately after item i in the same bin.
    
    Strategy:
    We design the heuristic to guide ants toward bin configurations that:
    1. Fit within capacity.
    2. Reward pairs that create "dead gaps" too small for any remaining item.
       This is done by making eta inversely proportional to the count of items
       that could still fit into the residual space capacity - (d_i + d_j).
       If no other item fits (dead gap), the count is 0, so we use a large value.
       If many items could fit, the count is high, so we use a smaller value.
    3. Also factor in the combined demand (d_i + d_j) to prefer larger items,
       as they are harder to place and contribute more to bin utilization.
    
    Formula:
    For fitting pairs:
        remaining = capacity - (d_i + d_j)
        count_fit = number of items k (k != i, k != j) such that d_k <= remaining
        
        If count_fit == 0 (dead gap):
            eta[i,j] = (d_i + d_j) * 10.0  # strong reward
        Else:
            eta[i,j] = (d_i + d_j) / (1.0 + count_fit)
    
    For non-fitting pairs: eta[i,j] = 1e-9
    """
    demands = np.asarray(demands, dtype=float)
    n = len(demands)
    
    # Compute pairwise sums
    pair_sums = demands[:, None] + demands[None, :]
    
    # Determine which pairs fit in the bin
    fits = pair_sums <= capacity
    
    # Compute remaining space for fitting pairs
    remaining = capacity - pair_sums
    
    # Initialize heuristic matrix
    eta = np.full((n, n), 1e-9, dtype=float)
    
    # For each fitting pair, count how many other items could still fit in the remaining space
    # We do this efficiently using vectorized operations
    if np.any(fits):
        # For each pair (i, j), count items k where demands[k] <= remaining[i,j]
        # and k is not i and k is not j
        # We can use searchsorted for efficiency, but let's use a simple approach first
        
        # Create a mask for each pair: which items could fit
        # remaining has shape (n, n), demands has shape (n,)
        # We want for each (i, j), count of k where demands[k] <= remaining[i, j]
        
        # Use broadcasting: remaining[:, :, None] >= demands[None, None, :]
        # This would be n x n x n, which might be too large for n=120 (1.7M * 8 bytes = ~13MB, acceptable)
        # But let's try a more memory-efficient approach using searchsorted
        
        # Sort demands for searchsorted
        sorted_demands = np.sort(demands)
        
        # For each (i, j), find the count of items with demand <= remaining[i, j]
        # Using searchsorted: np.searchsorted(sorted_demands, remaining[i,j], side='right')
        # This gives the count of items <= remaining[i,j]
        
        # But we need to exclude items i and j themselves
        # This is tricky with searchsorted. Let's use the broadcasting approach for n=120
        # n=120, so n^3 = 1,728,000, which is manageable
        
        # Create a 3D boolean array: can_fit[i, j, k] = (demands[k] <= remaining[i, j])
        # But this is n^3 = 1.7M booleans, ~1.7MB, acceptable
        
        # Actually, let's do it more carefully. We want for each (i, j), the count of k
        # such that demands[k] <= remaining[i, j] AND k != i AND k != j
        
        # Let's use a loop over i and j for clarity, but vectorize over k
        # Or use the 3D approach
        
        # 3D approach:
        # remaining_expanded = remaining[:, :, None]  # shape (n, n, 1)
        # demands_expanded = demands[None, None, :]   # shape (1, 1, n)
        # can_fit = demands_expanded <= remaining_expanded  # shape (n, n, n)
        
        # Then for each (i, j), count = sum(can_fit[i, j, :]) - (1 if demands[i] <= remaining[i,j] else 0) - (1 if demands[j] <= remaining[i,j] else 0)
        # But we need to be careful: if i == j, we only subtract once
        
        # Let's do this:
        remaining_expanded = remaining[:, :, None]  # shape (n, n, 1)
        demands_expanded = demands[None, None, :]   # shape (1, 1, n)
        can_fit = demands_expanded <= remaining_expanded  # shape (n, n, n)
        
        # Count for each (i, j)
        count_fit = np.sum(can_fit, axis=2)  # shape (n, n)
        
        # Now subtract the self-references
        # For pair (i, j), we need to subtract 1 if demands[i] <= remaining[i, j] (item i could fit)
        # and 1 if demands[j] <= remaining[i, j] (item j could fit)
        # But note: item i is already in the bin, so we should not count it as "available"
        # Similarly for item j
        
        # Create indices
        idx = np.arange(n)
        
        # Subtract self-references
        # For each i, j: subtract 1 if demands[i] <= remaining[i, j]
        self_i = demands[idx[:, None]] <= remaining  # shape (n, n): is item i <= remaining[i,j]?
        self_j = demands[idx[None, :]] <= remaining  # shape (n, n): is item j <= remaining[i,j]?
        
        # But wait, when i == j, we're subtracting twice for the same item
        # Let's handle this:
        # count_adjusted = count_fit - self_i - self_j + (i == j) * (demands[i] <= remaining[i,j])
        # Actually, if i == j, then self_i and self_j are the same, so we subtracted twice
        # but we only need to subtract once. So we add back 1 if i == j and demands[i] <= remaining[i,j]
        
        diag_mask = (idx[:, None] == idx[None, :])  # shape (n, n), True where i == j
        self_i_bool = demands[idx[:, None]] <= remaining
        self_j_bool = demands[idx[None, :]] <= remaining
        
        count_adjusted = count_fit - self_i_bool.astype(float) - self_j_bool.astype(float)
        # Add back for diagonal where we double-subtracted
        count_adjusted += diag_mask * self_i_bool.astype(float)
        
        # Ensure non-negative
        count_adjusted = np.maximum(count_adjusted, 0)
        
        # Now compute eta for fitting pairs
        # For fitting pairs where count_adjusted == 0 (dead gap), give a high value
        # For fitting pairs where count_adjusted > 0, give (d_i + d_j) / (1 + count_adjusted)
        
        fitting_mask = fits
        dead_gap = fitting_mask & (count_adjusted == 0)
        non_dead_gap = fitting_mask & (count_adjusted > 0)
        
        # For dead gaps: strong reward
        if np.any(dead_gap):
            eta[dead_gap] = pair_sums[dead_gap] * 10.0
        
        # For non-dead gaps: standard formula
        if np.any(non_dead_gap):
            eta[non_dead_gap] = pair_sums[non_dead_gap] / (1.0 + count_adjusted[non_dead_gap])
    
    return eta
# EVOLVE-BLOCK-END
