"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    n = len(demands)
    pair_sum = demands[:, None] + demands[None, :]
    gap = capacity - pair_sum
    
    # Initialize with very small value for invalid pairs
    H = np.full((n, n), 1e-9)
    
    # Valid pairs are those where both items fit together
    valid = gap >= 0
    
    # For each valid pair (i, j), compute how many other items could fit
    # into the remaining space gap[i,j] = capacity - d_i - d_j
    # This rewards pairs that leave "dead gaps" (too small for any remaining item)
    # or exact fits, rather than large unused voids.
    
    # For each pair (i,j) with gap >= 0, count items k (k != i, k != j) where d_k <= gap
    # We can do this efficiently using sorted demands
    sorted_demands = np.sort(demands)
    
    # For each pair (i,j), the residual space is gap[i,j]
    # Count how many items fit: items with demand <= gap[i,j]
    # We need to exclude items i and j themselves from this count
    
    # Efficient approach: for each possible gap value, count items that fit
    # Since demands are small integers (20-99) and capacity is 150,
    # gap values range from 0 to 130 (150 - 20 - 20 = 110 max, but min is 0)
    
    # Build a cumulative count of items by demand size
    # demands are floats but represent integers 20-99
    demand_ints = np.round(demands).astype(int)
    
    # Count how many items have demand <= each possible value
    max_demand = int(np.max(demand_ints))
    min_demand_val = int(np.min(demand_ints))
    
    # Create a frequency array
    freq = np.zeros(max_demand + 1, dtype=np.int64)
    for d in demand_ints:
        freq[d] += 1
    
    # Cumulative sum: cumsum[k] = number of items with demand <= k
    cumsum = np.cumsum(freq)
    
    # Now for each pair (i,j), compute how many items fit in gap[i,j]
    # gap[i,j] = capacity - d_i - d_j
    # Number of items with demand <= gap[i,j] = cumsum[gap[i,j]] if gap[i,j] <= max_demand
    # But we need to exclude items i and j themselves
    
    # For efficiency, let's compute this for all valid pairs
    H[valid] = 0.0
    
    # For each valid pair, compute the count of fitting items
    # We'll do this in a vectorized way
    gap_valid = gap[valid]
    
    # Clip gap to valid range for cumsum lookup
    gap_clipped = np.clip(gap_valid, 0, max_demand).astype(int)
    
    # Count of items with demand <= gap
    fit_counts = cumsum[gap_clipped]
    
    # For each pair (i,j), subtract 1 if item i fits in its own gap, and 1 if item j fits
    # Item i has demand d_i. It fits in gap[i,j] if d_i <= gap[i,j]
    # Similarly for item j
    
    # Get indices of valid pairs
    valid_indices = np.where(valid)
    i_idx = valid_indices[0]
    j_idx = valid_indices[1]
    
    # Check if item i fits in gap[i,j]: d_i <= gap[i,j]
    # d_i <= gap[i,j] <=> d_i <= capacity - d_i - d_j <=> 2*d_i + d_j <= capacity
    i_fits = (2.0 * demands[i_idx] + demands[j_idx]) <= capacity
    # Check if item j fits in gap[i,j]: d_j <= gap[i,j]
    # d_j <= gap[i,j] <=> d_j <= capacity - d_i - d_j <=> d_i + 2*d_j <= capacity
    j_fits = (demands[i_idx] + 2.0 * demands[j_idx]) <= capacity
    
    # Adjust fit_counts: subtract 1 for each item that fits and is i or j
    adjusted_counts = fit_counts - i_fits.astype(np.int64) - j_fits.astype(np.int64)
    
    # Ensure counts are non-negative
    adjusted_counts = np.maximum(adjusted_counts, 0)
    
    # The heuristic should reward pairs with FEWER fitting items in the residual
    # So we use inverse of (count + 1) to avoid division by zero
    # Pairs with 0 fitting items (dead gap or exact fit) get the highest score
    H[valid] = 1.0 / (adjusted_counts + 1.0)
    
    # Boost exact fits even more
    exact_fit = valid & (gap == 0)
    H[exact_fit] = 10.0
    
    return H
# EVOLVE-BLOCK-END
