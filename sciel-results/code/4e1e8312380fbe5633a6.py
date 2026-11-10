"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# We keep a module-level cache to track state across calls within the same instance
# since the function is called sequentially to build the solution.
# However, the interface is stateless. We must rely on the inputs.
# The inputs are the *remaining* capacity, weights, and values.
# This implies that the items already picked are removed from the arrays.
# We don't have the full history, so we can't do complex DP on the fly easily without
# re-deriving the state, which is impossible without the original full set.
#
# Therefore, we must design a heuristic that is "greedy" but smarter than pure density.
#
# Strategies for 0/1 Knapsack heuristics:
# 1. Value Density (v/w): Current seed.
# 2. Value (v): Good for high value, low weight items.
# 3. Weight (w): Avoid heavy items.
# 4. Combined Score: A weighted combination of value, weight, and density.
# 5. "Fit" metric: How well the item fits into the remaining capacity.
#    If remaining_capacity is small, we should prefer items that fit exactly or are small.
#    If remaining_capacity is large, we can afford heavier items.
#
# Let's analyze the problem parameters:
# - 100 items.
# - Capacity 12.5.
# - Weights and Values are generated. Typical distributions might be uniform or normal.
# - If weights are uniform [0, 1], average weight 0.5. Capacity 12.5 allows ~25 items.
# - If weights are larger, fewer items.
#
# A common improvement over pure density is to consider the "efficiency" of filling the remaining space.
# If an item has high density but is too heavy to fit well with other high-density items, it might block better combinations.
# However, without lookahead, density is a strong baseline.
#
# Let's try a hybrid score:
# score = value * (1 + alpha * (1 - weight / remaining_capacity))
# This penalizes items that consume a large fraction of the remaining capacity, encouraging smaller items that leave room for others.
# But wait, if the item is small, it has low value.
#
# Alternative:
# score = value / weight  (density)
# But adjust for "waste".
# waste = remaining_capacity - weight
# If waste is very large, maybe we should pick a bigger item?
#
# Actually, a very effective simple heuristic for knapsack is:
# 1. Sort by density.
# 2. Pick the first feasible item.
# This is what the seed does (argmax density among feasible).
#
# Can we do better?
# Consider the case where two items have similar density.
# Item A: w=1, v=10 (density 10)
# Item B: w=1.1, v=11 (density 10)
# Both fit. Picking A leaves 11.5. Picking B leaves 11.4.
#
# What if we look at the *next* best item?
# This requires lookahead, which is expensive if done naively (O(n^2) per step, total O(n^3) or worse).
# With 100 items, O(n^2) per step is 10,000 ops. 100 steps -> 1,000,000 ops. This is fast in Python/NumPy if vectorized.
#
# Let's try a 1-step lookahead heuristic.
# For each feasible item i, calculate the total value if we pick i and then pick the best feasible item j (by density) from the remaining items (excluding i).
# Score_i = v_i + max(0, v_j) where j is the best density item among remaining \ {i} that fits in (cap - w_i).
#
# This is a greedy approximation of the 2-step optimal path.
#
# Implementation:
# 1. Identify feasible items (w_i <= cap).
# 2. For each feasible item i:
#    - new_cap = cap - w_i
#    - Find the best item j (by density) among all items except i that fits in new_cap.
#    - If no such j exists, lookahead_value = 0.
#    - Total_value_i = v_i + lookahead_value
# 3. Pick the item i with the highest Total_value_i.
#
# Complexity:
# N items.
# For each i (N), we scan N items to find best j.
# Total O(N^2) per call.
# Number of calls ~ N (since we pick one item per call until none fit).
# Total O(N^3) = 1,000,000 operations.
# This is well within the 120s limit.
#
# Let's refine the "best item j".
# We should pick j based on density? Or value?
# Density is generally better for packing.
# So, for a given remaining capacity C', the best next item is the one with max density among those with w <= C'.
#
# We can pre-compute densities for all items.
#
# Algorithm:
# 1. Compute densities d = v / w.
# 2. Sort items by density descending? No, we need to evaluate each feasible i.
# 3. For each feasible i:
#    - C' = cap - w_i
#    - Find max density j such that j != i and w_j <= C'.
#    - If found, val = v_i + v_j. Else val = v_i.
# 4. Return argmax val.
#
# Optimization:
# Instead of scanning all N items for each i, we can sort items by density.
# Let items be sorted by density: idx_0, idx_1, ..., idx_{N-1}.
# For a given C', the best item is the first one in this sorted list that has weight <= C' (and index != i).
# We can pre-compute a prefix minimum of weights? No, we need the first one that fits.
# Actually, we just need to find the highest density item that fits.
# If we have the items sorted by density, we can iterate through them until we find one that fits.
# In the worst case, we iterate through all N.
# So it's still O(N) per i, total O(N^2).
#
# Let's implement this.
#
# Edge cases:
# - If no item fits, return -1.
# - If only one item fits, return it.
#
# Let's add a safety margin for time.
# If N is large, O(N^2) might be slow in pure Python loops.
# We should use NumPy vectorization where possible.
#
# Vectorized approach for 1-step lookahead:
# 1. Mask feasible items: feasible = w <= cap.
# 2. If no feasible items, return -1.
# 3. For each feasible item i, we want to find the best j.
#    This is hard to vectorize directly because the condition "j != i" and "w_j <= cap - w_i" varies per i.
#
# However, we can approximate.
# If we assume that the best j is independent of i (which is false), we'd just pick the best density item.
# But the "j != i" constraint and the changing capacity make it dependent.
#
# Let's stick to a Python loop for the outer item i, but use NumPy for the inner search?
# Or just pure Python if N=100 is small enough.
# 100 * 100 = 10,000 iterations per call.
# 100 calls * 10,000 = 1,000,000 iterations.
# Python can do ~10^7 simple ops per second. So 0.1s. This is very safe.
#
# Let's write the code.

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1
    
    # Convert to lists for easier indexing if needed, or keep as arrays
    w = np.array(remaining_weights)
    v = np.array(remaining_values)
    cap = remaining_capacity
    
    # Check feasibility
    feasible_mask = w <= cap
    if not np.any(feasible_mask):
        return -1
    
    # Get indices of feasible items
    feasible_indices = np.where(feasible_mask)[0]
    
    # Precompute densities for all items
    # Avoid division by zero, though weights should be > 0
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = np.where(w > 0, v / w, 0)
    
    # Sort all items by density descending to help find best j quickly
    # But we need to map back to original indices
    # Actually, for the inner loop, we just need to find the max density item that fits.
    # We can just scan the feasible items for j.
    
    best_score = -1
    best_idx = -1
    
    # Optimization: If we just pick the best density item, that's the seed.
    # The lookahead adds the value of the *next* best item.
    
    # To speed up, we can sort feasible items by density?
    # No, we need to evaluate each feasible i as a candidate for the *first* pick.
    
    # Let's iterate through feasible items as potential first picks
    for i in feasible_indices:
        v_i = v[i]
        w_i = w[i]
        remaining_cap_after_i = cap - w_i
        
        # Find the best second item j
        # j must be feasible in the new capacity: w_j <= remaining_cap_after_i
        # j != i
        # We want to maximize v_j (or density? Let's try value first, or density?)
        # Using density for the second pick is consistent with the greedy strategy.
        # Let's try maximizing value for the second pick, as we have a fixed remaining capacity.
        # Actually, density is better for packing. Let's stick to density for the second pick too.
        
        best_second_value = 0
        # Find j
        # We can vectorize this search for j for a given i?
        # mask_j = (w <= remaining_cap_after_i) & (np.arange(n) != i)
        # if any(mask_j):
        #     candidate_indices = np.where(mask_j)[0]
        #     best_j_idx = candidate_indices[np.argmax(densities[candidate_indices])]
        #     best_second_value = v[best_j_idx]
        
        # This creates a mask of size n for each i. O(n) per i. Total O(n^2).
        # Let's try it.
        
        if remaining_cap_after_i >= 0:
            # Create mask for j
            # Note: w is the full array
            mask_j = (w <= remaining_cap_after_i)
            mask_j[i] = False  # Exclude i
            
            if np.any(mask_j):
                # Get densities of candidates
                cand_densities = densities[mask_j]
                # Find index of max density
                # argmax returns index in the flattened boolean array, we need to map back
                # It's easier to just get the indices
                cand_indices = np.where(mask_j)[0]
                # Find the one with max density
                # We can use argmax on the densities of the candidates
                # But we need the original index to get the value
                # Let's just find the max density value
                max_dens = np.max(cand_densities)
                # There might be ties, but any max is fine
                # Get the value of that item
                # We need to find which candidate had that density
                # Actually, we just need the value.
                # Is the item with max density also the one with max value? Not necessarily.
                # But for the heuristic "pick best density", we assume the item with max density is the best next pick.
                # So we take the value of that item.
                # Let's find the index in cand_indices that corresponds to max_dens
                # Or simpler:
                # best_j_idx = cand_indices[np.argmax(cand_densities)]
                # best_second_value = v[best_j_idx]
                
                # Let's do it directly
                best_j_local_idx = np.argmax(cand_densities)
                best_j_orig_idx = cand_indices[best_j_local_idx]
                best_second_value = v[best_j_orig_idx]
        
        total_score = v_i + best_second_value
        
        if total_score > best_score:
            best_score = total_score
            best_idx = i
    
    return int(best_idx)
# EVOLVE-BLOCK-END
