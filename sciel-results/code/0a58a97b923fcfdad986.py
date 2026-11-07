"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, weights):
    """
    Returns attractiveness scores for items in a Multiple-Constraint Knapsack Problem.
    
    Strategy: Competitive Exclusion / Opportunity Cost Score.
    We estimate the 'cost' of selecting an item by calculating the total prize
    of all other items that would become infeasible if this item were selected.
    This directly models the opportunity cost in a tight knapsack environment.
    
    1. Calculate total prize for each item (sum of prize rows).
    2. For each item i, determine which other items j cannot fit alongside i.
       An item j is incompatible with i if for any constraint c:
       weights[c, i] + weights[c, j] > capacity (25.0)
       Since capacities are fixed and relatively small compared to sum of weights,
       many items will be mutually exclusive.
    3. Sum the prizes of all incompatible items j for each i.
    4. Score = Prize_i / (Incompatible_Prize_Sum_i + epsilon)
    
    This heuristic favors items that have high prize but low "conflict" with the
    rest of the item pool, effectively identifying items that are "cheap" in terms
    of the alternatives they forego.
    """
    start_time = time.time()
    
    # prizes and weights have shape (5, 100)
    capacity = 25.0
    eps = 1e-9
    
    # 1. Total prize per item
    # The task description implies 'prize' is a single value, but input is (5,100).
    # We assume the objective is the sum of prizes across the 5 dimensions, 
    # consistent with the seed program's interpretation.
    total_prize = prizes.sum(axis=0)  # shape (100,)
    
    n_items = weights.shape[1]
    
    # 2 & 3. Calculate incompatible prize sum for each item
    # We need to check for each pair (i, j) if they are incompatible.
    # Incompatible if: exists c such that weights[c, i] + weights[c, j] > capacity
    
    # To do this efficiently, we can compute the sum of weights for all pairs.
    # weights is (5, 100).
    # Let W be weights. W[c, i] + W[c, j] > 25.
    
    # We can construct a matrix of incompatibility.
    # For each constraint c, create a matrix M_c where M_c[i, j] = 1 if W[c, i] + W[c, j] > 25.
    # Then an item i is incompatible with j if ANY M_c[i, j] is 1.
    
    incompatible_mask = np.zeros((n_items, n_items), dtype=bool)
    
    for c in range(5):
        # W_c is shape (100,)
        w_c = weights[c, :]
        # Sum of weights for all pairs
        # w_i + w_j > 25
        pair_sums = w_c[:, None] + w_c[None, :]
        incompatible_mask |= (pair_sums > capacity)
        
        # Early exit if time is running out? 
        # This loop is small (5 iterations of 100x100 ops), so it's fast.
        # But let's check time just in case.
        if time.time() - start_time > 290:
             break

    # Ensure diagonal is True (item is incompatible with itself? No, we want OTHERS)
    # The mask currently includes i=j. If w_i + w_i > 25, it's True.
    # We want sum of prizes of j != i where incompatible.
    # So we should zero out the diagonal before summing, or subtract self if needed.
    # Actually, if i is incompatible with itself, it doesn't mean it's "lost" by choosing i.
    # We only care about j != i.
    np.fill_diagonal(incompatible_mask, False)
    
    # Now, for each item i, sum the prizes of all j where incompatible_mask[i, j] is True.
    # incompatible_mask is (100, 100) boolean.
    # total_prize is (100,).
    
    # incompatible_prize_sums[i] = sum(total_prize[j] for j where incompatible_mask[i, j])
    # This is matrix multiplication: incompatible_mask @ total_prize
    
    incompatible_prize_sums = incompatible_mask.astype(float) @ total_prize
    
    # 4. Score
    # Higher prize and lower incompatible prize sum is better.
    # Score = total_prize / (incompatible_prize_sums + eps)
    
    scores = total_prize / (incompatible_prize_sums + eps)
    
    # Ensure positive values and finite
    scores = np.where(np.isfinite(scores), scores, 1e-9)
    scores = np.maximum(scores, 1e-9)
    
    # Final time check
    if time.time() - start_time > 295:
         return scores
        
    return scores
# EVOLVE-BLOCK-END
