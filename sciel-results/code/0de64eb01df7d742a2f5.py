"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    # prizes: (5, 100), weights: (5, 100)
    # We want a score per item (100,)
    
    # 1. Calculate the total prize sum for each constraint dimension
    # prizes has shape (5, 100). Summing along axis=1 gives total prize per constraint (5,)
    # Actually, we want the sum of prizes of ALL items for each constraint dimension.
    # Since every item contributes to every constraint's "competition pool" in a global sense?
    # No, the task says: "prize relative to the sum of prizes of all other items that would violate the same constraint if selected"
    
    # Let's interpret "items that would violate the same constraint if selected".
    # In a knapsack, adding item j to a knapsack that is near capacity for constraint c 
    # means we might have to drop another item i that also uses constraint c.
    # A simple proxy for "competition" in constraint c is the total prize of all items 
    # that have non-zero weight in constraint c. Since weights are uniform in [0,1], 
    # all items have non-zero weight in all constraints (almost surely).
    
    # So for each item j and each constraint c:
    # Total prize in constraint c = sum(prizes[c, :])
    # My prize in constraint c = prizes[c, j]
    # Other prizes in constraint c = sum(prizes[c, :]) - prizes[c, j]
    
    # The "competitive exclusion score" idea:
    # If I pick item j, I gain prizes[c, j] in constraint c.
    # But I consume weights[c, j] capacity.
    # The "opportunity cost" is related to how much value I could have gotten 
    # from other items using that capacity.
    
    # Let's try a formulation:
    # Score_j = sum_c [ prizes[c, j] / (sum_k prizes[c, k] * weights[c, k] / weights[c, j] + epsilon) ]
    # This is getting complicated.
    
    # Let's stick to the prompt's specific phrasing:
    # "item's prize relative to the sum of prizes of all other items that would violate the same constraint if selected"
    
    # Interpretation A:
    # For item j, look at constraint c.
    # The "other items" are all k != j.
    # Sum of their prizes in c: S_c = sum_{k != j} prizes[c, k]
    # Ratio: prizes[c, j] / S_c
    # This measures how dominant item j is in constraint c.
    
    # Interpretation B (Marginal Utility / Opportunity Cost):
    # Value density relative to competition.
    # Let's compute for each item j:
    # total_prize_c = sum(prizes[c, :])
    # other_prize_c = total_prize_c - prizes[c, j]
    # score_c_j = prizes[c, j] / (other_prize_c + 1e-9)
    # Final score = sum_c score_c_j
    
    # This rewards items that have high prize relative to the rest of the pool in each dimension.
    # It effectively downweights items that are "average" and upweights "top" items in each dimension.
    
    total_prizes = np.sum(prizes, axis=1) # shape (5,)
    
    # For each item j and constraint c:
    # other_prizes = total_prizes[c] - prizes[c, j]
    # ratio = prizes[c, j] / (other_prizes + 1e-9)
    
    # Vectorized:
    # prizes is (5, 100)
    # total_prizes is (5,) -> reshape to (5, 1)
    total_prizes_col = total_prizes[:, np.newaxis]
    
    other_prizes = total_prizes_col - prizes # shape (5, 100)
    
    ratios = prizes / (other_prizes + 1e-9) # shape (5, 100)
    
    # Sum across constraints to get a single score per item
    scores = np.sum(ratios, axis=0)
    
    # Ensure positivity and finite values
    scores = np.where(np.isfinite(scores) & (scores > 0), scores, 1e-9)
    
    return scores
# EVOLVE-BLOCK-END
