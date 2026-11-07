"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multi-constraint knapsack.
    
    Strategy: competitive exclusion score based on opportunity cost.
    For each item, we estimate how 'expensive' it is in terms of the prizes
    of other items it would exclude from each constraint.
    
    Args:
        prizes: array of shape (5, 100) - prize values for each constraint/item
        weights: array of shape (5, 100) - weight/cost for each constraint/item
    
    Returns:
        array of shape (100,) - attractiveness score per item (larger = more attractive)
    """
    # Validate inputs
    if prizes.shape != weights.shape:
        raise ValueError("prizes and weights must have the same shape")
    
    n_constraints, n_items = prizes.shape
    
    # Convert to float for numerical stability
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # For each item j, compute a score that reflects its marginal utility
    # relative to the opportunity cost across all constraints.
    
    # Idea: For item j, in constraint c, if we pick item j, we use weights[c, j] of capacity.
    # The "opportunity cost" is roughly the sum of prizes of other items that would
    # be excluded due to this weight usage.
    
    # A simpler, more direct approach:
    # 1. Compute prize per unit weight for each constraint: prize_density[c, j] = prizes[c, j] / weights[c, j]
    # 2. The "competitive exclusion" idea: an item is attractive if it has high prize density
    #    but low weight (so it excludes fewer other items).
    
    # Let's try a different angle based on the directive:
    # "item's prize relative to the sum of prizes of all other items that would violate 
    #  the same constraint if selected"
    
    # Interpretation: For item j in constraint c, consider all other items i != j.
    # If we select item j, it uses weights[c, j] capacity. Items i that have weights[c, i] > 0
    # are all "competing" for the same capacity.
    
    # A practical heuristic: 
    # For each item j, compute:
    #   score[j] = sum over constraints c of (prizes[c, j] / (sum of weights[c, :] * prizes[c, :] / sum of prizes[c, :]))
    # This normalizes by how "expensive" the item is relative to the total prize-weight tradeoff.
    
    # Even simpler and likely effective:
    # Use the ratio of prize to total weight across all constraints, but weighted by 
    # how constrained each constraint is.
    
    # Let me try the competitive exclusion more directly:
    # For item j, in constraint c:
    #   - Its "cost" in terms of excluded prize potential is proportional to weights[c, j]
    #   - The total prize available in constraint c is sum(prizes[c, :])
    #   - So the fraction of prize potential it "blocks" is approximately weights[c, j] * (mean prize per unit weight)
    
    # Actually, let's use a cleaner formulation:
    # score[j] = sum_c [ prizes[c, j] / (1 + weights[c, j] * avg_prize_per_unit_weight[c]) ]
    
    # Where avg_prize_per_unit_weight[c] = sum(prizes[c, :]) / sum(weights[c, :])
    
    # This way, items with high prize and low weight get high scores, and the denominator
    # accounts for the opportunity cost in each constraint.
    
    # Compute average prize per unit weight for each constraint
    sum_prizes_per_c = prizes.sum(axis=1)  # shape (n_constraints,)
    sum_weights_per_c = weights.sum(axis=1)  # shape (n_constraints,)
    
    # Avoid division by zero
    sum_weights_per_c_safe = np.where(sum_weights_per_c > 1e-12, sum_weights_per_c, 1.0)
    
    avg_prize_per_unit_weight = sum_prizes_per_c / sum_weights_per_c_safe  # shape (n_constraints,)
    
    # For each item j and constraint c:
    #   contribution[c, j] = prizes[c, j] / (1 + weights[c, j] * avg_prize_per_unit_weight[c])
    
    # Vectorized:
    # weights has shape (n_constraints, n_items)
    # avg_prize_per_unit_weight has shape (n_constraints,)
    # We need to broadcast avg_prize_per_unit_weight to (n_constraints, n_items)
    
    opp_cost = weights * avg_prize_per_unit_weight[:, np.newaxis]  # shape (n_constraints, n_items)
    
    contributions = prizes / (1.0 + opp_cost)  # shape (n_constraints, n_items)
    
    # Sum over constraints to get final score per item
    scores = contributions.sum(axis=0)  # shape (n_items,)
    
    # Ensure all scores are positive and finite
    scores = np.where(np.isfinite(scores) & (scores > 0), scores, 1e-9)
    
    return scores
# EVOLVE-BLOCK-END
