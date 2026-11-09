"""Rewritten program with Pareto-dominance-based heuristic for ACO knapsack."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items using a multi-objective dominance filter.
    
    Approach:
    1. For each item, compute its prize and its "effective cost" in the most constraining dimension.
    2. Identify Pareto-optimal items: an item is Pareto-optimal if no other item has
       >= prize in all dimensions AND strictly better in at least one, while having
       <= weight in all constraints AND strictly less in at least one.
    
    Since we have 5 prize dimensions and 5 weight constraints, we use a simplified
    dominance: an item i dominates item j if:
      - prize_i >= prize_j for all 5 dimensions (or total prize_i >= total prize_j)
      - weight_i <= weight_j for all 5 constraints
      - and at least one strict inequality holds
    
    We assign high scores to Pareto-optimal items and low scores to dominated ones.
    To make the heuristic informative for the ACO, we also incorporate a ratio-based
    component but only among Pareto-optimal items.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    n_items = prizes.shape[1]
    n_constraints = prizes.shape[0]
    
    # Total prize per item (sum across dimensions)
    total_prize = prizes.sum(axis=0)  # shape (n_items,)
    
    # For dominance: we use total prize and total weight as proxy,
    # but also consider per-constraint weights.
    # An item i dominates j if:
    #   total_prize[i] >= total_prize[j] AND
    #   weights[k, i] <= weights[k, j] for all k
    #   AND at least one strict inequality
    
    # Efficient dominance check using numpy broadcasting
    # prize_ge[i,j] = total_prize[i] >= total_prize[j]
    prize_ge = total_prize[:, None] >= total_prize[None, :]  # (n, n)
    
    # weight_le[i,j] = weights[:, i] <= weights[:, j] for all constraints
    # weights shape: (n_constraints, n_items)
    # We want: for each pair (i, j), check if weights[k, i] <= weights[k, j] for all k
    # weights[:, i] is a vector of length n_constraints
    # Compare element-wise: weights[:, i] <= weights[:, j]
    # This can be done as: (weights[:, None, :] <= weights[:, :, None]) -> (n_constraints, n_items, n_items)
    weight_le_all = np.all(weights[:, None, :] <= weights[:, :, None], axis=0)  # (n_items, n_items)
    
    # Strict in prize or weight
    prize_strict = total_prize[:, None] > total_prize[None, :]
    weight_strict = np.any(weights[:, None, :] < weights[:, :, None], axis=0)
    
    # i dominates j if prize_ge[i,j] AND weight_le_all[i,j] AND (prize_strict[i,j] OR weight_strict[i,j])
    dominates = prize_ge & weight_le_all & (prize_strict | weight_strict)
    
    # An item is Pareto-optimal if no other item dominates it
    # dominated_by[i] = any j != i such that j dominates i
    # dominates matrix: dominates[i, j] means i dominates j
    # So item j is dominated if any i (i != j) has dominates[i, j] = True
    dominated = np.any(dominates[:, :], axis=0)  # For each j, is there any i that dominates j?
    # Wait, need to exclude self-comparison. dominates[i,i] could be True if there are equal prizes and weights.
    # Let's set diagonal to False
    np.fill_diagonal(dominates, False)
    dominated = np.any(dominates, axis=0)  # dominated[j] = True if any i dominates j
    
    pareto_mask = ~dominated  # True for Pareto-optimal items
    
    # Assign scores:
    # Pareto-optimal items get high scores based on their prize-to-weight ratio
    # Dominated items get a small base score
    
    # Compute a ratio score for all items
    # Use the minimum weight across constraints as a proxy for "cost"
    min_weight = weights.min(axis=0)  # shape (n_items,)
    ratio = total_prize / (min_weight + 1e-9)
    
    # Normalize ratio to [0, 1]
    ratio_min = ratio.min()
    ratio_max = ratio.max()
    if ratio_max > ratio_min:
        ratio_norm = (ratio - ratio_min) / (ratio_max - ratio_min)
    else:
        ratio_norm = np.ones(n_items) * 0.5
    
    # Base score for all items
    scores = np.full(n_items, 0.1)
    
    # For Pareto-optimal items, assign higher scores based on normalized ratio
    scores[pareto_mask] = 1.0 + ratio_norm[pareto_mask]
    
    # Ensure all scores are positive
    scores = np.maximum(scores, 1e-9)
    
    return scores
# EVOLVE-BLOCK-END
