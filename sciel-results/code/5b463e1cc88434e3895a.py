"""Improved heuristics for ACO multiple-constraint knapsack."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, weights):
    """
    prizes: (5, 100) uniform [0,1]
    weights: (5, 100) uniform [0,1]
    Returns: (100,) attractiveness scores, larger = more attractive.
    
    Strategy:
    - Compute a "feasibility margin" for each item: how much capacity it
      consumes relative to the total capacity. Items that are "cheap" in
      terms of capacity usage relative to their prize are better.
    - Use a soft-min (log-sum-exp) over constraints to get a smooth measure
      of the worst-case constraint violation, which guides the heuristic.
    - Also incorporate a greedy ratio: prize / (sum of weights) as a base.
    - Blend: high prize, low weight consumption, and good ratio.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    n_items = weights.shape[1]
    n_constraints = weights.shape[0]
    capacity = 25.0
    
    # Base ratio: prize per unit of total weight
    total_weight_per_item = weights.sum(axis=0)  # (100,)
    total_prize_per_item = prizes.sum(axis=0)    # (100,)
    
    # Avoid division by zero
    eps = 1e-9
    ratio = total_prize_per_item / (total_weight_per_item + eps)
    
    # Soft-min over constraints: for each item, compute a smooth measure of
    # how "tight" the worst constraint is. We want items that don't strain
    # any single constraint too much.
    # For each constraint c and item i, the fraction of capacity used is weights[c,i]/capacity.
    # A high fraction means that constraint is nearly saturated by this item alone.
    # We use log-sum-exp to get a smooth max over constraints of the weight fraction.
    
    # weight_fractions[c, i] = weights[c, i] / capacity
    weight_fractions = weights / capacity  # (5, 100)
    
    # Smooth max via log-sum-exp: for each item, compute soft_max_c(weight_fractions[c, i])
    # This gives a smooth surrogate for the worst constraint strain.
    # We want to PENALIZE high soft_max, so we'll use negative or inverse.
    
    # Temperature for log-sum-exp (lower = sharper, closer to hard max)
    beta = 10.0
    # scaled = beta * weight_fractions
    scaled = beta * weight_fractions
    # log-sum-exp per item
    max_scaled = np.max(scaled, axis=0)  # (100,)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled[:, np.newaxis]), axis=0))
    soft_max = log_sum_exp / beta  # (100,) - smooth max of weight fractions
    
    # Items with lower soft_max are less constraining -> more attractive
    # Use inverse: 1 / (soft_max + eps)
    constraint_score = 1.0 / (soft_max + eps)
    
    # Prize score: normalize prizes (sum over constraints) to [0,1]
    prize_max = total_prize_per_item.max()
    if prize_max > eps:
        prize_score = total_prize_per_item / prize_max
    else:
        prize_score = np.ones(n_items)
    
    # Ratio score: normalize ratio
    ratio_max = ratio.max()
    if ratio_max > eps:
        ratio_score = ratio / ratio_max
    else:
        ratio_score = np.ones(n_items)
    
    # Weight efficiency: items with lower total weight are better
    weight_min = total_weight_per_item.min()
    weight_max = total_weight_per_item.max()
    if weight_max > weight_min + eps:
        weight_score = 1.0 - (total_weight_per_item - weight_min) / (weight_max - weight_min)
    else:
        weight_score = np.ones(n_items)
    
    # Blend the scores with weights
    # Emphasize ratio (prize/weight) and prize, penalize constraint strain
    heuristic = (
        1.0 * prize_score +
        1.5 * ratio_score +
        1.0 * constraint_score +
        0.5 * weight_score
    )
    
    # Ensure positivity
    heuristic = np.maximum(heuristic, eps)
    
    return heuristic
# EVOLVE-BLOCK-END
