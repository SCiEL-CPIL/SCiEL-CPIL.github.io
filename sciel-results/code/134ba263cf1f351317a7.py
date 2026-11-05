"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# We'll use a simple memoization/caching approach with a time budget.
# The function is called repeatedly as items are added. We need to be fast.
#
# Strategy:
# 1. If remaining_capacity is small relative to the smallest item weight, we might still fit something.
# 2. We want to maximize value. A pure density heuristic (value/weight) is a good start.
# 3. However, sometimes taking a slightly lower density item now allows taking a very high value item later
#    that wouldn't fit if we took the highest density item.
# 4. But since this is a greedy "next item" selection without backtracking in the main loop,
#    we can only make a local decision.
#
# A slightly better heuristic:
# - Compute density = value / weight
# - Also consider the "efficiency" of filling the remaining space.
# - If an item's weight is very close to the remaining capacity, it's a very good candidate
#   because it uses the space efficiently (high fill rate).
#
# Let's define a score:
# score = density * (1 + alpha * fill_rate)
# where fill_rate = weight / remaining_capacity
#
# This boosts items that fill a large portion of the remaining capacity.
#
# However, we need to be careful: if an item is too heavy to fit, it shouldn't be considered.
# But the interface says "remaining (feasible) arrays", which might imply all items in the arrays
# are feasible? Let's check the task description:
# "index into the remaining (feasible) arrays"
# This suggests that the arrays passed in already contain only items that fit?
# Or does it mean we return an index into the arrays, and we must ensure it fits?
# The seed program just does argmax(v/w) without checking capacity.
# Let's assume we MUST check if the item fits, or the arrays are pre-filtered.
# Given the name "remaining_capacity", it's likely we need to check fit.
# But if the arrays are "feasible", maybe they are already filtered?
# Let's look at the seed: it doesn't filter. It just returns argmax.
# If I pick an item that doesn't fit, the evaluator might reject it or it might cause an error.
# The task says "0/1 knapsack", "picks the next item to add".
# Usually, in such heuristics, you must ensure the item fits.
# Let's assume we should only consider items that fit.
# If no item fits, return -1.

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    if len(remaining_weights) == 0:
        return -1
    
    weights = np.asarray(remaining_weights)
    values = np.asarray(remaining_values)
    
    # Filter to only items that fit
    feasible_mask = weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
    
    feasible_weights = weights[feasible_mask]
    feasible_values = values[feasible_mask]
    feasible_indices = np.where(feasible_mask)[0]
    
    # Calculate density
    densities = feasible_values / feasible_weights
    
    # Calculate fill rate: how much of the remaining capacity this item uses
    fill_rates = feasible_weights / remaining_capacity
    
    # We want high density and high fill rate.
    # Let's try a combined score.
    # A common approach is to weight density heavily, but boost items that fill the space well.
    
    # Idea: If an item fills > 50% of the capacity, it's a strong candidate.
    # Let's try: score = density * (1 + beta * fill_rate)
    # with beta = 1.0
    
    beta = 1.0
    scores = densities * (1.0 + beta * fill_rates)
    
    # Another idea: Look ahead?
    # If we pick item i, what is the best we can do next?
    # This is expensive.
    
    # Let's stick to the composite score for now.
    # To break ties or handle edge cases, we can also consider absolute value.
    # But density usually dominates.
    
    # Let's try a different weighting:
    # score = density^alpha * fill_rate^beta
    # This might be more robust.
    
    # Actually, the previous best was 1.0 with pure density.
    # The inspiration 1 (0.9934) used lookahead.
    # Let's try a simple lookahead of depth 1 or 2 if time permits, but we must be fast.
    
    # Since we don't know the time budget precisely, let's stick to a fast heuristic.
    # The composite score is O(N).
    
    best_idx_in_feasible = int(np.argmax(scores))
    return int(feasible_indices[best_idx_in_feasible])
# EVOLVE-BLOCK-END
