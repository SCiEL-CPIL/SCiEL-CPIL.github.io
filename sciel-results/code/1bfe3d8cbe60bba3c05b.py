import numpy as np
import time

# We track time to ensure we don't exceed the 150s limit, though this O(N log N) 
# approach is extremely fast. The UCB approach avoids the O(N^2) or O(N^3) 
# simulation costs of the previous strategy.

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    weights = np.asarray(remaining_weights, dtype=float)
    values = np.asarray(remaining_values, dtype=float)
    
    n = len(weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    # Filter out items that don't fit? 
    # The interface implies we return an index into the *remaining* arrays.
    # If an item doesn't fit, picking it is invalid. 
    # Usually, the caller handles infeasibility, but the prompt says 
    # "picks the next item to add... or -1 to stop".
    # If we pick an item that doesn't fit, the state might become invalid 
    # or the caller might reject it. 
    # However, standard 0/1 knapsack heuristics usually only consider feasible items.
    # Let's assume we should only return indices of items that fit.
    # If no items fit, return -1.
    
    feasible_mask = weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
    
    # Get indices of feasible items
    feasible_indices = np.where(feasible_mask)[0]
    f_weights = weights[feasible_mask]
    f_values = values[feasible_mask]
    n_feasible = len(f_weights)
    
    if n_feasible == 0:
        return -1

    # UCB Strategy:
    # We model each feasible item as an arm.
    # Reward component 1: Value Density (exploitation of high value/weight)
    # Reward component 2: Capacity Synergy (exploitation of items that fit well, 
    #                     leaving "good" remaining capacity for others)
    # Exploration: We don't have historical counts in this stateless function.
    # However, in a single-step selection context, "exploration" can be 
    # interpreted as considering items that are not the absolute best density 
    # but might unlock better subsequent combinations due to weight granularity.
    
    # Since we cannot maintain state across calls (stateless function), 
    # we must define a deterministic "exploration" or use a proxy for it.
    # A common proxy in stateless heuristics for "uncertainty" or "potential synergy" 
    # is how "tight" the fit is or how unique the weight is.
    # Alternatively, we can use a deterministic tie-breaker or a score that 
    # penalizes items that are "too heavy" relative to remaining capacity 
    # (which might block many subsequent items) vs items that are "light".
    
    # Let's define the UCB score as:
    # Score = Value_Density + Exploration_Bonus
    # Exploration_Bonus here is a proxy for "potential to combine well".
    # Items with weights that are a "nice" fraction of the remaining capacity 
    # might be better? Or items that are lighter?
    
    # Actually, a better structural change is to use a "Lookahead-free" 
    # composite score that approximates the marginal gain.
    # Let's stick to the UCB inspiration:
    # Exploitation: Value / Weight
    # Exploration: A term that favors items with weights that are distinct 
    #              or that leave a capacity that is divisible by common small weights?
    #              This is hard to compute fast.
    
    # Let's try a different "Bandit" framing:
    # The "Reward" of picking item i is not just its value, but how much 
    # *additional* value it enables. 
    # Since we can't simulate, we approximate the "enabled value" by 
    # assuming the remaining capacity will be filled optimally.
    # Optimal fill value ~ (remaining_capacity - w_i) * avg_density.
    # So Total Est Value = v_i + (C - w_i) * avg_density.
    # This is equivalent to maximizing v_i - w_i * avg_density.
    # This is just a shifted density score.
    
    # To make it a true UCB with exploration, we need a variance term.
    # In a stateless setting, we can use the "surprise" of the item's weight 
    # relative to the distribution of remaining weights.
    # Items with weights that are "common" might be safer (lower variance in 
    # how they pack), while "rare" weights might be risky but high reward?
    
    # Let's use a simpler, robust heuristic that fits the "Bandit" description 
    # by balancing immediate value (exploitation) against the "risk" of 
    # wasting capacity (exploration/penalty).
    
    # Calculate mean density of the current pool
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = np.where(f_weights > 0, f_values / f_weights, 0)
    
    # Filter out zero density items if possible, or just handle them.
    mean_density = np.mean(densities)
    
    # UCB Formula:
    # Score = Density + c * sqrt(ln(N) / n_i)
    # Since n_i (counts) is 1 for all items in a stateless view, the 
    # exploration term is constant for all items, which makes it useless 
    # for ranking.
    
    # Therefore, we must define a *different* exploration metric.
    # Let's define "Exploration" as the potential for *weight synergy*.
    # Synergy Score: How well does this item's weight complement the *other* items?
    # Approximation: An item is "synergistic" if its weight is small enough 
    # to allow many other items to fit, OR if it fills a "gap" in the 
    # weight distribution.
    
    # A computationally cheap proxy for synergy:
    # 1. High Density (Exploitation)
    # 2. Low Weight relative to Capacity (Exploration/Flexibility)
    #    Lighter items leave more capacity for future picks, which is 
    #    generally "safer" (lower regret) in 0/1 knapsack.
    
    # Score = alpha * (Value / Weight) + beta * (1 - Weight / Capacity)
    
    # Let's tune alpha and beta.
    # If we pick high density, we get value.
    # If we pick low weight, we preserve capacity for future high density items.
    
    alpha = 1.0
    beta = 0.5 # Exploration/Flexibility weight
    
    # Normalize densities to [0, 1] to balance with the weight term?
    max_density = np.max(densities)
    if max_density > 0:
        norm_densities = densities / max_density
    else:
        norm_densities = densities
    
    # Weight term: (1 - w/C) is in [0, 1]
    weight_term = 1.0 - (f_weights / remaining_capacity)
    
    # Composite Score
    scores = alpha * norm_densities + beta * weight_term
    
    # Select the best scoring item
    best_local_idx = np.argmax(scores)
    
    # Map back to original index
    best_global_idx = feasible_indices[best_local_idx]
    
    return int(best_global_idx)
