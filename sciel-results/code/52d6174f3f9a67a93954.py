"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: the evaluator kills after 150s. We want to stay well under.
# Since each call is fast, we track a global deadline to be safe.
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # 10s safety margin


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Pick the next item index (into the remaining arrays) to add, or -1 to stop.
    
    Strategy:
    1. Filter to feasible items (weight <= remaining_capacity).
    2. If none feasible, return -1.
    3. Use a lookahead heuristic: for each feasible item, compute a score that
       balances immediate value density with future potential. We use a 
       modified value-density that accounts for the fraction of capacity 
       the item consumes, and also considers the "regret" of not picking 
       high-value items.
    
    A practical strong heuristic for 0/1 knapsack with small capacity:
    - Compute value/weight density for each feasible item.
    - Also consider: for each item, the "opportunity cost" is the total value
      of items we can no longer take after picking it.
    - A simpler strong heuristic: pick the item that maximizes 
      (value + expected future value gain), where expected future value gain
      is approximated by the sum of values of items that would still fit 
      after picking this item, weighted by their density.
    
    For speed with 100 items, we do a 1-step lookahead:
    For each candidate item i:
      - remaining_cap_after = remaining_capacity - w[i]
      - If remaining_cap_after < 0, skip (infeasible)
      - score[i] = v[i] + sum of v[j] for j != i where w[j] <= remaining_cap_after,
        but we can't sum all because that's O(n^2). Instead, approximate:
      - We use: score[i] = v[i] + (remaining_cap_after / total_remaining_weight_after) * total_remaining_value_after
        This is a rough proxy for how much more value we can extract.
    
    Actually, a cleaner and well-known strong heuristic is:
    - For each feasible item, compute the "density" v/w.
    - But also penalize items that consume a large fraction of remaining capacity
      unless their value is proportionally high.
    
    Let's use: score = v[i] * (1 + remaining_capacity / w[i])
    This rewards items with high value and also those that leave more capacity.
    Wait, that's not right either.
    
    Better approach: use the "surrogate" method. The classic strong heuristic
    for knapsack is to sort by value/weight and greedily pick. But we can do
    a 1-step lookahead to improve.
    
    For each feasible item i:
      cap_after = remaining_capacity - w[i]
      # Compute the best we can do with cap_after from the remaining items (excluding i)
      # This is expensive. Instead, approximate by:
      #   approx_future = sum of v[j] for j where w[j] <= cap_after, sorted by density,
      #   but take only the top ones that fit.
    
    Given n=100, we can afford O(n^2) per call. Let's do:
    For each feasible item i:
      cap_after = remaining_capacity - w[i]
      if cap_after < 0: continue
      # From the remaining items (excluding i), greedily pick by density to fill cap_after
      # This gives a lower bound on future value.
      # score[i] = v[i] + greedy_future_value(i, cap_after)
    
    This is O(n^2 log n) per call, which is fine for n=100.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        return -1
    
    weights = np.asarray(remaining_weights, dtype=np.float64)
    values = np.asarray(remaining_values, dtype=np.float64)
    
    n = len(weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    # Filter to feasible items
    feasible_mask = weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    feasible_weights = weights[feasible_mask]
    feasible_values = values[feasible_mask]
    
    # If only one feasible item, pick it
    if len(feasible_indices) == 1:
        return int(feasible_indices[0])
    
    # For each feasible item, compute a lookahead score
    best_score = -1
    best_idx = -1
    
    # Precompute densities for all feasible items
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = np.where(feasible_weights > 0, feasible_values / feasible_weights, 0)
    
    # Sort feasible items by density descending for greedy future computation
    order = np.argsort(-densities)
    
    for local_i in range(len(feasible_indices)):
        # Skip if deadline approaching
        if time.time() > _DEADLINE - 0.5:
            break
        
        w_i = feasible_weights[local_i]
        v_i = feasible_values[local_i]
        cap_after = remaining_capacity - w_i
        
        if cap_after < 0:
            continue
        
        # Compute greedy future value from remaining items (excluding local_i)
        # We iterate through items sorted by density, skipping local_i
        future_value = 0.0
        remaining_cap = cap_after
        for local_j in order:
            if local_j == local_i:
                continue
            if feasible_weights[local_j] <= remaining_cap:
                future_value += feasible_values[local_j]
                remaining_cap -= feasible_weights[local_j]
            if remaining_cap <= 0:
                break
        
        total_score = v_i + future_value
        
        if total_score > best_score:
            best_score = total_score
            best_idx = int(feasible_indices[local_i])
    
    if best_idx == -1:
        # Fallback: pick highest density feasible item
        best_density = -1
        best_idx = -1
        for local_i in range(len(feasible_indices)):
            if densities[local_i] > best_density:
                best_density = densities[local_i]
                best_idx = int(feasible_indices[local_i])
    
    return best_idx
# EVOLVE-BLOCK-END
