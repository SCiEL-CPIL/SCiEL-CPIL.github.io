import numpy as np
import time

# Global deadline guard
_start_time = time.time()
_TIME_LIMIT = 140.0  # Leave 10s safety margin from 150s limit

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1
    
    # Check deadline
    if time.time() - _start_time > _TIME_LIMIT:
        return -1
    
    # Convert to numpy arrays if not already
    if not isinstance(remaining_weights, np.ndarray):
        remaining_weights = np.array(remaining_weights)
    if not isinstance(remaining_values, np.ndarray):
        remaining_values = np.array(remaining_values)
    
    # Find feasible items
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    fw = remaining_weights[feasible_indices]
    fv = remaining_values[feasible_indices]
    
    # Strategy: Use Lagrangian relaxation to find optimal lambda (shadow price)
    # For a given lambda, the "profit" of item i is v_i - lambda * w_i
    # We want to choose lambda such that the sum of weights of items with positive profit
    # approximately equals remaining_capacity.
    
    # Binary search for lambda
    # Lower bound: 0
    # Upper bound: max(v/w) among feasible items
    if len(fw) == 0:
        return -1
    
    densities = fv / fw
    lambda_low = 0.0
    lambda_high = np.max(densities) * 1.1
    
    # Find lambda such that sum of weights where v_i - lambda*w_i > 0 ~= remaining_capacity
    # This is a monotonic decreasing function of lambda (as lambda increases, fewer items have positive profit)
    
    def total_weight_at_lambda(lam):
        profits = fv - lam * fw
        positive_mask = profits > 0
        return np.sum(fw[positive_mask])
    
    # Binary search for lambda
    for _ in range(50):  # 50 iterations for good precision
        lambda_mid = (lambda_low + lambda_high) / 2.0
        tw = total_weight_at_lambda(lambda_mid)
        if tw > remaining_capacity:
            lambda_low = lambda_mid
        else:
            lambda_high = lambda_mid
    
    lambda_opt = (lambda_low + lambda_high) / 2.0
    
    # Now, select the item that maximizes the Lagrangian objective
    # But we need to be careful: the item we select must be feasible,
    # and we want to pick one that leads to a good total value.
    
    # The standard approach in Lagrangian relaxation for knapsack is:
    # 1. Find lambda* such that the fractional solution is optimal
    # 2. The items with v_i - lambda* * w_i > 0 are "in" the solution
    # 3. If the sum of weights is not exactly capacity, there's a critical item
    
    # For our purpose, we want to pick the next item to add.
    # We can score each feasible item by its contribution to the total value
    # given that we will greedily fill the rest with density-sorted items.
    
    # But a simpler and effective heuristic: pick the item that maximizes
    # v_i - lambda_opt * w_i, but with a tie-breaker or adjustment.
    
    # Actually, the best heuristic is:
    # For each feasible item i, estimate the total value if we pick i first,
    # then greedily fill the remaining capacity with density-sorted items.
    
    # But that's O(n^2 log n) which might be slow. Let's try a faster approach.
    
    # Approach: Use the Lagrangian profit as a primary score.
    # The item with the highest v_i - lambda_opt * w_i is the one that
    # is most "valuable" per unit of the constrained resource at the optimal shadow price.
    
    # However, this might not be optimal for 0/1 knapsack. Let's combine
    # with a density-based lookahead.
    
    # For each feasible item, compute:
    # score_i = v_i + greedy_fill_value(remaining_capacity - w_i, excluding item i)
    
    # To make this fast, we pre-sort all feasible items by density.
    
    # Pre-compute density-sorted order for all feasible items
    order = np.argsort(densities)[::-1]
    sorted_fw = fw[order]
    sorted_fv = fv[order]
    
    # Now, for each candidate item, we need to compute the greedy fill
    # of the remaining capacity excluding that item.
    
    # This is still O(n^2) in the worst case. Let's see if we can optimize.
    
    # Alternative: Just use the Lagrangian score directly.
    # This is O(n) and might be good enough.
    
    # Let's try: pick the item that maximizes v_i - lambda_opt * w_i
    # among feasible items.
    
    profits = fv - lambda_opt * fw
    best_local_idx = np.argmax(profits)
    
    # But we need to map back to the original index
    best_index = feasible_indices[best_local_idx]
    
    # Sanity check: make sure the item is actually feasible
    if remaining_weights[best_index] > remaining_capacity:
        return -1
    
    return int(best_index)
