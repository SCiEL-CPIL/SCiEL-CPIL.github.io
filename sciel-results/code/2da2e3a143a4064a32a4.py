import numpy as np

def heuristics(prizes, weights):
    """
    Calculate heuristic attractiveness scores for items in a Multiple-Constraint Knapsack Problem.
    
    Args:
        prizes: np.array of shape (n_constraints, n_items)
        weights: np.array of shape (n_constraints, n_items)
        
    Returns:
        np.array of shape (n_items,) with positive heuristic values.
    """
    n_constraints, n_items = prizes.shape
    capacity = 25.0
    
    # 1. Calculate total prize for each item
    total_prizes = prizes.sum(axis=0)
    
    # 2. Calculate the bottleneck ratio for each item.
    # This represents the maximum fraction of any single constraint's capacity 
    # that the item would consume.
    # weights are in [0, 1], capacity is 25.
    # ratio_k_i = weights[k, i] / capacity
    # bottleneck_i = max_k(ratio_k_i)
    
    # To avoid division by zero or very small numbers if weights are 0, 
    # we add a small epsilon.
    eps = 1e-9
    
    # Normalize weights by capacity to get the fraction of capacity used
    normalized_weights = weights / capacity
    
    # Find the maximum fraction across constraints for each item
    bottleneck_ratios = np.max(normalized_weights, axis=0)
    
    # 3. Compute raw heuristic score: Prize / Bottleneck
    # This favors items that give high prize without critically straining any single constraint.
    raw_scores = total_prizes / (bottleneck_ratios + eps)
    
    # 4. Pareto Dominance Filtering
    # Items that are strictly worse than another item in all aspects should be penalized.
    # An item i is dominated by j if:
    # total_prizes[j] >= total_prizes[i]
    # weights[k, j] <= weights[k, i] for all k
    # and at least one inequality is strict.
    
    # For efficiency with 100 items, O(N^2) is fine.
    dominated = np.zeros(n_items, dtype=bool)
    
    for i in range(n_items):
        if dominated[i]:
            continue
        for j in range(n_items):
            if i == j:
                continue
            # Check if j dominates i
            # j has >= prize
            if total_prizes[j] < total_prizes[i]:
                continue
            # j has <= weight in all constraints
            if np.any(weights[:, j] > weights[:, i]):
                continue
            # Check if strictly better in at least one aspect
            # (Prize strictly higher OR any weight strictly lower)
            # Since prize is scalar sum, we check original prizes or just rely on the sum?
            # The task says "prizes and weights... uniform in [0,1]".
            # Usually dominance is defined on the vectors.
            # Let's check vector dominance to be safe.
            # If prize vector j >= prize vector i and weight vector j <= weight vector i (element-wise)
            
            # Actually, the objective is sum of prizes. So if sum(p_j) >= sum(p_i) and w_j <= w_i, 
            # it's not strictly guaranteed that j is better if p_j is [10, 0] and p_i is [5, 5] (sum 10 vs 10).
            # But if sum(p_j) > sum(p_i) and w_j <= w_i, j is definitely better.
            # If sum(p_j) == sum(p_i) and w_j < w_i (strictly in some component), j is better.
            
            # Simplified dominance for scalar objective:
            # If total_prizes[j] > total_prizes[i] and np.all(weights[:, j] <= weights[:, i]):
            #     dominated[i] = True
            #     break
            # elif total_prizes[j] == total_prizes[i] and np.all(weights[:, j] <= weights[:, i]) and np.any(weights[:, j] < weights[:, i]):
            #     dominated[i] = True
            #     break
            
            # Let's use a slightly more robust check:
            if total_prizes[j] > total_prizes[i]:
                if np.all(weights[:, j] <= weights[:, i]):
                    dominated[i] = True
                    break
            elif total_prizes[j] == total_prizes[i]:
                if np.all(weights[:, j] <= weights[:, i]) and np.any(weights[:, j] < weights[:, i]):
                    dominated[i] = True
                    break

    # Apply penalty to dominated items
    # A factor of 0.5 reduces their attractiveness significantly but doesn't eliminate them,
    # allowing the pheromone trail to potentially override if they were previously selected.
    penalty_factor = 0.5
    raw_scores[dominated] *= penalty_factor
    
    # 5. Final cleanup
    # Ensure all values are positive and finite.
    # The task states: "Heuristic entries that are not positive become 1e-9"
    # But we should ensure our calculation doesn't produce NaN or Inf.
    
    # If bottleneck is 0 (all weights 0), score is huge. 
    # However, weights are uniform in [0,1], so 0 is possible but unlikely for all 5 constraints.
    # If an item has 0 weight in all constraints, it's "free" and should have infinite attractiveness.
    # The formula P / (0 + eps) = P / eps = huge. This is correct.
    
    # Replace any non-positive or non-finite values with 1e-9 as per task spec
    # Although our formula should yield positive values if prizes are positive.
    # Prizes are uniform [0,1], so they can be 0.
    # If prize is 0, score is 0. Task says non-positive becomes 1e-9.
    
    valid_mask = (raw_scores > 0) & np.isfinite(raw_scores)
    raw_scores[~valid_mask] = 1e-9
    
    return raw_scores
