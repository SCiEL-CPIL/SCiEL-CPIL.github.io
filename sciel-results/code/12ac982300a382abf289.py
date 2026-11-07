"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items using a marginal utility heuristic
    that accounts for resource scarcity across multiple constraints.
    
    prizes: (5, 100) array of item prizes
    weights: (5, 100) array of item weights per constraint
    
    Returns: (100,) array of heuristic scores (larger = more attractive)
    """
    # Validate inputs
    if prizes.shape != (5, 100) or weights.shape != (5, 100):
        raise ValueError("Invalid input shapes")
    
    if not np.all(np.isfinite(prizes)) or not np.all(np.isfinite(weights)):
        raise ValueError("Non-finite values in input")
    
    n_constraints, n_items = prizes.shape
    
    # Compute efficiency: prize per unit weight for each constraint
    # eff[i, j] = prize_j / weight_{i,j}
    # Items with zero weight in a constraint get high efficiency for that constraint
    
    # For each constraint, compute the scarcity-adjusted value
    # Strategy: For each item, compute how "expensive" it is relative to the
    # best available items in terms of resource consumption
    
    # Step 1: Compute per-constraint efficiency (prize / weight)
    # Handle zero weights by setting efficiency to a large value
    eff = np.zeros_like(prizes)
    for i in range(n_constraints):
        w = weights[i]
        p = prizes[i]
        mask = w > 1e-12
        eff[i, mask] = p[mask] / w[mask]
        eff[i, ~mask] = np.max(p[mask]) if np.any(mask) else 0.0
    
    # Step 2: For each constraint, rank items by efficiency and compute
    # a scarcity multiplier. Items that are among the top-k most efficient
    # for a scarce resource get boosted.
    
    # The total capacity is 25 per constraint.
    # Estimate how many items we can fit: roughly capacity / average_weight
    # But a better approach: use the distribution of weights to estimate
    # how many items of each type can be selected.
    
    # Alternative approach: Compute a "marginal value" that balances
    # prize against the opportunity cost of consuming scarce resources.
    
    # For each item j and constraint i:
    # opportunity_cost_{i,j} = sum of prizes of items that would be displaced
    # by item j in constraint i, weighted by scarcity.
    
    # Simplified: Use the ratio of this item's weight to the median weight
    # among high-efficiency items as a scarcity indicator.
    
    # Compute median weight for each constraint
    med_weight = np.median(weights, axis=1)  # shape (5,)
    
    # For each item, compute a "resource cost score" across all constraints
    # cost_{i,j} = weights[i,j] / med_weight[i]
    # This normalizes weight by the typical item weight
    
    cost = weights / med_weight[:, np.newaxis]  # shape (5, 100)
    
    # The item's total resource burden is the sum of normalized costs
    total_cost = cost.sum(axis=0)  # shape (100,)
    
    # Step 3: Compute a scarcity-aware prize
    # For each constraint, the "shadow price" is higher when the constraint
    # is tighter. Estimate tightness by the ratio of total weight to capacity.
    
    total_weight_per_constraint = weights.sum(axis=1)  # shape (5,)
    capacity = 25.0
    
    # Tightness: how much of the capacity would be consumed if we took all items
    # A better measure: expected utilization if we select items greedily by efficiency
    
    # For each constraint, simulate greedy selection to estimate how "scarce" it is
    scarcity = np.zeros(n_constraints)
    for i in range(n_constraints):
        # Sort items by efficiency for this constraint
        order = np.argsort(-eff[i])
        remaining_cap = capacity
        count_selected = 0
        for idx in order:
            w = weights[i, idx]
            if remaining_cap >= w:
                remaining_cap -= w
                count_selected += 1
            else:
                break
        # Scarcity: 1 if all items fit, less if not
        # Use the fraction of items that can be selected
        scarcity[i] = count_selected / n_items
    
    # Step 4: Compute heuristic as prize adjusted by scarcity
    # High prize items in scarce constraints get boosted
    # Low cost items get boosted
    
    # For each item, compute a weighted sum of prizes where the weight
    # reflects how much that item contributes to relieving scarcity
    heuristic = np.zeros(n_items)
    for i in range(n_constraints):
        # The contribution of item j to constraint i is prizes[i,j]
        # But if the constraint is scarce, we should prefer items with
        # high prize/weight ratio
        # Combine: prize * scarcity_factor / (1 + cost)
        scarcity_factor = 1.0 + (1.0 - scarcity[i]) * 2.0  # Boost scarce constraints
        heuristic += prizes[i] * scarcity_factor / (1.0 + cost[i])
    
    # Normalize to ensure positive values
    heuristic = heuristic - np.min(heuristic) + 1e-6
    
    # Ensure all values are positive and finite
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
