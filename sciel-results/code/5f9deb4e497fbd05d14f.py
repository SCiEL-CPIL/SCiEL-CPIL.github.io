"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute marginal utility heuristic for each item.
    
    The heuristic estimates the opportunity cost of each item by considering
    the scarcity of each resource. For each constraint (resource), we identify
    how "scarce" that resource is by looking at the efficiency of items that
    consume it. Items that consume scarce resources heavily are penalized,
    while items that offer high prize relative to the scarcity of resources
    they consume are rewarded.
    
    Approach:
    1. For each constraint j, compute the efficiency ratio prize_i / weight_ij for all items.
    2. Estimate scarcity of resource j by the average efficiency of the top-k items
       that consume resource j (indicating how valuable the resource is).
    3. The marginal utility of item i is its prize divided by a weighted sum of
       the scarcities of the resources it consumes, weighted by how much of each
       resource it uses.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    n_constraints, n_items = weights.shape
    
    # Ensure positive weights to avoid division issues
    weights_safe = np.maximum(weights, 1e-9)
    
    # Step 1: Compute efficiency for each item-constraint pair
    # efficiency[i, j] = prizes[i] / weights[i, j]
    efficiency = prizes[:, np.newaxis] / weights_safe  # shape (n_constraints, n_items)
    
    # Step 2: For each constraint, find the top-k most efficient items
    # This indicates how "valuable" or "scarce" each resource is
    k = min(20, n_items)  # top 20 items per constraint
    
    # For each constraint, get the top-k efficiencies
    # We want to estimate scarcity: if the top items have high efficiency,
    # the resource is highly contended/scarce
    scarcity = np.zeros(n_constraints)
    for j in range(n_constraints):
        top_effs = np.sort(efficiency[j, :])[-k:]
        scarcity[j] = np.mean(top_effs)
    
    # Step 3: Compute marginal utility for each item
    # For item i, the opportunity cost is the weighted sum of scarcities
    # of the resources it consumes, where weights are the actual consumption
    # normalized by the scarcity itself (so we're measuring "how much scarce
    # resource capacity this item uses")
    
    # Normalized consumption: how much of each resource's "scarce capacity" 
    # does this item use?
    # marginal_cost[i, j] = weights[i, j] * scarcity[j] / (capacity_j)
    # But we don't have capacity directly; capacity is 25 for all constraints.
    # So marginal_cost[i, j] = weights[i, j] * scarcity[j] / 25
    
    capacity = 25.0
    marginal_cost = weights_safe * scarcity[np.newaxis, :] / capacity  # shape (n_constraints, n_items)
    
    # The total opportunity cost for item i is the sum over all constraints
    total_opportunity_cost = marginal_cost.sum(axis=0)  # shape (n_items,)
    
    # The heuristic is prize / total_opportunity_cost
    # This rewards items with high prize relative to the scarcity-weighted
    # resources they consume
    heuristic = prizes.sum(axis=0) / (total_opportunity_cost + 1e-9)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
