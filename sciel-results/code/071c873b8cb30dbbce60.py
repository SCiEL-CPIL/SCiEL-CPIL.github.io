"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores using a competitive exclusion / opportunity cost model.
    
    For each item, we estimate its marginal utility by considering how much it
    'competes' with other items for each constraint. The idea is that an item is
    more attractive if it has high prize but low opportunity cost (i.e., it doesn't
    severely compete with many high-prize items for the same resources).
    
    Approach:
    - For each item i and each constraint c, compute the total prize of other items
      that would violate constraint c if i were added (i.e., items j where
      weights[c, j] > capacity - weights[c, i]).
    - The opportunity cost for item i is the sum (or some aggregation) of these
      competing prizes across all constraints.
    - The heuristic is then prize[i] / (1 + opportunity_cost[i]), which rewards
      high-prize items with low competitive pressure.
    """
    n_constraints, n_items = prizes.shape
    capacity = 25.0
    
    # Total prize per item
    total_prizes = prizes.sum(axis=0)
    
    # For each constraint c, for each item i, find the total prize of items j
    # such that adding item i would make constraint c tight/violated by j.
    # Specifically, items j where weights[c, j] > capacity - weights[c, i]
    # These are the items that "compete" with i for constraint c.
    
    # We'll compute opportunity cost per item as a sum over constraints.
    # For efficiency, we can use broadcasting.
    
    # weights has shape (n_constraints, n_items)
    # For constraint c, item i competes with items j where weights[c, j] > capacity - weights[c, i]
    
    # Let's compute this per constraint
    opportunity_cost = np.zeros(n_items)
    
    for c in range(n_constraints):
        # w_c[i] is the weight of item i in constraint c
        w_c = weights[c]
        # p_c[j] is the prize of item j in constraint c (but we want total prize)
        # Actually, we want the total prize of competing items, not just constraint c prize.
        # So we use total_prizes[j] for the prize of item j.
        
        # For each item i, find items j where w_c[j] > capacity - w_c[i]
        # This is equivalent to: w_c[i] + w_c[j] > capacity
        
        # We can compute this using matrix operations
        # comp_mask[i, j] = 1 if w_c[i] + w_c[j] > capacity
        # But this is O(n_items^2) per constraint, which is 100^2 = 10000, times 5 = 50000. Fine.
        
        # Sum of total_prizes[j] for all j where w_c[i] + w_c[j] > capacity
        # Exclude j = i? The problem says "all other items", so exclude j=i.
        
        # Let's do it with broadcasting
        # sum_w = w_c[:, None] + w_c[None, :]  # shape (n_items, n_items)
        # mask = sum_w > capacity
        # mask = np.where(np.eye(n_items, dtype=bool), 0, mask)  # exclude self
        # comp_prize = mask @ total_prizes  # sum of total_prizes[j] for competing j
        
        sum_w = w_c[:, None] + w_c[None, :]
        mask = (sum_w > capacity).astype(float)
        # Exclude self
        mask[np.diag_indices_from(mask)] = 0
        comp_prize = mask @ total_prizes
        
        opportunity_cost += comp_prize
    
    # Heuristic: prize / (1 + opportunity_cost)
    # Use total_prizes as the prize measure
    heuristic = total_prizes / (1.0 + opportunity_cost)
    
    # Ensure positive values
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
