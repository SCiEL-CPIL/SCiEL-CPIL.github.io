"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute heuristic values for ant colony optimization.
    
    Args:
        prizes: (5, 100) array of prizes per constraint per item
        weights: (5, 100) array of weights per constraint per item
    
    Returns:
        (100,) array of heuristic values, one per item
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Total prize per item (sum across constraints)
    total_prize = np.sum(prizes, axis=0)
    
    # Total weight per item (sum across constraints)
    total_weight = np.sum(weights, axis=0)
    
    # Per-constraint weight: how much of each constraint this item consumes
    # Higher per-constraint weight means more "expensive" in that dimension
    
    # Value-to-weight ratio as base heuristic
    value_weight_ratio = total_prize / (total_weight + 1e-9)
    
    # Per-constraint efficiency: prize contribution relative to weight in that constraint
    # For each constraint c, item i has prize[i] and weight[i]. 
    # The "cost" of item i in constraint c is weight[c,i] / capacity.
    # We want items that give high prize for low cost across all constraints.
    
    # Compute per-constraint cost (fraction of capacity used)
    capacity = 25.0
    cost_per_constraint = weights / capacity  # shape (5, 100)
    
    # Per-constraint value density: prize / cost for each constraint
    # This tells us how much prize we get per unit of capacity consumed in each constraint
    value_density_per_constraint = prizes / (cost_per_constraint + 1e-9)  # shape (5, 100)
    
    # Average value density across constraints
    avg_value_density = np.mean(value_density_per_constraint, axis=0)
    
    # Minimum value density across constraints (bottleneck constraint)
    min_value_density = np.min(value_density_per_constraint, axis=0)
    
    # Competitive exclusion score: for each item, estimate how many other items
    # it "excludes" due to shared constraints. An item with high weight in a constraint
    # excludes more items from that constraint.
    # For constraint c, item i with weight w_c_i excludes roughly w_c_i / capacity * n_items
    # items from that constraint (on average).
    
    # Opportunity cost: sum over constraints of (weight in constraint / capacity) * (avg prize of items)
    # This estimates the total prize we give up by including this item
    avg_prize_per_item = np.mean(total_prize)
    opportunity_cost = np.sum(weights / capacity, axis=0) * avg_prize_per_item
    
    # Marginal utility: total_prize - opportunity_cost (net gain)
    # But this can be negative, so let's use a different formulation
    
    # Better: competitive score = prize / (1 + sum of weights across constraints / capacity)
    # Items with high prize and low total resource consumption get high scores
    total_cost_fraction = np.sum(weights / capacity, axis=0)
    competitive_score = total_prize / (1.0 + total_cost_fraction)
    
    # Normalize each component to [0, 1]
    def normalize(x):
        mn = np.min(x)
        mx = np.max(x)
        if mx > mn:
            return (x - mn) / (mx - mn)
        else:
            return np.ones_like(x) * 0.5
    
    norm_vw = normalize(value_weight_ratio)
    norm_avg_density = normalize(avg_value_density)
    norm_min_density = normalize(min_value_density)
    norm_competitive = normalize(competitive_score)
    
    # Combine components with weights emphasizing efficiency and competitive advantage
    heuristic = (0.25 * norm_vw + 
                 0.25 * norm_avg_density + 
                 0.20 * norm_min_density + 
                 0.30 * norm_competitive)
    
    # Scale to a reasonable range [1, 10]
    heuristic = 1.0 + 9.0 * heuristic
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
