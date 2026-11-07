import numpy as np
import time

_DEADLINE = 325.0
_START_TIME = time.time()

def heuristics(prizes, weights):
    """
    Computes attractiveness scores for items in a Multiple-Constraint Knapsack Problem.
    
    Strategy: Scarcity-Weighted Geometric Mean Efficiency.
    1. Estimate the tightness (scarcity) of each constraint based on total weight vs capacity.
    2. Use the tightness as weights in a weighted geometric mean of per-constraint prize-to-weight ratios.
       This ensures that efficiency in the tightest (most binding) constraints dominates the heuristic score.
    3. Normalize the result to have a mean of 1, stabilizing interaction with the initial pheromone level.
    """
    # Check deadline
    if time.time() - _START_TIME > _DEADLINE - 5.0:
        # Fallback to simple geometric mean if running out of time
        p = np.asarray(prizes, dtype=np.float64)
        w = np.asarray(weights, dtype=np.float64)
        p = np.clip(p, 1e-9, None)
        w = np.clip(w, 1e-9, None)
        ratios = p / w
        log_ratios = np.log(ratios)
        mean_log_ratio = np.mean(log_ratios, axis=0)
        heuristic = np.exp(mean_log_ratio)
        heuristic = np.maximum(heuristic, 1e-9)
        mean_h = np.mean(heuristic)
        if mean_h > 0:
            heuristic = heuristic / mean_h
        return heuristic

    # Ensure inputs are float64 arrays
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    if prizes.shape != (5, 100) or weights.shape != (5, 100):
        return np.ones(100)
        
    eps = 1e-9
    capacity = 25.0
    
    # 1. Calculate per-constraint ratios: Prize / Weight
    # Clip weights and prizes to avoid division by zero or log(0)
    safe_weights = np.clip(weights, eps, None)
    safe_prizes = np.clip(prizes, eps, None)
    
    ratios = safe_prizes / safe_weights  # Shape (5, 100)
    
    # 2. Calculate scarcity weights for each constraint
    # Total weight of all items in each constraint.
    # If a constraint has a high total weight relative to capacity, it is more likely to be the binding constraint.
    total_weights_per_constraint = np.sum(weights, axis=1)  # Shape (5,)
    
    # Scarcity: how much of the capacity is "demanded"
    # If total_weight > capacity, the constraint is binding.
    # We use the ratio total_weight / capacity as a proxy for tightness.
    # To ensure weights are positive and meaningful, we clip to at least 1.0? 
    # Actually, the inspiration used raw total weights or clipped scarcity.
    # Let's use the scarcity ratio clipped to [1, inf) to ensure tight constraints are weighted >= 1.
    # Or simply use total_weights directly as in Inspiration 1?
    # Inspiration 1 used: scarcity_weights = total_weights_per_constraint
    # Inspiration 2 used: scarcity = clip(total/capacity, 1, None)
    
    # Let's try the approach from Inspiration 1 which scored 1.0108:
    # "scarcity_weights = total_weights_per_constraint"
    # This implicitly weights constraints by their total demand.
    
    scarcity_weights = total_weights_per_constraint
    sum_scarcity = np.sum(scarcity_weights)
    
    if sum_scarcity < eps:
        # Fallback if all weights are zero (unlikely given uniform [0,1])
        scarcity_weights = np.ones(5)
        sum_scarcity = 5.0
    
    # Normalize scarcity weights so they sum to 1 for the weighted average in log-space
    normalized_scarcity = scarcity_weights / sum_scarcity  # Shape (5,)
    
    # 3. Compute Weighted Geometric Mean
    # log(geo_mean) = sum( w_c * log(ratio_c) )
    log_ratios = np.log(ratios)  # Shape (5, 100)
    
    # Weighted sum of log ratios
    # normalized_scarcity shape (5,), log_ratios shape (5, 100)
    weighted_log_sum = np.sum(normalized_scarcity[:, np.newaxis] * log_ratios, axis=0)
    
    # Convert back to linear scale
    heuristic = np.exp(weighted_log_sum)
    
    # 4. Ensure positivity and finiteness
    heuristic = np.nan_to_num(heuristic, nan=1e-9, posinf=1e9, neginf=1e-9)
    heuristic = np.maximum(heuristic, 1e-9)
    
    # 5. Normalize to have mean 1
    # This stabilizes the interaction with pheromones which start at 1.0.
    mean_h = np.mean(heuristic)
    if mean_h > 0:
        heuristic = heuristic / mean_h
        
    return heuristic
