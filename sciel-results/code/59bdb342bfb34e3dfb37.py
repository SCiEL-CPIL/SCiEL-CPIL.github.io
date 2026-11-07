import numpy as np
import time

_DEADLINE = 330.0
_START_TIME = time.time()

def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multiple-constraint knapsack problem.
    
    Strategy:
    Uses a Scarcity-Weighted Geometric Mean of per-constraint prize-to-weight ratios.
    1. The geometric mean ensures that an item is only highly rated if it is efficient 
       across all constraints, avoiding the 'bottleneck' problem where an item is 
       efficient in most constraints but terrible in one.
    2. The weights for the geometric mean are determined by the total weight of all 
       items in each constraint. Constraints that are 'tighter' (higher total demand 
       relative to capacity) are given more weight in the heuristic, penalizing items 
       that are inefficient in those specific tight constraints more heavily.
    
    Args:
        prizes: numpy array of shape (5, 100) with prizes for each constraint/item.
        weights: numpy array of shape (5, 100) with weights for each constraint/item.
        
    Returns:
        numpy array of shape (100,) with heuristic scores (larger is more attractive).
    """
    # Check deadline
    if time.time() - _START_TIME > _DEADLINE - 5.0:
        # Fallback to simple geometric mean if running out of time
        p = np.clip(prizes, 1e-9, None)
        w = np.clip(weights, 1e-9, None)
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
    
    eps = 1e-9
    
    # 1. Calculate per-constraint ratios: Prize / Weight
    # Clip weights to avoid division by zero or log(0)
    safe_weights = np.clip(weights, eps, None)
    safe_prizes = np.clip(prizes, eps, None)
    
    ratios = safe_prizes / safe_weights  # Shape (5, 100)
    
    # 2. Calculate tightness weights for each constraint
    # Tightness = total_weight / capacity, raised to a power to amplify binding constraints
    capacity = 25.0
    total_weights_per_constraint = np.sum(weights, axis=1)  # Shape (5,)
    tightness = total_weights_per_constraint / capacity  # Shape (5,)
    
    # Amplify tight constraints with a power > 1
    power = 2.0
    scarcity_weights = np.power(np.clip(tightness, eps, None), power)
    sum_scarcity = np.sum(scarcity_weights)
    
    if sum_scarcity < eps:
        # Fallback if all weights are zero (unlikely given uniform [0,1])
        scarcity_weights = np.ones(5)
        sum_scarcity = 5.0
    
    # 3. Compute Weighted Geometric Mean
    # log(geo_mean) = sum( w_c * log(ratio_c) ) / sum(w_c)
    log_ratios = np.log(ratios)  # Shape (5, 100)
    
    # Weighted sum of log ratios
    # scarcity_weights shape (5,), log_ratios shape (5, 100)
    # Use broadcasting: scarcity_weights[:, np.newaxis] * log_ratios
    weighted_log_sum = np.sum((scarcity_weights[:, np.newaxis] / sum_scarcity) * log_ratios, axis=0)
    
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
