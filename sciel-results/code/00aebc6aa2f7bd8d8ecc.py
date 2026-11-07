import numpy as np
import time

def heuristics(prizes, weights):
    """
    Computes attractiveness scores for items in a Multiple-Constraint Knapsack Problem.
    
    Strategy: Tightness-Adjusted Geometric Mean Efficiency.
    1. Estimate the tightness of each constraint based on total weight vs capacity.
    2. Adjust the weight of each item by the tightness of the constraints it consumes.
    3. Compute the geometric mean of the prize-to-adjusted-weight ratios across constraints.
       This rewards items that are efficient in all constraints, especially those that are globally tight.
    """
    start_time = time.time()
    deadline = start_time + 290.0 # Safety margin
    
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    if prizes.shape != (5, 100) or weights.shape != (5, 100):
        # Fallback to simple ratio if shape is wrong, though contract says shape is fixed
        return np.ones(100)
        
    eps = 1e-9
    capacity = 25.0
    
    # 1. Calculate tightness for each constraint
    # Total weight available in each constraint
    total_weights = np.sum(weights, axis=1) # Shape (5,)
    tightness = total_weights / capacity   # Shape (5,)
    
    # 2. Calculate adjusted efficiency for each item and constraint
    # We want to penalize items that consume resources in tight constraints more heavily.
    # Effective weight for item j in constraint i is w_ij * tightness_i
    # To avoid broadcasting issues, reshape tightness to (5, 1)
    tightness_col = tightness[:, np.newaxis]
    
    # Denominator: weights * tightness + eps
    denom = weights * tightness_col + eps
    
    # Efficiency ratios: Shape (5, 100)
    efficiencies = prizes / denom
    
    # 3. Geometric Mean across constraints
    # log(geo_mean) = mean(log(efficiencies))
    # Clip efficiencies to positive to allow log
    efficiencies = np.maximum(efficiencies, eps)
    
    log_efficiencies = np.log(efficiencies)
    mean_log_eff = np.mean(log_efficiencies, axis=0) # Shape (100,)
    
    # Convert back to linear scale
    heuristic = np.exp(mean_log_eff)
    
    # 4. Cleanup: Ensure positive and finite
    # The prompt states: "Heuristic entries that are not positive become 1e-9"
    # Our heuristic is always positive due to exp and clipping.
    # However, if any NaN or Inf slipped in, we handle it.
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # Check deadline (unlikely to trigger for this small computation)
    if time.time() > deadline:
         return heuristic
         
    return heuristic
