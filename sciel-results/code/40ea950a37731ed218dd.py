import numpy as np
import time

def heuristics(prizes, weights):
    """
    Returns a heuristic array of shape (100,) for the MCKP.
    Uses a scarcity-weighted efficiency heuristic.
    """
    start_time = time.time()
    deadline = 300.0  # Seconds
    
    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Basic validity check
    if prizes.shape != weights.shape:
        # Fallback to simple heuristic if shapes don't match (shouldn't happen)
        return np.ones(prizes.shape[1])
        
    eps = 1e-9
    
    # 1. Total prize per item
    total_prize = np.sum(prizes, axis=0)
    
    # 2. Total weight per constraint
    total_weight_per_constraint = np.sum(weights, axis=1)
    
    # 3. Scarcity of each constraint
    capacity = 25.0
    scarcity = total_weight_per_constraint / capacity
    scarcity = np.maximum(scarcity, eps)
    
    # 4. Effective weight per item (weighted by scarcity)
    effective_weight = np.dot(scarcity, weights)
    
    # 5. Raw heuristic: Prize / Effective Weight
    raw_heuristic = total_prize / (effective_weight + eps)
    
    # 6. Simple heuristic: Prize / Total Weight (unweighted)
    total_weight_per_item = np.sum(weights, axis=0)
    simple_heuristic = total_prize / (total_weight_per_item + eps)
    
    # 7. Blend the two heuristics
    # The scarcity-weighted one is likely better, but blending adds robustness.
    # We'll use 80% scarcity-weighted and 20% simple.
    blend = 0.8 * raw_heuristic + 0.2 * simple_heuristic
    
    # 8. Normalize
    min_b = np.min(blend)
    max_b = np.max(blend)
    
    if max_b - min_b > eps:
        normalized = (blend - min_b) / (max_b - min_b)
    else:
        normalized = np.ones_like(blend) / blend.size
        
    # Scale to [0.1, 1.0]
    heuristic = 0.1 + 0.9 * normalized
    
    # Ensure positive and finite
    heuristic = np.nan_to_num(heuristic, nan=0.1, posinf=1.0, neginf=0.1)
    heuristic = np.maximum(heuristic, eps)
    
    # Deadline guard (unlikely to be needed)
    if time.time() - start_time > deadline - 1.0:
         return heuristic
         
    return heuristic
