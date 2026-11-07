import numpy as np
import time

def heuristics(prizes, weights):
    start_time = time.time()
    deadline = start_time + 290  # Safety margin

    # prizes: (5, 100), weights: (5, 100)
    # 5 constraints, capacity 25 each
    
    # Ensure input is float
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Calculate efficiency ratios for each constraint and item
    # R[k, j] = prize[k, j] / weight[k, j]
    # Use epsilon to avoid division by zero
    epsilon = 1e-9
    safe_weights = np.where(weights > epsilon, weights, 1.0)
    ratios = prizes / safe_weights  # Shape (5, 100)
    
    # 1. Harmonic Mean of ratios for each item
    # H_j = n / sum(1/r_kj)
    # 1/r_kj = safe_weights[k, j] / prizes[k, j]
    # We need to handle prizes == 0. If prize is 0, ratio is 0, 1/ratio is inf.
    # If prize is 0, the item is useless. We want H_j to be 0.
    # So, if any prize is 0 (or very small), H_j should be 0.
    
    # Let's compute inverse ratios safely
    # If prizes[k,j] is 0, inverse ratio is inf.
    # Sum of inverses will be inf. H_j = 5/inf = 0.
    
    inv_ratios = safe_weights / np.where(prizes > epsilon, prizes, 1.0)
    # If prizes was 0, we set denom to 1.0, so inv_ratio is finite but large?
    # No, if prizes is 0, the item has no value. We should explicitly zero it out.
    # Let's mask items with zero total prize or zero individual prizes?
    # If an item has 0 prize in one constraint but positive in others, it's still valuable.
    # The harmonic mean of [10, 0] is 0. This is correct for "efficiency" if one dimension is infinitely inefficient.
    # However, if weight is 0 and prize is 0, ratio is undefined.
    
    # Let's refine:
    # We want H_j to be high if p/w is high for all k.
    # If w_kj is 0 and p_kj > 0, ratio is inf. 1/ratio is 0.
    # If p_kj is 0 and w_kj > 0, ratio is 0. 1/ratio is inf.
    # If both 0, it's 0/0. Let's treat as neutral (1/1=1) or bad?
    # Given uniform [0,1], zeros are rare but possible.
    
    # Safe calculation of 1/ratio:
    # r = p/w
    # 1/r = w/p
    # If p < epsilon, 1/r is very large (bad).
    # If w < epsilon and p >= epsilon, 1/r is 0 (good).
    
    denom = np.where(prizes > epsilon, prizes, 1.0)
    inv_ratios = safe_weights / denom
    
    # If original prize was 0 (and weight > 0), inv_ratio is large (weight/1.0).
    # This correctly penalizes the harmonic mean.
    # If original weight was 0 (and prize > 0), inv_ratio is 0.
    # This correctly boosts the harmonic mean.
    
    sum_inv_ratios = np.sum(inv_ratios, axis=0) # (100,)
    
    # Harmonic Mean H_j
    # If sum_inv_ratios is 0, it means all inv_ratios were 0, which implies all weights were 0 (and prizes > 0).
    # In that case, efficiency is infinite. We cap it or set to a large value.
    H = np.where(sum_inv_ratios > epsilon, 5.0 / sum_inv_ratios, 1e6)
    
    # 2. Min-Ratio (Bottleneck) M_j
    M = np.min(ratios, axis=0) # (100,)
    
    # 3. Geometric Blending
    # Based on lesson: "Combining min-ratio ... with harmonic mean ... via geometric blending (1.5/0.5 power)"
    # Interpretation: eta = H^1.5 * M^0.5 ? Or normalized exponents?
    # Let's try eta = H^0.75 * M^0.25 as a balanced blend favoring H.
    # The "1.5/0.5" might refer to weights in a convex combination of logs?
    # log(eta) = 0.75 log(H) + 0.25 log(M)
    
    # Ensure positive for log
    H_safe = np.maximum(H, epsilon)
    M_safe = np.maximum(M, epsilon)
    
    # Exponents
    alpha = 0.75
    beta = 0.25
    
    heuristic = np.power(H_safe, alpha) * np.power(M_safe, beta)
    
    # 4. Normalization
    # Normalize to mean 1 to match pheromone scale
    mean_h = np.mean(heuristic)
    if mean_h > epsilon and np.isfinite(mean_h):
        heuristic = heuristic / mean_h
    else:
        heuristic = np.ones_like(heuristic)
        
    # 5. Safety Checks
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, epsilon)
    
    if heuristic.shape != (100,):
        heuristic = heuristic.reshape(100)
        
    if time.time() > deadline:
        return heuristic
        
    return heuristic
