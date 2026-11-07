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
    
    epsilon = 1e-9
    
    # Safe weights to avoid division by zero
    safe_weights = np.where(weights > epsilon, weights, 1.0)
    
    # 1. Calculate Efficiency Ratios R_kj = p_kj / w_kj
    ratios = prizes / safe_weights  # Shape (5, 100)
    
    # 2. Geometric Mean of Ratios for each item
    # G_j = exp( mean_k( log(R_kj) ) )
    # Use log-space to avoid underflow/overflow
    # If R_kj is 0, log is -inf. We clip to a small positive number.
    safe_ratios = np.maximum(ratios, epsilon)
    log_ratios = np.log(safe_ratios)
    mean_log_ratios = np.mean(log_ratios, axis=0)  # Shape (100,)
    G = np.exp(mean_log_ratios)  # Shape (100,)
    
    # 3. Identify the Bottleneck Weight for each item
    # The constraint that limits the item most is the one with the highest weight.
    # W_max_j = max_k w_kj
    W_max = np.max(weights, axis=0)  # Shape (100,)
    
    # 4. Combine Efficiency and Bottleneck Weight
    # We want high efficiency and low bottleneck weight.
    # Heuristic = G / (W_max + epsilon)
    # This penalizes items that are "wide" in any dimension, regardless of their efficiency in that dimension.
    # This is often better than Min-Ratio because an item with 100 efficiency but 10 weight is worse 
    # than an item with 10 efficiency but 1 weight in a tight packing scenario? 
    # Actually, ACO pheromones accumulate. The heuristic is a static guide.
    # High G means "good value per unit weight on average".
    # Low W_max means "fits easily in the tightest constraint".
    
    heuristic = G / (W_max + epsilon)
    
    # 5. Optional: Add a small component of Total Prize to break ties or boost high-value items
    # Total Prize P_sum_j = sum_k p_kj
    P_sum = np.sum(prizes, axis=0)
    
    # Blend: Geometric Efficiency / Max Weight  vs  Total Prize
    # Let's try a power blend.
    # H_new = (G / W_max)^0.7 * (P_sum)^0.3 ?
    # Or simply: H = G / (W_max + epsilon) is quite strong.
    # Let's stick to the pure "Efficiency / Bottleneck" first, as it directly addresses the MCKP "widest plank" problem.
    
    # 6. Normalization
    # Normalize to mean 1
    mean_h = np.mean(heuristic)
    if mean_h > epsilon and np.isfinite(mean_h):
        heuristic = heuristic / mean_h
    else:
        heuristic = np.ones_like(heuristic)
        
    # 7. Safety Checks
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, epsilon)
    
    if heuristic.shape != (100,):
        heuristic = heuristic.reshape(100)
        
    if time.time() > deadline:
        return heuristic
        
    return heuristic
