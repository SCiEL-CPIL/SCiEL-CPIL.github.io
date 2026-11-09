import numpy as np
import time

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    # Convert to numpy arrays for speed
    w = np.asarray(remaining_weights, dtype=float)
    v = np.asarray(remaining_values, dtype=float)
    c = float(remaining_capacity)
    
    n = len(w)
    if n == 0:
        return -1
        
    # Filter feasible items
    # The prompt says "remaining (feasible) arrays", but to be safe:
    mask = w <= c
    if not np.any(mask):
        return -1
        
    # If the prompt guarantees feasible arrays, mask is all True.
    # If not, we work on the subset.
    # However, the return index must be into the ORIGINAL arrays.
    
    # Let's assume the input arrays are already filtered as per "feasible arrays" description.
    # But standard practice is to handle infeasible ones if they slip in.
    
    # To be safe and efficient, let's just operate on the passed arrays.
    # If some are infeasible, they should not be picked.
    
    # Calculate scores for all items, penalize infeasible ones heavily.
    
    # Strategies:
    # 1. Density
    # 2. Value
    # 3. Inverse Weight
    # 4. Value / (Weight + Capacity/10)  -- Smoothed density
    # 5. Value * sqrt(1/Weight)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        s1 = v / w
        s2 = v
        s3 = 1.0 / w
        s4 = v / (w + c * 0.1) # Regularized density
        s5 = v * np.sqrt(1.0 / w)
        
    # Set infeasible items to -inf so they are never picked
    infeasible = w > c
    s1[infeasible] = -np.inf
    s2[infeasible] = -np.inf
    s3[infeasible] = -np.inf
    s4[infeasible] = -np.inf
    s5[infeasible] = -np.inf
    
    # If no item is feasible, return -1
    if np.all(infeasible):
        return -1
        
    # Normalize each score to [0, 1]
    # Min-Max normalization
    def normalize(s):
        s_min = np.min(s)
        s_max = np.max(s)
        if s_max == s_min:
            return np.zeros_like(s)
        return (s - s_min) / (s_max - s_min)
        
    # Handle -inf in normalization
    # We only normalize the feasible items?
    # It's easier to normalize the whole vector, but -inf breaks min/max.
    
    # Let's extract feasible indices
    feasible_idx = np.where(~infeasible)[0]
    
    if len(feasible_idx) == 0:
        return -1
        
    # Get sub-arrays for feasible items
    w_f = w[feasible_idx]
    v_f = v[feasible_idx]
    
    with np.errstate(divide='ignore', invalid='ignore'):
        f1 = v_f / w_f
        f2 = v_f
        f3 = 1.0 / w_f
        f4 = v_f / (w_f + c * 0.1)
        f5 = v_f * np.sqrt(1.0 / w_f)
        
    # Normalize each
    f1_n = (f1 - np.min(f1)) / (np.max(f1) - np.min(f1) + 1e-9)
    f2_n = (f2 - np.min(f2)) / (np.max(f2) - np.min(f2) + 1e-9)
    f3_n = (f3 - np.min(f3)) / (np.max(f3) - np.min(f3) + 1e-9)
    f4_n = (f4 - np.min(f4)) / (np.max(f4) - np.min(f4) + 1e-9)
    f5_n = (f5 - np.min(f5)) / (np.max(f5) - np.min(f5) + 1e-9)
    
    # Composite score
    # Weight them equally? Or give more weight to density?
    # Density is usually the strongest predictor.
    composite = 2.0 * f1_n + 1.0 * f2_n + 1.0 * f3_n + 1.0 * f4_n + 1.0 * f5_n
    
    # Find best item in feasible set
    best_local_idx = np.argmax(composite)
    
    # Map back to original index
    best_global_idx = feasible_idx[best_local_idx]
    
    return int(best_global_idx)
