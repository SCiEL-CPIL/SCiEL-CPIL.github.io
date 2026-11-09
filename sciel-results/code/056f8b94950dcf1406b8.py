# EVOLVE-BLOCK-START
import numpy as np

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    # Find indices of items that fit
    feasible_indices = [i for i in range(n) if remaining_weights[i] <= remaining_capacity]
    
    if not feasible_indices:
        return -1

    best_index = -1
    best_score = -1.0
    best_density = -1.0
    best_value = -1.0
    best_weight = float('inf')

    # Helper to compute LP relaxation value for a set of items
    def compute_lp_value(cap, weights, values):
        if cap <= 0 or len(weights) == 0:
            return 0.0
        
        items = []
        for w, v in zip(weights, values):
            if w > 0:
                items.append((v / w, v, w))
            else:
                items.append((float('inf'), v, 0))
        
        if not items:
            return 0.0
            
        items.sort(key=lambda x: x[0], reverse=True)
        
        total_val = 0.0
        rem_cap = cap
        
        for density, v, w in items:
            if w == 0:
                total_val += v
            elif rem_cap >= w:
                total_val += v
                rem_cap -= w
            else:
                total_val += v * (rem_cap / w)
                rem_cap = 0
                break
                
        return total_val

    for i in feasible_indices:
        new_cap = remaining_capacity - remaining_weights[i]
        
        rem_weights = []
        rem_values = []
        for j in range(n):
            if j != i:
                rem_weights.append(remaining_weights[j])
                rem_values.append(remaining_values[j])
        
        lp_val = compute_lp_value(new_cap, rem_weights, rem_values)
        score = remaining_values[i] + lp_val
        
        # Calculate density for tie-breaking
        w_i = remaining_weights[i]
        v_i = remaining_values[i]
        d_i = v_i / w_i if w_i > 0 else float('inf')
        
        # Check for improvement
        # Use a small epsilon for floating point comparison
        epsilon = 1e-9
        if score > best_score + epsilon:
            best_score = score
            best_index = i
            best_density = d_i
            best_value = v_i
            best_weight = w_i
        elif abs(score - best_score) <= epsilon:
            # Tie-breaking logic
            # 1. Higher density is preferred (more efficient use of remaining capacity)
            if d_i > best_density + 1e-9:
                best_index = i
                best_density = d_i
                best_value = v_i
                best_weight = w_i
            elif abs(d_i - best_density) <= 1e-9:
                # 2. Higher value is preferred
                if v_i > best_value + 1e-9:
                    best_index = i
                    best_density = d_i
                    best_value = v_i
                    best_weight = w_i
                elif abs(v_i - best_value) <= 1e-9:
                    # 3. Lower weight is preferred (preserves capacity)
                    if w_i < best_weight - 1e-9:
                        best_index = i
                        best_density = d_i
                        best_value = v_i
                        best_weight = w_i

    return best_index
# EVOLVE-BLOCK-END
