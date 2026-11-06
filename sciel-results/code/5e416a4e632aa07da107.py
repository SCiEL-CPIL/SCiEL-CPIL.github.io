"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.monotonic()
_TOTAL_BUDGET = 110.0


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """Select next item using a blend of value density and absolute value,
    with a small lookahead to avoid items that block better combinations."""
    n = len(remaining_weights)
    if n == 0:
        return -1
    
    # Check time budget
    if time.monotonic() - _START > _TOTAL_BUDGET:
        return -1
    
    # Only consider items that fit
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    feasible_weights = remaining_weights[feasible_mask]
    feasible_values = remaining_values[feasible_mask]
    
    # Compute value density
    density = feasible_values / feasible_weights
    
    # Blend: 70% density + 30% normalized value
    max_val = np.max(feasible_values)
    if max_val > 0:
        norm_val = feasible_values / max_val
    else:
        norm_val = np.zeros_like(feasible_values)
    
    score = 0.7 * density + 0.3 * norm_val
    
    # Simple lookahead: for each candidate, estimate how much value we can still get
    # after picking it (greedy fill with remaining items)
    # Only do this if we have time and the instance is small enough
    if n <= 50 and time.monotonic() - _START < _TOTAL_BUDGET * 0.8:
        best_score = -1
        best_idx = -1
        for i in feasible_indices:
            w_i = feasible_weights[i]
            v_i = feasible_values[i]
            rem_cap = remaining_capacity - w_i
            if rem_cap <= 0:
                total_val = v_i
            else:
                # Greedy fill with remaining items (excluding i)
                other_mask = np.ones(n, dtype=bool)
                other_mask[i] = False
                other_mask &= feasible_mask
                other_w = remaining_weights[other_mask]
                other_v = remaining_values[other_mask]
                other_feasible = other_w <= rem_cap
                if np.any(other_feasible):
                    ow = other_w[other_feasible]
                    ov = other_v[other_feasible]
                    od = ov / ow
                    order = np.argsort(-od)
                    total_val = v_i
                    rc = rem_cap
                    for j in order:
                        if ow[j] <= rc:
                            total_val += ov[j]
                            rc -= ow[j]
                        else:
                            break
                else:
                    total_val = v_i
            
            # Combine with density score for tie-breaking
            combined = total_val + 0.1 * score[i]
            if combined > best_score:
                best_score = combined
                best_idx = i
        
        if best_idx >= 0:
            return int(best_idx)
    
    # Fallback: best blended score
    best_local = int(np.argmax(score))
    return int(feasible_indices[best_local])
# EVOLVE-BLOCK-END
