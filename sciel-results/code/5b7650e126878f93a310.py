import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0

def _check_time():
    return time.time() < _DEADLINE

def _greedy_pack(capacity, weights, values, sort_key='density'):
    n = len(weights)
    if n == 0 or capacity <= 1e-9:
        return 0.0
    
    if sort_key == 'density':
        d = np.where(weights > 1e-9, values / weights, np.inf)
        indices = np.argsort(-d)
    elif sort_key == 'value':
        indices = np.argsort(-values)
    else:
        return 0.0

    total_val = 0.0
    curr_cap = capacity
    
    for idx in indices:
        w = weights[idx]
        if w <= curr_cap:
            total_val += values[idx]
            curr_cap -= w
            if curr_cap <= 1e-9:
                break
    return total_val

def _frac_knapsack_value(capacity, weights, values):
    """
    Returns the upper bound value using fractional knapsack.
    """
    n = len(weights)
    if n == 0 or capacity <= 1e-9:
        return 0.0
        
    d = np.where(weights > 1e-9, values / weights, np.inf)
    indices = np.argsort(-d)
    
    total_val = 0.0
    curr_cap = capacity
    
    for idx in indices:
        w = weights[idx]
        if w <= curr_cap:
            total_val += values[idx]
            curr_cap -= w
            if curr_cap <= 1e-9:
                break
        else:
            # Fractional part
            if w > 0:
                total_val += values[idx] * (curr_cap / w)
            break
            
    return total_val

def _exact_knapsack_mid(capacity, weights, values):
    n = len(weights)
    if n == 0:
        return 0.0
    if n == 1:
        return values[0] if weights[0] <= capacity else 0.0
        
    mid = n // 2
    left_w = weights[:mid]
    left_v = values[:mid]
    right_w = weights[mid:]
    right_v = values[mid:]
    
    left_subsets = []
    for mask in range(1 << mid):
        w_sum = 0.0
        v_sum = 0.0
        for i in range(mid):
            if mask & (1 << i):
                w_sum += left_w[i]
                v_sum += left_v[i]
        if w_sum <= capacity:
            left_subsets.append((w_sum, v_sum))
            
    right_subsets = []
    for mask in range(1 << (n - mid)):
        w_sum = 0.0
        v_sum = 0.0
        for i in range(n - mid):
            if mask & (1 << i):
                w_sum += right_w[i]
                v_sum += right_v[i]
        if w_sum <= capacity:
            right_subsets.append((w_sum, v_sum))
            
    if not left_subsets or not right_subsets:
        return 0.0
        
    right_subsets.sort(key=lambda x: x[0])
    rw = np.array([x[0] for x in right_subsets])
    rv = np.array([x[1] for x in right_subsets])
    
    prefix_max_v = np.maximum.accumulate(rv)
    
    best_total = 0.0
    for w_l, v_l in left_subsets:
        rem_cap = capacity - w_l
        idx = np.searchsorted(rw, rem_cap, side='right') - 1
        if idx >= 0:
            total_val = v_l + prefix_max_v[idx]
            if total_val > best_total:
                best_total = total_val
                
    return best_total

def _estimate_remainder_value(capacity, weights, values):
    n = len(weights)
    if n == 0 or capacity <= 1e-9:
        return 0.0
        
    valid_mask = weights <= capacity
    if not np.any(valid_mask):
        return 0.0
        
    w_rem = weights[valid_mask]
    v_rem = values[valid_mask]
    
    if len(w_rem) <= 20:
        return _exact_knapsack_mid(capacity, w_rem, v_rem)
    else:
        val1 = _greedy_pack(capacity, w_rem, v_rem, 'density')
        val2 = _greedy_pack(capacity, w_rem, v_rem, 'value')
        return max(val1, val2)

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1
        
    if not _check_time():
        return -1

    best_idx = -1
    best_total_val = -1.0
    
    w = np.asarray(remaining_weights, dtype=float)
    v = np.asarray(remaining_values, dtype=float)
    
    for i in range(n):
        if not _check_time():
            break
            
        w_i = w[i]
        if w_i > remaining_capacity:
            continue
            
        v_i = v[i]
        
        best_j_val = -1.0
        
        for j in range(n):
            if i == j:
                continue
            if not _check_time():
                break
                
            w_j = w[j]
            if w_j > remaining_capacity - w_i:
                continue
                
            v_j = v[j]
            
            rem_cap = remaining_capacity - w_i - w_j
            
            if rem_cap <= 1e-9:
                total_val = v_i + v_j
                if total_val > best_j_val:
                    best_j_val = total_val
                continue
                
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            mask[j] = False
            
            fit_mask = w <= rem_cap
            valid_mask = mask & fit_mask
            
            if not np.any(valid_mask):
                future_val = 0.0
            else:
                w_rem = w[valid_mask]
                v_rem = v[valid_mask]
                
                # Pruning with fractional upper bound
                ub = v_i + v_j + _frac_knapsack_value(rem_cap, w_rem, v_rem)
                if ub < best_total_val:
                    continue
                    
                future_val = _estimate_remainder_value(rem_cap, w_rem, v_rem)
            
            total_val = v_i + v_j + future_val
            
            if total_val > best_j_val:
                best_j_val = total_val
                
        if best_j_val > best_total_val:
            best_total_val = best_j_val
            best_idx = i
            
    return int(best_idx)
