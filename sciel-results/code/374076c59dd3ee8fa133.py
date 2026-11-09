"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math
import random

# Deadline guard
_START_TIME = None
_DEADLINE = 140.0  # Safety margin before 150s kill

def _soft_min(values, temperature=1.0):
    """Compute soft-min using log-sum-exp trick for stability."""
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    
    # Use log-sum-exp for numerical stability
    # soft_min(x) = -log(sum(exp(-x_i / T))) * T
    # To avoid overflow, shift by min value
    min_val = min(values)
    # If values are very close to min, we can just return min
    # But for the surrogate, we want the smooth version
    
    # For small temperature, soft_min approaches true min
    # For large temperature, it approaches average
    # We use T=1.0 as a reasonable default
    
    # Compute log(sum(exp(-x_i/T)))
    # = log(sum(exp(-(x_i - min_val)/T))) + (-min_val/T)
    # Let y_i = -(x_i - min_val)/T = (min_val - x_i)/T
    # Then log_sum_exp = log(sum(exp(y_i))) - min_val/T
    # soft_min = -T * log_sum_exp = -T * (log(sum(exp(y_i))) - min_val/T)
    #          = -T * log(sum(exp(y_i))) + min_val
    
    y_values = [(min_val - v) / temperature for v in values]
    
    # Compute log(sum(exp(y_i)))
    max_y = max(y_values)
    if max_y > 500:  # Overflow protection
        return min_val
    
    sum_exp = sum(math.exp(y - max_y) for y in y_values)
    log_sum_exp = math.log(sum_exp) + max_y
    
    soft_min_val = -temperature * log_sum_exp + min_val
    
    # Clamp to be reasonable (should be >= min_val)
    if soft_min_val < min_val:
        soft_min_val = min_val
    
    return soft_min_val

def dispatch(ready_operations, time):
    global _START_TIME
    
    if _START_TIME is None:
        _START_TIME = _time.time()
    
    # Deadline check
    elapsed = _time.time() - _START_TIME
    if elapsed > _DEADLINE:
        # Fallback: return first operation or a simple heuristic
        if not ready_operations:
            return 0
        # Simple fallback: Longest Remaining Work
        return int(max(range(len(ready_operations)), key=lambda i: ready_operations[i].get("remaining_work", 0)))
    
    if not ready_operations:
        return 0
    
    n_ops = len(ready_operations)
    
    # If only one operation, return it
    if n_ops == 1:
        return 0
    
    # Features extraction using soft-min for machine duration
    features = []
    for op in ready_operations:
        rw = op.get("remaining_work", 0)
        mo = op.get("machine_options", {})
        if mo:
            durations = list(mo.values())
            # Use soft-min to capture the distribution of machine times
            # This is smoother than just min, considering all options
            sm_d = _soft_min(durations, temperature=0.5)
            min_d = min(durations)
            max_d = max(durations)
            avg_d = sum(durations) / len(durations)
            n_machines = len(mo)
        else:
            sm_d = 1.0
            min_d = 1.0
            max_d = 1.0
            avg_d = 1.0
            n_machines = 1
        
        features.append({
            'rw': rw,
            'sm_d': sm_d,
            'min_d': min_d,
            'max_d': max_d,
            'avg_d': avg_d,
            'n_machines': n_machines
        })
    
    # Normalize features
    max_rw = max(f['rw'] for f in features) or 1.0
    min_sm_d = min(f['sm_d'] for f in features)
    max_sm_d = max(f['sm_d'] for f in features) or 1.0
    min_n_machines = min(f['n_machines'] for f in features) or 1
    max_n_machines = max(f['n_machines'] for f in features) or 1
    
    # Weights: Adjusted for soft-min surrogate
    # The soft-min better captures the "effective" processing time considering all machine options
    # So we can give it more weight than the raw min
    alpha = 0.55  # Weight for remaining work (urgency)
    beta = 0.35   # Weight for efficiency (soft-min processing time)
    gamma = 0.10  # Weight for flexibility (more machine options)
    
    best_idx = 0
    best_score = -float('inf')
    
    for idx in range(n_ops):
        f = features[idx]
        
        # Normalize rw (higher is better for urgency)
        rw_score = f['rw'] / max_rw
        
        # Normalize soft-min duration (lower is better)
        if max_sm_d > min_sm_d:
            eff_score = 1.0 - (f['sm_d'] - min_sm_d) / (max_sm_d - min_sm_d)
        else:
            eff_score = 1.0
        
        # Normalize n_machines (higher is better for flexibility)
        if max_n_machines > min_n_machines:
            flex_score = (f['n_machines'] - min_n_machines) / (max_n_machines - min_n_machines)
        else:
            flex_score = 1.0
        
        score = alpha * rw_score + beta * eff_score + gamma * flex_score
        
        if score > best_score:
            best_score = score
            best_idx = idx
        elif abs(score - best_score) < 1e-9:
            # Tie-breaker: prefer shorter soft-min duration
            if f['sm_d'] < features[best_idx]['sm_d']:
                best_idx = idx
    
    return int(best_idx)

# EVOLVE-BLOCK-END
