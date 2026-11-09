"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math
import random

# Deadline guard
_START_TIME = None
_DEADLINE = 140.0  # Safety margin before 150s kill

def _soft_min(values, temperature=1.0):
    """Compute soft-min using log-sum-exp trick for numerical stability."""
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    
    # soft-min(x) = -T * log(sum(exp(-x_i / T)))
    # = min(x) - T * log(sum(exp(-(x_i - min(x)) / T)))
    min_val = min(values)
    # Shift for numerical stability
    shifted = [v - min_val for v in values]
    exp_sum = sum(math.exp(-s / temperature) for s in shifted)
    if exp_sum == 0:
        return min_val
    return min_val - temperature * math.log(exp_sum)

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
    
    # Critical path-based dispatch with soft-min machine selection
    # For each ready operation, compute:
    # 1. Critical path estimate: remaining_work (sum of min durations of remaining ops)
    # 2. Soft-min duration: smooth approximation of best machine duration
    # 3. Estimated completion time: time + soft_min_duration
    # 4. Flexibility: number of machine options
    #
    # Score to maximize:
    # - High critical path (urgency)
    # - Low soft-min duration (efficiency)  
    # - High flexibility (more machine options)
    
    # Compute features for all operations
    features = []
    for op in ready_operations:
        rw = op.get("remaining_work", 0)
        mo = op.get("machine_options", {})
        
        if mo:
            durations = list(mo.values())
            n_machines = len(mo)
            # Soft-min for smooth machine selection
            sm = _soft_min(durations, temperature=0.5)
        else:
            sm = 1.0
            n_machines = 1
        
        features.append({
            'rw': rw,
            'soft_min': sm,
            'n_machines': n_machines
        })
    
    # Normalize features
    max_rw = max(f['rw'] for f in features) or 1.0
    min_sm = min(f['soft_min'] for f in features)
    max_sm = max(f['soft_min'] for f in features) or 1.0
    min_nm = min(f['n_machines'] for f in features) or 1
    max_nm = max(f['n_machines'] for f in features) or 1
    
    # Weights tuned for critical path prioritization with soft-min efficiency
    alpha = 0.35  # Weight for critical path (remaining work)
    beta = 0.25   # Weight for efficiency (soft-min duration)
    gamma = 0.40  # Weight for flexibility (machine options)
    
    best_idx = 0
    best_score = -1.0
    
    for idx in range(n_ops):
        f = features[idx]
        
        # Critical path score: higher remaining work = more urgent
        cp_score = f['rw'] / max_rw
        
        # Efficiency score: lower soft-min = better (inverse)
        if max_sm > min_sm:
            eff_score = 1.0 - (f['soft_min'] - min_sm) / (max_sm - min_sm)
        else:
            eff_score = 1.0
        
        # Flexibility score: more machine options = better
        if max_nm > min_nm:
            flex_score = (f['n_machines'] - min_nm) / (max_nm - min_nm)
        else:
            flex_score = 1.0
        
        score = alpha * cp_score + beta * eff_score + gamma * flex_score
        
        if score > best_score:
            best_score = score
            best_idx = idx
        elif abs(score - best_score) < 1e-9:
            # Tie-breaker: prefer shorter soft-min duration
            if f['soft_min'] < features[best_idx]['soft_min']:
                best_idx = idx
    
    return int(best_idx)

# EVOLVE-BLOCK-END
