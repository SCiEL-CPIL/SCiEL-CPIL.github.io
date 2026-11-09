"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math

# Deadline guard
_START_TIME = None
_DEADLINE = 150.0

def _fast_dispatch(ready_operations, time):
    """Extremely fast fallback: Longest Remaining Work (LRW) / MWR."""
    if not ready_operations:
        return 0
    max_rw = -1
    best_idx = 0
    for i, op in enumerate(ready_operations):
        rw = op.get("remaining_work", 0)
        if rw > max_rw:
            max_rw = rw
            best_idx = i
    return best_idx

def _compute_critical_path(op):
    """
    Compute the longest remaining path for an operation.
    This is the sum of minimum durations across all remaining operations in the job.
    We use remaining_work as a proxy since we don't have per-op duration breakdowns.
    A better estimate: remaining_work / avg_min_duration_per_op
    But we'll use remaining_work directly as it correlates with total remaining work.
    """
    rw = op.get("remaining_work", 0)
    remaining_ops = op.get("remaining_ops", 1)
    
    # Critical path estimate: total remaining work
    # Higher critical path means this job has more work ahead, so it's more urgent
    # to start its current operation to avoid blocking the critical path.
    return rw

def _soft_min(values, beta=5.0):
    """
    Compute a smooth approximation of the minimum using log-sum-exp.
    soft_min(x) = -log(sum(exp(-beta * x_i))) / beta
    This captures the distribution of machine times rather than just the minimum.
    """
    if not values:
        return 1.0
    if len(values) == 1:
        return values[0]
    
    # Scale to avoid overflow
    vals = [v for v in values if v is not None and v > 0]
    if not vals:
        return 1.0
    
    min_val = min(vals)
    # For numerical stability, shift by min
    exp_sum = 0.0
    for v in vals:
        exp_sum += math.exp(-beta * (v - min_val))
    
    if exp_sum == 0:
        return min_val
    
    return min_val - math.log(exp_sum) / beta

def dispatch(ready_operations, time):
    global _START_TIME
    
    if _START_TIME is None:
        _START_TIME = _time.time()
    
    # Check time budget
    elapsed = _time.time() - _START_TIME
    if elapsed > _DEADLINE - 10.0:
        return _fast_dispatch(ready_operations, time)
    
    if not ready_operations:
        return 0
    
    # Critical path-based dispatch:
    # 1. For each ready op, compute its critical path (longest remaining path in job)
    # 2. For each op, find the machine that minimizes completion time
    # 3. Score combines critical path urgency and machine load balancing
    
    n_ops = len(ready_operations)
    
    # Pre-compute critical paths and min durations
    critical_paths = []
    min_durations = []
    soft_min_durations = []
    
    for op in ready_operations:
        cp = _compute_critical_path(op)
        critical_paths.append(cp)
        
        mo = op.get("machine_options", {})
        if mo:
            durations = list(mo.values())
            min_d = min(durations)
            sm_d = _soft_min(durations, beta=3.0)
        else:
            min_d = 1.0
            sm_d = 1.0
        
        min_durations.append(min_d)
        soft_min_durations.append(sm_d)
    
    # Normalize critical paths
    max_cp = max(critical_paths) if critical_paths else 1.0
    if max_cp == 0:
        max_cp = 1.0
    
    # Normalize min durations (lower is better)
    min_min_d = min(min_durations) if min_durations else 1.0
    max_min_d = max(min_durations) if min_durations else 1.0
    d_range = max_min_d - min_min_d
    if d_range == 0:
        d_range = 1.0
    
    # Weights: critical path is primary, machine efficiency is secondary
    w_cp = 0.65
    w_eff = 0.35
    
    best_score = -1
    best_idx = 0
    
    for i in range(n_ops):
        cp = critical_paths[i]
        min_d = min_durations[i]
        sm_d = soft_min_durations[i]
        
        # Normalize critical path to [0, 1]
        norm_cp = cp / max_cp
        
        # Efficiency: prefer shorter durations
        # Use soft-min to capture the distribution of machine options
        # Lower soft_min is better, so we invert it
        norm_eff = 1.0 / (sm_d + 0.5)
        
        # Normalize efficiency for balance
        # Find max norm_eff for scaling
        # Actually, let's just use the raw value and tune weights
        
        score = w_cp * norm_cp + w_eff * norm_eff
        
        if score > best_score:
            best_score = score
            best_idx = i
        elif abs(score - best_score) < 1e-9:
            # Tie breaker: prefer shorter min duration
            if min_d < min_durations[best_idx]:
                best_idx = i
    
    return best_idx

# EVOLVE-BLOCK-END
