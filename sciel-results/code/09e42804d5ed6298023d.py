"""Job-Shop Scheduling dispatch rule: Normalized multi-criteria heuristic with SPT and MWRT."""
# EVOLVE-BLOCK-START
import time as _time

_START = _time.time()
_DEADLINE = 140.0  # Safe margin before 150s kill


def dispatch(ready_operations, time):
    """
    Dispatch rule: Normalized multi-criteria heuristic combining:
    - Shortest Processing Time (SPT) on best machine
    - Longest Remaining Work (MWRT) to avoid starving long jobs
    - Normalization to make criteria comparable
    
    Strategy:
    1. For each ready operation, compute min duration across machine options.
    2. Normalize all features to [0, 1] range for fair comparison.
    3. Composite score: primarily SPT, secondarily MWRT, with small remaining_ops penalty.
    
    The normalization ensures that the relative importance of criteria is consistent
    across different problem instances, unlike absolute weighting.
    """
    # Deadline guard
    if _time.time() - _START > _DEADLINE:
        return 0

    if not ready_operations:
        return 0

    n = len(ready_operations)
    if n == 1:
        return 0

    # Precompute features for each ready operation
    best_duration = []      # shortest duration across available machines
    remaining_work = []     # total remaining work for the job
    remaining_ops = []      # number of remaining operations

    for op in ready_operations:
        mo = op.get("machine_options", {})
        rw = op.get("remaining_work", 0)
        ro = op.get("remaining_ops", 0)
        
        if mo:
            durations = list(mo.values())
            min_d = min(durations) if durations else 1.0
        else:
            min_d = float('inf')
        
        best_duration.append(min_d)
        remaining_work.append(rw if rw is not None and rw >= 0 else 0)
        remaining_ops.append(ro if ro is not None and ro >= 0 else 0)

    # Normalize features to [0, 1] range
    def normalize(vals):
        finite_vals = [v for v in vals if v != float('inf')]
        if not finite_vals:
            return [0.0] * len(vals)
        mn = min(finite_vals)
        mx = max(finite_vals)
        rng = mx - mn
        if rng < 1e-9:
            return [0.5] * len(vals)
        result = []
        for v in vals:
            if v == float('inf'):
                result.append(1.0)  # worst case for SPT
            else:
                result.append((v - mn) / rng)
        return result

    # Normalize best_duration: lower is better for SPT, so we want to minimize
    norm_dur = normalize(best_duration)  # 0 = shortest, 1 = longest
    
    # Normalize remaining_work: higher is better (more work = higher priority)
    norm_rw = normalize(remaining_work)  # 0 = least work, 1 = most work
    
    # Normalize remaining_ops: higher is better
    norm_ro = normalize(remaining_ops)  # 0 = fewest ops, 1 = most ops
    
    # Compute composite score: lower is better
    # Primary: short processing time (weight 0.60) - strongly prefer short ops
    # Secondary: more remaining work (weight 0.25) - invert so more work = lower score
    # Tertiary: more remaining ops (weight 0.15) - invert so more ops = lower score
    
    best_idx = 0
    best_score = float('inf')
    
    for i in range(n):
        # SPT: shorter duration = better (lower score)
        s_spt = norm_dur[i]
        
        # More remaining work = better priority (lower score when work is high)
        s_rw = 1.0 - norm_rw[i]
        
        # More remaining ops = better priority (lower score when ops are high)
        s_ro = 1.0 - norm_ro[i]
        
        # Composite: lower is better
        score = 0.60 * s_spt + 0.25 * s_rw + 0.15 * s_ro
        
        if score < best_score - 1e-12:
            best_score = score
            best_idx = i
    
    return int(best_idx)
# EVOLVE-BLOCK-END
