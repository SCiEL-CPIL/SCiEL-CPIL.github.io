"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

_dispatch_start_time = _time.time()
_DEADLINE_MARGIN = 130.0  # safety margin before 150s kill


def _deadline_exceeded():
    return (_time.time() - _dispatch_start_time) > _DEADLINE_MARGIN


def dispatch(ready_operations, time):
    """
    Pick the best ready operation to schedule next.
    
    Strategy: Two-tier priority:
    1. Primary: Longest Remaining Processing Time (LRPT) - prioritize jobs with most remaining work
    2. Secondary: Shortest Processing Time (SPT) on the chosen machine - reduce immediate blockage
    3. Tertiary: Most remaining operations - prioritize longer job chains
    
    This leverages the well-known LRPT heuristic which is strong for job-shop scheduling,
    with SPT as a tiebreaker to minimize machine occupation time.
    """
    if not ready_operations:
        return 0
    
    if _deadline_exceeded():
        return 0
    
    n = len(ready_operations)
    
    # Precompute features for each ready operation
    remaining_work = []
    remaining_ops = []
    min_duration = []
    
    for op in ready_operations:
        rw = op.get("remaining_work", 0)
        ro = op.get("remaining_ops", 0)
        mo = op.get("machine_options", {})
        
        if mo:
            durations = list(mo.values())
            min_d = min(durations)
        else:
            min_d = 0
        
        remaining_work.append(rw)
        remaining_ops.append(ro)
        min_duration.append(min_d)
    
    # Two-tier scoring:
    # Primary score: remaining_work (higher is better)
    # Secondary score: -min_duration (lower duration is better, so negate)
    # Tertiary score: remaining_ops (higher is better)
    
    # Find max/min for normalization
    max_rw = max(remaining_work) if remaining_work else 0
    min_rw = min(remaining_work) if remaining_work else 0
    rw_range = max_rw - min_rw
    
    max_ro = max(remaining_ops) if remaining_ops else 0
    min_ro = min(remaining_ops) if remaining_ops else 0
    ro_range = max_ro - min_ro
    
    max_md = max(min_duration) if min_duration else 0
    min_md = min(min_duration) if min_duration else 0
    md_range = max_md - min_md
    
    # Compute composite score with strong LRPT priority
    # LRPT dominates: weight 0.7
    # SPT tiebreaker: weight 0.2
    # Remaining ops: weight 0.1
    
    w_rw = 0.7
    w_spt = 0.2
    w_ro = 0.1
    
    scores = []
    for i in range(n):
        # Normalize remaining work to [0, 1]
        if rw_range > 1e-9:
            s_rw = (remaining_work[i] - min_rw) / rw_range
        else:
            s_rw = 0.5
        
        # Normalize min_duration to [0, 1], then negate (shorter = better)
        if md_range > 1e-9:
            s_spt = 1.0 - (min_duration[i] - min_md) / md_range
        else:
            s_spt = 0.5
        
        # Normalize remaining ops to [0, 1]
        if ro_range > 1e-9:
            s_ro = (remaining_ops[i] - min_ro) / ro_range
        else:
            s_ro = 0.5
        
        score = w_rw * s_rw + w_spt * s_spt + w_ro * s_ro
        scores.append(score)
    
    # Select the operation with the highest score
    best_idx = 0
    best_score = scores[0]
    for i in range(1, n):
        if scores[i] > best_score + 1e-9:
            best_score = scores[i]
            best_idx = i
    
    return int(best_idx)

# EVOLVE-BLOCK-END
