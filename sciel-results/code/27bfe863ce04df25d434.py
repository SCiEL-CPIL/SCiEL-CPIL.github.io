"""Job-Shop Scheduling dispatch rule: Ratio-based heuristic (Work/Duration) with SPT fallback."""
# EVOLVE-BLOCK-START
import time as _time

_START = _time.time()
_DEADLINE = 145.0  # Safe margin before 150s kill


def dispatch(ready_operations, time):
    """
    Dispatch rule: Ratio-based heuristic prioritizing operations from jobs
    with high remaining work relative to the current operation's duration.
    
    This implements a 'Work-to-Time' ratio logic (inverse of Duration/Work).
    - High Remaining Work / Low Current Duration -> High Priority (Low Score)
    - Low Remaining Work / High Current Duration -> Low Priority (High Score)
    
    This helps maintain flow for jobs with long tails (deep jobs) while still
    favoring shorter operations to keep machines utilized, avoiding the
    starvation or blocking issues seen in pure SPT or pure MWRT.
    """
    # Deadline guard
    if _time.time() - _START > _DEADLINE:
        return 0

    if not ready_operations:
        return 0

    n = len(ready_operations)
    if n == 1:
        return 0

    best_idx = 0
    best_score = float('inf')
    best_duration = float('inf')
    best_rw = -1.0

    for i in range(n):
        op = ready_operations[i]
        mo = op.get("machine_options", {})
        rw = op.get("remaining_work", 0)
        
        if mo:
            durations = list(mo.values())
            min_d = min(durations) if durations else float('inf')
        else:
            min_d = float('inf')
            
        # Handle edge cases
        if min_d == float('inf'):
            continue # Skip invalid ops if possible, or handle gracefully
            
        # Ensure rw is non-negative
        if rw is None or rw < 0:
            rw = 0.0
            
        # Core Heuristic: Minimize (Duration / Remaining Work)
        # This is equivalent to Maximizing (Remaining Work / Duration)
        # Adding 1.0 to denominator to avoid division by zero and to
        # smoothly transition to SPT when RW is 0 (score = Dur / 1 = Dur).
        
        # If RW is 0, score = min_d. (Pure SPT for last op)
        # If RW is large, score approaches 0. (High priority for deep jobs)
        
        score = min_d / (rw + 1.0)
        
        # Tie-breaking logic:
        # 1. Lower score is better.
        # 2. If scores are very close, prefer shorter duration (SPT) for machine efficiency.
        # 3. If still tied, prefer higher remaining work (urgency).
        
        if score < best_score - 1e-9:
            best_score = score
            best_idx = i
            best_duration = min_d
            best_rw = rw
        elif abs(score - best_score) <= 1e-9:
            # Tie-breaker 1: Shorter duration
            if min_d < best_duration - 1e-9:
                best_idx = i
                best_duration = min_d
                best_rw = rw
            # Tie-breaker 2: Higher remaining work (if duration is also same)
            elif min_d == best_duration and rw > best_rw:
                best_idx = i
                best_rw = rw

    return int(best_idx)
# EVOLVE-BLOCK-END
