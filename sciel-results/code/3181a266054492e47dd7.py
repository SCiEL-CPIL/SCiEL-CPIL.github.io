import time as _time
import math

_START_TIME = _time.time()
_DEADLINE = 140.0

def dispatch(ready_operations, time):
    n = len(ready_operations)
    if n == 0:
        return 0
    
    # Deadline guard
    if _time.time() > _DEADLINE:
        # Fallback: Max remaining work, tie-break min duration
        best_idx = 0
        best_rw = -1
        best_md = float('inf')
        for i in range(n):
            rw = ready_operations[i]["remaining_work"]
            mds = list(ready_operations[i]["machine_options"].values())
            md = min(mds) if mds else 0
            if rw > best_rw or (rw == best_rw and md < best_md):
                best_rw = rw
                best_md = md
                best_idx = i
        return best_idx

    best_idx = 0
    best_score = -float('inf')
    best_min_duration = float('inf')
    best_remaining_work = -1

    for i, op in enumerate(ready_operations):
        durations = list(op["machine_options"].values())
        min_duration = min(durations) if durations else 0
        
        remaining_work = op["remaining_work"]
        
        # Heuristic: WSPT (Weighted Shortest Processing Time)
        # Weight: Remaining work of the job (proxy for critical path length)
        # Time: Minimum possible duration of the current operation
        # This prioritizes jobs with more work left, but favors operations
        # that can be processed quickly to reduce machine blockage.
        
        # Avoid division by zero
        effective_duration = max(min_duration, 1e-6)
        
        score = remaining_work / effective_duration
        
        # Secondary consideration: If scores are very close, prefer shorter duration
        # This helps in reducing idle time on machines.
        
        if score > best_score:
            best_score = score
            best_idx = i
            best_min_duration = min_duration
            best_remaining_work = remaining_work
        elif abs(score - best_score) < 1e-9:
            # Tie-breaker: Prefer shorter processing time
            if min_duration < best_min_duration:
                best_idx = i
                best_min_duration = min_duration
                best_remaining_work = remaining_work
            # If duration is same, prefer higher remaining work (should be covered by score, but safe)
            elif min_duration == best_min_duration and remaining_work > best_remaining_work:
                best_idx = i
                best_min_duration = min_duration
                best_remaining_work = remaining_work

    return best_idx
