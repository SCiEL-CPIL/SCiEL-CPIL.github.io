"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_DEADLINE = 145.0  # safety margin before 150s kill
_start_time = _time.time()


def _elapsed():
    return _time.time() - _start_time


def dispatch(ready_operations, time):
    """
    Dispatch rule using a hybrid of Critical Ratio (CR) and NSPT signals.
    
    The Critical Ratio prioritizes operations with the least slack relative to
    their remaining processing time, capturing dynamic urgency.
    NSPT (Normalized SPT) balances machine-freeing with job-progress.
    
    Returns the index of the ready operation to start next.
    """
    if not ready_operations:
        return 0

    # Deadline guard: if we're close to the limit, just return the first op
    if _elapsed() > _DEADLINE:
        return 0

    n = len(ready_operations)
    
    # Extract metrics for each ready operation
    remaining_works = []
    min_durations = []
    max_durations = []
    
    for op in ready_operations:
        remaining_works.append(op["remaining_work"])
        
        machine_opts = op["machine_options"]
        if machine_opts:
            durations = list(machine_opts.values())
            min_dur = min(durations)
            max_dur = max(durations)
        else:
            min_dur = 0
            max_dur = 0
        min_durations.append(min_dur)
        max_durations.append(max_dur)
        
    # Normalize metrics to [0, 1] range based on current set
    max_rw = max(remaining_works) if remaining_works else 1
    max_min_dur = max(min_durations) if min_durations else 1
    
    # Handle edge cases where max is 0 to avoid division by zero
    if max_rw == 0:
        max_rw = 1
    if max_min_dur == 0:
        max_min_dur = 1
        
    best_index = 0
    best_score = -float('inf')
    
    for i, op in enumerate(ready_operations):
        rw = remaining_works[i]
        md = min_durations[i]
        M = max_durations[i]
        
        norm_rw = rw / max_rw
        norm_md = md / max_min_dur
        
        # Critical Ratio: lower is more urgent (less slack relative to work)
        # CR = (remaining_time - remaining_work) / remaining_work
        # Since we don't have a deadline, use a proxy: time relative to remaining work
        # Lower CR means more urgent
        if rw > 0:
            cr_proxy = time / rw
        else:
            cr_proxy = float('inf')
        
        # NSPT: Normalized SPT - balance between short processing and job progress
        # Higher NSPT score is better
        if md > 0:
            nspt = (1.0 - norm_md) * (rw / max_rw)
        else:
            nspt = 0.0
        
        # Combine: weight CR urgency and NSPT
        # Normalize CR proxy to [0, 1] - lower is more urgent, so invert
        max_cr = max([t / r if r > 0 else float('inf') for t, r in [(time, rw_) for rw_ in remaining_works]])
        if max_cr == 0 or max_cr == float('inf'):
            norm_cr = 1.0
        else:
            norm_cr = 1.0 - (cr_proxy / max_cr) if cr_proxy != float('inf') else 0.0
        
        # Final score: combine urgency (CR) and efficiency (NSPT)
        score = 0.5 * norm_cr + 0.5 * nspt
        
        if score > best_score:
            best_score = score
            best_index = i

    return int(best_index)
# EVOLVE-BLOCK-END
