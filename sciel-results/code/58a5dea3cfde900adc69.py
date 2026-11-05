"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

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
    remaining_works = np.zeros(n)
    min_durations = np.zeros(n)
    remaining_ops_counts = np.zeros(n, dtype=np.int32)
    
    for i, op in enumerate(ready_operations):
        remaining_works[i] = op["remaining_work"]
        remaining_ops_counts[i] = op["remaining_ops"]
        
        machine_opts = op["machine_options"]
        if machine_opts:
            durations = list(machine_opts.values())
            min_durations[i] = min(durations)
        else:
            min_durations[i] = 0.0

    # Compute Critical Ratio: slack / processing_time
    # Slack is estimated as: remaining_work - min_duration (approximation of
    # how much "buffer" the job has beyond this operation)
    # Actually, a better CR estimate: (remaining_work - min_duration) / min_duration
    # Lower CR = more urgent
    
    # NSPT: min_duration / (remaining_work + 1)
    # Lower NSPT = shorter relative to job's remaining work = good to do now
    
    # We want to MINIMIZE a combined score (lower = more urgent/better)
    # CR component: (remaining_work - min_duration) / (min_duration + epsilon)
    #   If remaining_work >> min_duration, CR is high (not urgent)
    #   If remaining_work ~= min_duration, CR is low (very urgent, last op)
    
    epsilon = 1e-9
    
    # CR: lower means more urgent
    cr = (remaining_works - min_durations) / (min_durations + epsilon)
    
    # NSPT: lower means shorter relative to remaining work
    nspt = min_durations / (remaining_works + 1.0)
    
    # Normalize both to [0, 1] range for blending
    cr_min, cr_max = cr.min(), cr.max()
    nspt_min, nspt_max = nspt.min(), nspt.max()
    
    if cr_max > cr_min:
        cr_norm = (cr - cr_min) / (cr_max - cr_min)
    else:
        cr_norm = np.zeros(n)
    
    if nspt_max > nspt_min:
        nspt_norm = (nspt - nspt_min) / (nspt_max - nspt_min)
    else:
        nspt_norm = np.zeros(n)
    
    # Blend: 0.5 * CR (urgency) + 0.5 * NSPT (machine-freeing balanced with job progress)
    # Lower combined score = better choice
    combined = 0.5 * cr_norm + 0.5 * nspt_norm
    
    # Tie-break: prefer operations with fewer remaining ops (sooner completion of that job)
    # This is a very light tie-break
    romin, romax = remaining_ops_counts.min(), remaining_ops_counts.max()
    if romax > romin:
        rom_norm = (remaining_ops_counts.astype(float) - romin) / (romax - romin)
    else:
        rom_norm = np.zeros(n)
    
    # Add a tiny tie-break penalty for more remaining ops
    combined += 0.01 * rom_norm
    
    best_index = int(np.argmin(combined))
    
    return int(best_index)
# EVOLVE-BLOCK-END
