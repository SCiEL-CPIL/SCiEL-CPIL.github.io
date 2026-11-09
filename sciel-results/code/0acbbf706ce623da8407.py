import time as _time
import numpy as np

# Module-level deadline guard
_START_TIME = None
_TIME_LIMIT = 145.0  # 5s safety margin before 150s kill
_DEADLINE_REACHED = False

def _check_deadline():
    global _START_TIME, _DEADLINE_REACHED
    if _START_TIME is None:
        _START_TIME = _time.monotonic()
    if _time.monotonic() - _START_TIME > _TIME_LIMIT:
        _DEADLINE_REACHED = True
    return _DEADLINE_REACHED

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    
    # Check deadline
    if _check_deadline():
        return 0
    
    n = len(ready_operations)
    
    # Extract features
    durations = np.zeros(n, dtype=np.float64)
    min_durations = np.zeros(n, dtype=np.float64)
    remaining_works = np.zeros(n, dtype=np.float64)
    remaining_ops = np.zeros(n, dtype=np.float64)
    job_ids = []
    
    for i, op in enumerate(ready_operations):
        machine_opts = op.get("machine_options", {})
        if machine_opts:
            vals = list(machine_opts.values())
            durations[i] = max(vals)
            min_durations[i] = min(vals)
        else:
            durations[i] = 0
            min_durations[i] = 0
        
        remaining_works[i] = op.get("remaining_work", 0)
        remaining_ops[i] = op.get("remaining_ops", 0)
        job_ids.append(op.get("job", i))
    
    # Normalize features
    dur_max = durations.max()
    if dur_max > 0:
        dur_norm = durations / dur_max
    else:
        dur_norm = np.zeros(n)
    
    min_dur_max = min_durations.max()
    if min_dur_max > 0:
        min_dur_norm = min_durations / min_dur_max
    else:
        min_dur_norm = np.zeros(n)
    
    rw_max = remaining_works.max()
    if rw_max > 0:
        rw_norm = remaining_works / rw_max
    else:
        rw_norm = np.zeros(n)
    
    ro_max = remaining_ops.max()
    if ro_max > 0:
        ro_norm = remaining_ops / ro_max
    else:
        ro_norm = np.zeros(n)
    
    # Composite score:
    # - Primary: Longest Processing Time (LPT) - prioritize critical long operations
    # - Secondary: Remaining work of the job - keep busy jobs progressing
    # - Tertiary: Fewer remaining ops - favor jobs close to completion (reduces tail)
    # - Quaternary: Shorter minimum duration - can be slotted in more easily
    #
    # Weighting based on open-shop scheduling heuristics:
    # LPT is well-known for makespan minimization.
    # Remaining work helps avoid job starvation.
    # Fewer remaining ops helps complete jobs early.
    # Minimum duration as a small tie-breaker.
    
    w_dur = 0.50
    w_rw = 0.30
    w_ro = 0.15
    w_min_dur = 0.05
    
    scores = w_dur * dur_norm + w_rw * rw_norm + w_ro * (1.0 - ro_norm) + w_min_dur * (1.0 - min_dur_norm)
    
    # Find the index with the maximum score
    best_idx = int(np.argmax(scores))
    
    return best_idx
