"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Module-level state for deadline guard
_START_TIME = None
_DEADLINE = 150.0
_SAFETY_MARGIN = 10.0  # seconds


def _check_deadline():
    """Check if we're approaching the deadline."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = _time.time()
    return _time.time() - _START_TIME > (_DEADLINE - _SAFETY_MARGIN)


def _soft_min(values, temperature=0.3):
    """
    Compute a soft minimum using log-sum-exp trick.
    This provides a smooth surrogate for the hard min.
    """
    if len(values) == 0:
        return 0.0
    if len(values) == 1:
        return values[0]
    
    v = np.array(values, dtype=np.float64)
    v_min = np.min(v)
    # Soft min: v_min - temperature * log(sum(exp(-(v - v_min)/temperature)))
    exp_term = np.exp(-(v - v_min) / temperature)
    soft_min_val = v_min - temperature * np.log(np.sum(exp_term))
    return soft_min_val


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    
    Strategy: Critical-path-aware multi-criteria heuristic combining:
      1. Longest remaining critical path (sum of min durations across remaining ops)
      2. Soft-min current machine duration (smooth surrogate for best machine)
      3. Remaining work (total work left in job)
      4. Remaining operations count (prioritize near-completion jobs)
      5. Flexibility (number of machine options)
    
    The critical path term helps prevent idle time by prioritizing operations
    on the longest remaining chain, while the soft-min and other terms balance
    load and machine utilization.
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    if n == 1:
        return 0
    
    # Deadline guard: if we're running low on time, use a simple fast heuristic
    if _check_deadline():
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    # Compute features for each ready operation
    remaining_works = np.zeros(n)
    soft_min_durations = np.zeros(n)
    remaining_ops = np.zeros(n)
    flexibility = np.zeros(n)  # number of machine options
    critical_path = np.zeros(n)  # longest remaining path estimate
    
    for i, op in enumerate(ready_operations):
        durations = op["machine_options"]
        remaining_works[i] = op["remaining_work"]
        remaining_ops[i] = op["remaining_ops"]
        
        if durations:
            dur_values = list(durations.values())
            sm = _soft_min(dur_values, temperature=0.3)
            soft_min_durations[i] = sm
            flexibility[i] = len(dur_values)
            # Critical path: use remaining_work as a proxy for total remaining
            # work, but weight by the number of remaining ops to get an average
            # per-op duration estimate, then multiply by remaining ops count
            # This gives a rough estimate of the longest remaining chain
            if remaining_ops[i] > 0:
                avg_dur = remaining_works[i] / remaining_ops[i]
                critical_path[i] = avg_dur * remaining_ops[i]  # = remaining_works[i]
                # Actually, let's use a different approach: the critical path
                # is bounded by the remaining work, but we can also consider
                # the minimum duration of the current op plus the remaining work
                # of the job (minus current op's work if we can estimate it)
                # For simplicity, use remaining_work as the critical path estimate
                critical_path[i] = remaining_works[i]
            else:
                critical_path[i] = 0.0
        else:
            soft_min_durations[i] = 0.0
            flexibility[i] = 0.0
            critical_path[i] = remaining_works[i]
    
    # Normalize features to [0, 1] range
    rw_min, rw_max = remaining_works.min(), remaining_works.max()
    sd_min, sd_max = soft_min_durations.min(), soft_min_durations.max()
    ro_min, ro_max = remaining_ops.min(), remaining_ops.max()
    fl_min, fl_max = flexibility.min(), flexibility.max()
    cp_min, cp_max = critical_path.min(), critical_path.max()
    
    rw_range = rw_max - rw_min if rw_max > rw_min else 1.0
    sd_range = sd_max - sd_min if sd_max > sd_min else 1.0
    ro_range = ro_max - ro_min if ro_max > ro_min else 1.0
    fl_range = fl_max - fl_min if fl_max > fl_min else 1.0
    cp_range = cp_max - cp_min if cp_max > cp_min else 1.0
    
    rw_norm = (remaining_works - rw_min) / rw_range
    sd_norm = (soft_min_durations - sd_min) / sd_range
    ro_norm = (remaining_ops - ro_min) / ro_range
    fl_norm = (flexibility - fl_min) / fl_range
    cp_norm = (critical_path - cp_min) / cp_range
    
    # Composite score: we want to minimize this
    # - Higher critical_path -> prioritize (negative weight) - main driver
    # - Lower soft_min_duration -> prioritize (positive weight for duration)
    # - Higher remaining_work -> prioritize (negative weight)
    # - Lower remaining_ops -> prioritize (negative weight)
    # - Higher flexibility -> slight bonus (negative weight)
    
    # Weights: critical path is most important, then remaining work, then others
    w_cp = 1.0
    w_rw = 0.8
    w_ro = 0.4
    w_sd = 0.25
    w_fl = 0.05
    
    scores = -w_cp * cp_norm - w_rw * rw_norm + w_sd * sd_norm - w_ro * ro_norm - w_fl * fl_norm
    
    # Pick the operation with the lowest score (highest priority)
    idx = int(np.argmin(scores))
    return idx
# EVOLVE-BLOCK-END
