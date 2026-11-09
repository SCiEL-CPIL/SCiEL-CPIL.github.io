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


def _soft_min(values, temperature=0.1):
    """
    Compute a soft minimum using log-sum-exp trick.
    This provides a smooth surrogate for the hard min, giving more nuanced
    information about the distribution of available durations.
    """
    if len(values) == 0:
        return 0.0
    if len(values) == 1:
        return values[0]
    
    v = np.array(values, dtype=np.float64)
    v_min = np.min(v)
    # Soft min: v_min - temperature * log(sum(exp(-(v - v_min)/temperature)))
    # As temperature -> 0, this approaches the hard min
    exp_term = np.exp(-(v - v_min) / temperature)
    soft_min_val = v_min - temperature * np.log(np.sum(exp_term))
    return soft_min_val


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    
    Strategy: Multi-criteria heuristic with soft-min surrogate for machine
    duration, combining:
      1. Most remaining work (prioritize long jobs)
      2. Soft-min completion time (smooth surrogate for best machine choice)
      3. Least remaining operations (finish jobs soon to reduce makespan)
      4. Machine flexibility bonus (more options = more scheduling flexibility)
    
    We compute a composite score for each ready operation and pick the one
    with the highest priority (lowest score).
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
    
    for i, op in enumerate(ready_operations):
        durations = op["machine_options"]
        if durations:
            dur_values = list(durations.values())
            sm = _soft_min(dur_values, temperature=0.15)
            soft_min_durations[i] = sm
            flexibility[i] = len(dur_values)
        else:
            soft_min_durations[i] = 0.0
            flexibility[i] = 0.0
        remaining_works[i] = op["remaining_work"]
        remaining_ops[i] = op["remaining_ops"]
    
    # Normalize features to [0, 1] range for robust weighting
    rw_min, rw_max = remaining_works.min(), remaining_works.max()
    sd_min, sd_max = soft_min_durations.min(), soft_min_durations.max()
    ro_min, ro_max = remaining_ops.min(), remaining_ops.max()
    fl_min, fl_max = flexibility.min(), flexibility.max()
    
    rw_range = rw_max - rw_min if rw_max > rw_min else 1.0
    sd_range = sd_max - sd_min if sd_max > sd_min else 1.0
    ro_range = ro_max - ro_min if ro_max > ro_min else 1.0
    fl_range = fl_max - fl_min if fl_max > fl_min else 1.0
    
    rw_norm = (remaining_works - rw_min) / rw_range
    sd_norm = (soft_min_durations - sd_min) / sd_range
    ro_norm = (remaining_ops - ro_min) / ro_range
    fl_norm = (flexibility - fl_min) / fl_range
    
    # Composite score: we want to minimize this
    # - Higher remaining_work -> should be prioritized (negative weight)
    # - Lower soft_min_duration -> should be prioritized (positive weight for duration)
    # - Lower remaining_ops -> should be prioritized (negative weight)
    # - Higher flexibility -> slight bonus (negative weight)
    
    # Weights: remaining work is most important, then remaining ops, then duration
    w_rw = 1.0
    w_ro = 0.35
    w_sd = 0.25
    w_fl = 0.05
    
    scores = -w_rw * rw_norm + w_sd * sd_norm - w_ro * ro_norm - w_fl * fl_norm
    
    # Pick the operation with the lowest score (highest priority)
    idx = int(np.argmin(scores))
    return idx
# EVOLVE-BLOCK-END
