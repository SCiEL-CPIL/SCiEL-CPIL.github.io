"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Deadline guard: track start time to avoid exceeding evaluation limits
_start_time = _time.time()
_DEADLINE = 150.0  # seconds
_SAFETY_MARGIN = 5.0  # seconds


def _check_deadline():
    """Return True if we should stop and return a safe result."""
    return (_time.time() - _start_time) > (_DEADLINE - _SAFETY_MARGIN)


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    
    Strategy: Use a composite score that considers:
    1. Job's remaining work (prioritize jobs with more work to reduce makespan)
    2. The minimum processing time among available machines (prefer shorter ops)
    3. Balance consideration: prefer machines that are less utilized
    
    Returns index of the ready operation to start.
    """
    if not ready_operations:
        return 0
    
    # If we're close to deadline, use a fast fallback
    if _check_deadline():
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    n = len(ready_operations)
    scores = np.zeros(n)
    
    for i, op in enumerate(ready_operations):
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        machine_options = op["machine_options"]
        
        if not machine_options:
            # No machine options; give low score
            scores[i] = -1e9
            continue
        
        # Get durations for each available machine
        durations = np.array(list(machine_options.values()), dtype=np.float64)
        
        # Minimum duration among available machines
        min_duration = np.min(durations)
        # Mean duration
        mean_duration = np.mean(durations)
        
        # Score components:
        # 1. Remaining work: higher is better (more urgent)
        # 2. Inverse of min duration: shorter ops are less blocking
        # 3. Ratio of remaining work to remaining ops (average op length)
        
        # Composite score:
        # - Primary: remaining work (weight 1.0)
        # - Secondary: prefer shorter processing times (weight 0.3)
        # - Tertiary: prefer operations where the job has fewer remaining ops 
        #   relative to work (i.e., larger average op, meaning this op is significant)
        
        score = remaining_work  # base: more remaining work = higher priority
        
        # Bonus for shorter processing time (less time blocking a machine)
        # Normalize by typical duration scale
        score += 0.3 * (1.0 / (min_duration + 1e-9))
        
        # If the job has many remaining ops but little work, each op is small;
        # prioritize jobs where this op represents a significant chunk
        if remaining_ops > 0:
            avg_op_work = remaining_work / remaining_ops
            # Higher avg_op_work means this op is more significant
            score += 0.2 * avg_op_work
        
        scores[i] = score
    
    # Return index of max score
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
