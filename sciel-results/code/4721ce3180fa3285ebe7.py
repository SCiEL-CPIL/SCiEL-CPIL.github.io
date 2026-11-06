"""Job-Shop Scheduling dispatch rule: combined heuristic with load-aware and work-aware scoring."""
import time as _time
import math
import numpy as np

# Deadline guard: keep a safety margin before the 150s wall-clock kill.
_DEADLINE_SECONDS = 145.0
_start_wall = _time.monotonic()


def _elapsed() -> float:
    return _time.monotonic() - _start_wall


def _is_deadline_near() -> bool:
    return _elapsed() >= _DEADLINE_SECONDS


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    
    Returns: index of the ready operation to start (int).
    Optionally could return (index, machine) but we stick with int for compatibility.
    
    Strategy: Combined heuristic that considers:
    1. Remaining work (longer remaining jobs get priority to avoid late bottlenecks)
    2. Machine load awareness: prefer operations that can go to less-loaded machines
    3. Shortest processing time as a tiebreaker to reduce idle time
    4. Remaining operations count to balance pipeline depth
    """
    if _is_deadline_near():
        # Fallback: simple rule to avoid timeout
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    scores = []
    for i, op in enumerate(ready_operations):
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        machine_options = op["machine_options"]
        
        # Min duration across available machines (shortest possible processing time)
        min_duration = min(machine_options.values())
        
        # Mean duration (average processing time on available machines)
        mean_duration = sum(machine_options.values()) / len(machine_options)
        
        # Number of machine options (flexibility)
        n_machines = len(machine_options)
        
        # Score components:
        # 1. Remaining work: higher is better (prioritize jobs with more work left)
        #    Normalize by max possible to keep scale consistent
        work_score = remaining_work
        
        # 2. Efficiency: shorter operations reduce machine contention
        #    Use inverse of mean duration as a proxy for "fast completion"
        efficiency_score = 1.0 / (mean_duration + 1e-9)
        
        # 3. Flexibility: more machine options means less risk of blocking
        flexibility_score = math.log(n_machines + 1)
        
        # 4. Pipeline depth: jobs with fewer remaining ops are closer to completion
        #    Prioritizing them can free up resources sooner
        pipeline_score = 1.0 / (remaining_ops + 1)
        
        # 5. Urgency: if this op has a short min_duration, it can finish quickly
        #    and free the machine
        urgency_score = 1.0 / (min_duration + 1e-9)
        
        # Weighted combination:
        # - Remaining work is the primary driver (avoids long tail jobs being stuck)
        # - Efficiency and urgency help reduce makespan by keeping machines busy
        # - Flexibility reduces the chance of a machine becoming a bottleneck
        # - Pipeline depth helps complete jobs early
        score = (
            1.0 * work_score +
            0.3 * efficiency_score * 100 +  # scaled to be comparable
            0.2 * flexibility_score * 10 +
            0.5 * pipeline_score * 50 +
            0.3 * urgency_score * 100
        )
        
        scores.append(score)

    # Return index of highest score
    return int(np.argmax(scores))
