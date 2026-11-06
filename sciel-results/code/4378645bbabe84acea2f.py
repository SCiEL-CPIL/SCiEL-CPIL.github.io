"""Job-Shop dispatch via Bottleneck-Aware Critical Path with Progress Bonus."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START_TIME = _time.time()
_TIME_LIMIT = 145.0  # safety margin before 150s kill


def _min_pair(machine_options):
    """Return (min_duration, machine) from {machine: duration}."""
    if not machine_options:
        return 1.0, None
    best_m, best_d = None, float("inf")
    for m, d in machine_options.items():
        if d < best_d:
            best_d = d
            best_m = m
    return best_d, best_m


def _is_time_up():
    return (_time.time() - _START_TIME) > _TIME_LIMIT


def dispatch(ready_operations, time):
    """
    Dispatch rule using Bottleneck-Aware Critical Path scoring.
    
    1. Base priority is 'remaining_work' (Critical Path proxy).
    2. Boost operations assigned to machines that are currently under high load
       (bottlenecks) to ensure they remain busy.
    3. Small bonus for jobs with fewer remaining operations (completion progress).
    """
    if not ready_operations:
        return 0
    
    # Deadline guard
    if _time.time() - _START_TIME > _TIME_LIMIT:
        return 0
    
    n = len(ready_operations)
    
    if n == 1:
        return 0

    # 1. Calculate machine loads to identify bottlenecks
    # Load is the sum of minimum durations for all ready ops targeting that machine.
    machine_loads = {}
    for op in ready_operations:
        min_dur, min_mach = _min_pair(op["machine_options"])
        if min_mach is not None:
            machine_loads[min_mach] = machine_loads.get(min_mach, 0) + min_dur
    
    max_load = max(machine_loads.values()) if machine_loads else 1.0
    if max_load == 0:
        max_load = 1.0

    best_idx = 0
    best_score = -float("inf")
    
    for i, op in enumerate(ready_operations):
        min_dur, min_mach = _min_pair(op["machine_options"])
        
        # Base Urgency: Remaining work
        urgency = op["remaining_work"]
        
        # Bottleneck Awareness:
        # If this op's machine is heavily loaded, it's a bottleneck.
        # We prioritize it to keep the bottleneck machine busy.
        load = machine_loads.get(min_mach, 0)
        load_ratio = load / max_load if max_load > 0 else 0
        
        # Boost factor: 1.0 (neutral) to 1.2 (strong boost for bottleneck)
        bottleneck_boost = 1.0 + 0.2 * load_ratio
        
        # Progress Bonus:
        # Prefer jobs closer to completion to clear the schedule.
        rem_ops = op["remaining_ops"]
        if rem_ops > 0:
            progress_bonus = 1.0 / rem_ops
        else:
            progress_bonus = 0.0
            
        # Final Score
        # Weighting: Urgency is dominant.
        score = urgency * bottleneck_boost + 0.5 * progress_bonus
        
        # Tie-breaker: Shortest Processing Time on the chosen machine
        # Slight penalty for longer durations to reduce machine blockage if scores are close
        score -= 0.001 * min_dur

        if score > best_score:
            best_score = score
            best_idx = i

    return int(best_idx)
# EVOLVE-BLOCK-END
