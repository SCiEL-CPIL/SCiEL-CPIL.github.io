"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math
from collections import defaultdict

# Global deadline guard
_START_TIME = _time.time()
_DEADLINE = _START_TIME + 110.0  # 10s safety margin before 120s kill


def dispatch(ready_operations, time):
    """
    Dispatch rule for open-shop scheduling.
    
    Strategy: Multi-factor scoring that considers:
    1. Remaining work (critical path length)
    2. Processing time of the operation (urgency)
    3. Machine flexibility (how many machines can process this op)
    4. Job flexibility (how many ops remain for this job)
    
    We score each ready operation and pick the one with the highest score.
    """
    if not ready_operations:
        return 0
    
    # Deadline guard - if we're running out of time, use a simple heuristic
    if _time.time() > _DEADLINE - 5.0:
        return int(max(range(len(ready_operations)), 
                      key=lambda i: ready_operations[i]["remaining_work"]))
    
    n = len(ready_operations)
    
    # Pre-compute some statistics
    max_remaining_work = max(op["remaining_work"] for op in ready_operations)
    if max_remaining_work == 0:
        max_remaining_work = 1
    
    scores = []
    
    for i, op in enumerate(ready_operations):
        job = op["job"]
        machine_options = op["machine_options"]
        remaining_ops = op["remaining_ops"]
        remaining_work = op["remaining_work"]
        
        # 1. Remaining work factor (normalized)
        work_score = remaining_work / max_remaining_work
        
        # 2. Processing time factor - shorter ops might be better to start
        #    to avoid blocking machines, but longer ops on critical path matter
        durations = list(machine_options.values())
        if not durations:
            proc_time_score = 0.0
        else:
            # Use the minimum duration as a proxy for how fast this op can complete
            min_dur = min(durations)
            max_dur = max(durations)
            # Higher score for operations that are "urgent" (long remaining work relative to duration)
            proc_time_score = work_score  # Keep it simple, weighted by work
            
        # 3. Machine flexibility: fewer options = more constrained = higher priority
        num_machine_options = len(machine_options)
        # Normalize: 1 option = highest priority, many options = lower
        # Use inverse: 1/num_options, but cap it
        if num_machine_options > 0:
            flexibility_score = 1.0 / num_machine_options
        else:
            flexibility_score = 0.0
        
        # 4. Job flexibility: fewer remaining ops = job is closer to completion
        #    Finishing jobs early can reduce makespan
        if remaining_ops > 0:
            job_flex_score = 1.0 / remaining_ops
        else:
            job_flex_score = 1.0  # No more ops, this is the last one
        
        # 5. Machine load consideration: prefer operations on machines with less load
        #    We don't have direct machine load info, but we can estimate from
        #    the number of ready ops that can use each machine
        machine_load_scores = {}
        for i2, op2 in enumerate(ready_operations):
            for m in op2["machine_options"]:
                if m not in machine_load_scores:
                    machine_load_scores[m] = 0
                machine_load_scores[m] += 1
        
        # Score for this op based on machine loads
        if machine_options:
            min_machine_load = min(machine_load_scores.get(m, 0) for m in machine_options)
            max_machine_load = max(machine_load_scores.get(m, 0) for m in machine_options)
            # Prefer ops on less loaded machines
            if max_machine_load > 0:
                load_score = 1.0 - (min_machine_load / max_machine_load)
            else:
                load_score = 0.5
        else:
            load_score = 0.5
        
        # Combine scores with weights
        # Work is most important, then flexibility, then load
        score = (
            0.5 * work_score +
            0.2 * flexibility_score +
            0.15 * job_flex_score +
            0.15 * load_score
        )
        
        scores.append(score)
    
    # Pick the operation with the highest score
    best_idx = int(max(range(n), key=lambda i: scores[i]))
    
    return best_idx
# EVOLVE-BLOCK-END
