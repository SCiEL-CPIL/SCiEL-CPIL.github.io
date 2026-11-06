"""Open-shop dispatch rule: Combined MWR job selection with Load-Balancing machine assignment."""
# EVOLVE-BLOCK-START
import time
import random

# Global time budget tracking
_START_TIME = time.time()
# Evaluator kills at 150s wall-clock. We reserve safety margin.
_DEADLINE = _START_TIME + 145.0

# State tracking for load balancing
_machine_total_load = {}

def _time_left():
    return _DEADLINE - time.time()


def dispatch(ready_operations, time):
    """
    Dispatch rule for open-shop scheduling.
    
    Strategy:
    1. Select the job with Most Work Remaining (MWR).
    2. For that job's ready operation, select the machine that minimizes the resulting total load
       (Load-Balancing / Water-Filling proxy).
    3. Return (index, machine) to explicitly assign the machine.
    """
    if not ready_operations:
        return 0
    
    # Time guard: if we're running low on time, use a simple fallback
    if _time_left() < 0.5:
        max_remaining = -1
        best_idx = 0
        for i, op in enumerate(ready_operations):
            rw = op.get("remaining_work", 0)
            if rw > max_remaining:
                max_remaining = rw
                best_idx = i
        return best_idx

    # 1. Find the job with maximum remaining work (MWR)
    # We assume 'remaining_work' in the op dict is the total remaining work for that job.
    
    job_remaining = {}
    for op in ready_operations:
        job_id = op["job"]
        rw = op.get("remaining_work", 0)
        if job_id not in job_remaining or rw > job_remaining[job_id]:
            job_remaining[job_id] = rw
    
    if not job_remaining:
        return 0
    
    max_job_remaining = max(job_remaining.values())
    
    # Find candidates: ops belonging to jobs with max remaining work
    candidates = []
    for i, op in enumerate(ready_operations):
        job_id = op["job"]
        rw = op.get("remaining_work", 0)
        if rw == max_job_remaining:
            candidates.append((i, op))
            
    if not candidates:
        return 0

    # 2. Among candidates, pick the best (index, machine) pair
    # We want to minimize the load on the machine.
    # Score for a machine m for job j: current_load[m] + duration[j][m]
    # We want to minimize this score.
    # Tie-breaker: shorter duration (SPT) is generally good for makespan.
    
    best_idx = -1
    best_machine = None
    best_score = float('inf')
    best_duration = float('inf')
    
    for i, op in candidates:
        machine_options = op["machine_options"]
        for m, d in machine_options.items():
            current_load = _machine_total_load.get(m, 0.0)
            new_load = current_load + d
            # We minimize the new load on the machine
            # If tied, prefer shorter duration (SPT)
            if new_load < best_score or (new_load == best_score and d < best_duration):
                best_score = new_load
                best_duration = d
                best_idx = i
                best_machine = m
                
    if best_idx == -1:
        # Fallback
        return 0

    # 3. Update global state
    _machine_total_load[best_machine] = best_score
    
    # Return tuple to specify machine assignment
    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
