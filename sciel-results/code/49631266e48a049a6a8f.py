"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import random

# Deadline guard: the evaluator kills after 150s wall-clock.
# We track elapsed time to ensure we return quickly.
_DEADLINE_START = _time.monotonic()
_DEADLINE_LIMIT = 145.0  # safety margin of 5s

# Global state to track estimated machine loads
# Since dispatch is stateless per call in the interface, we maintain a persistent
# dictionary of estimated completion times for each machine.
# Key: machine_id, Value: estimated time when this machine becomes free.
_machine_load = {}

def dispatch(ready_operations, time):
    """
    Dispatch rule for flexible job-shop scheduling.
    
    Args:
        ready_operations: list of dicts, each with:
            - job: job identifier
            - op: operation identifier
            - machine_options: dict {machine: duration}
            - remaining_ops: number of remaining operations in this job
            - remaining_work: total remaining processing time for this job
        time: current simulation time
    
    Returns:
        index of the ready operation to start (int), 
        or (index, machine) tuple to specify machine assignment.
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # Check deadline
    if _time.monotonic() - _DEADLINE_START > _DEADLINE_LIMIT:
        # Fallback: simple SRPT (Shortest Remaining Processing Time)
        # Pick the operation with the least remaining work
        return int(min(range(n), key=lambda i: ready_operations[i]["remaining_work"]))

    if n == 1:
        op = ready_operations[0]
        # Pick the machine with minimum duration to reduce makespan
        best_machine = min(op["machine_options"].items(), key=lambda x: x[1])[0]
        # Update global load state
        _machine_load[best_machine] = time + op["machine_options"][best_machine]
        return (0, best_machine)

    best_idx = -1
    best_score = float('inf')
    best_machine = None
    best_duration = float('inf')

    for i, op in enumerate(ready_operations):
        rem_work = op["remaining_work"]
        machine_opts = op["machine_options"]
        
        if not machine_opts:
            continue

        # For each candidate operation, find the best machine assignment
        # Strategy:
        # 1. For each machine option, calculate the effective start time.
        #    Since we don't know the exact current time of other machines,
        #    we use our estimated _machine_load.
        #    If the machine is free (load <= time), start time is 'time'.
        #    Else, start time is _machine_load[m].
        # 2. Completion time = start_time + duration.
        # 3. We want to minimize the completion time (earliest finish).
        #    If there's a tie, we prefer the operation with less remaining_work
        #    (SRPT logic) to clear small jobs quickly.
        
        min_completion_time = float('inf')
        candidate_machine = None
        candidate_duration = float('inf')

        for m, dur in machine_opts.items():
            # Estimate when this machine is free
            m_free_time = _machine_load.get(m, 0.0)
            
            # If the machine is currently busy (estimated), the job starts later
            start_time = max(time, m_free_time)
            completion_time = start_time + dur
            
            # We prefer the machine that allows this operation to finish earliest
            if completion_time < min_completion_time:
                min_completion_time = completion_time
                candidate_machine = m
                candidate_duration = dur

        if candidate_machine is None:
            continue

        # Composite Score:
        # Primary: Minimize completion time of this specific operation
        # Secondary: Minimize remaining work (SRPT) to reduce average flow time
        # Tertiary: Random tie-breaker to avoid deterministic loops if needed
        
        # We want to pick the operation that results in the earliest completion,
        # but we also care about the job's overall progress.
        # A strong heuristic for FJSP is to pick the operation that has the
        # smallest "penalty" if delayed. 
        # 
        # Let's use: score = completion_time + lambda * remaining_work
        # Lower score is better.
        # We set lambda small so completion_time dominates, but remaining_work
        # breaks ties to favor jobs that are almost done (reducing makespan variance).
        
        # However, a more robust approach for makespan is often to prioritize
        # jobs with MORE remaining work (MWL) to keep critical path jobs moving.
        # But SRPT is often better for average flow time.
        # Let's try a hybrid: 
        # If we pick the one with earliest completion, we might starve long jobs.
        # If we pick MWL, we might delay short jobs that could have been done.
        #
        # Let's stick to the "Earliest Completion Time" of the operation on the
        # best machine, but break ties by preferring LOWER remaining_work (SRPT).
        # This tends to keep the system "busy" with quick wins and prevents
        # short jobs from waiting behind long ones unnecessarily.
        
        score = min_completion_time + 0.01 * rem_work

        if score < best_score:
            best_score = score
            best_idx = i
            best_machine = candidate_machine
            best_duration = candidate_duration

    if best_idx == -1:
        # Fallback if something went wrong
        return 0

    # Update the global machine load state for the selected machine
    # This is crucial for the "Dynamic Load Balancing" aspect.
    if best_machine is not None:
        _machine_load[best_machine] = time + best_duration

    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
