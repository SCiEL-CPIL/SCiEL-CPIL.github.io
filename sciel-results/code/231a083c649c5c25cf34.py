"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# We need a global state to track machine availability and job progress
# since dispatch is called in an event-driven simulation.
# However, the problem states we must keep the function names and signatures.
# We can use a global variable to store state between calls, assuming the simulation
# runs sequentially and we can infer state from inputs or maintain our own.
# BUT: The input `ready_operations` provides `remaining_ops` and `remaining_work`.
# If we just use these, we don't know which machine is free when.
# Wait, the prompt says: "optionally, which eligible machine runs it".
# If we return (index, machine), we assign the machine.
# But we don't know the machine's current availability time from the input `time`.
# Actually, `time` is the current simulation time.
# If we pick a machine, we assume it is available at `time`? No, that's not necessarily true.
# In a standard FJSP simulation, if a machine is busy, the operation waits.
# The prompt says "picks which ready operation to start next and, optionally, which eligible machine runs it".
# This implies we can choose the machine. If the machine is busy, does the operation start at `time`?
# Usually, "start next" implies the operation becomes available on the machine at `max(time, machine_available_time)`.
# But we don't have `machine_available_time` in the input.
# Let's re-read carefully: "dispatch(ready_operations, time) -> index ... (optionally (index, machine))".
# If we return a machine, we are telling the simulator to assign it.
# If the simulator knows the machine is busy, it will delay the start.
# We can't know the delay without knowing the machine's state.
# However, we can track machine states ourselves IF we assume we see every event.
# But `dispatch` is only called when an operation is ready.
# We can maintain a global dictionary of machine availability times.
# We need to initialize it. But we don't know the total number of machines?
# The problem says "5 machines". We can hardcode or infer.
# Let's try to infer machine IDs from `machine_options` keys.
# We will maintain a global `machine_available` dict.
# Initially, all machines are available at t=0.
# When we assign an operation to a machine, we update `machine_available[machine] = time + duration`.
# But wait, if we return (index, machine), the simulator starts it.
# If the machine is busy until t=10, and current time is 5, the operation starts at 10.
# We don't know the actual start time if we don't simulate it.
# However, if we assume we are the one controlling the schedule, we can calculate the *effective* start time.
# But the simulator might have other logic.
# Let's assume the standard interpretation: We propose a machine. The simulator uses it.
# To make a good choice, we need to know which machine is free earliest.
# We can track this in a global variable.

# Global state
_machine_avail = {}
_job_remaining_work = {} # Not strictly needed if we use input, but good for consistency check

def _init_state(ready_operations):
    global _machine_avail
    # Initialize machine availability if not seen before
    for op in ready_operations:
        for m in op["machine_options"].keys():
            if m not in _machine_avail:
                _machine_avail[m] = 0.0

def dispatch(ready_operations, time):
    global _machine_avail
    
    if not ready_operations:
        return 0

    _init_state(ready_operations)
    
    best_score = float('inf')
    best_index = 0
    best_machine = None
    
    # For each ready operation, evaluate the best possible machine assignment
    for i, op in enumerate(ready_operations):
        job_id = op["job"]
        remaining_work = op["remaining_work"]
        # Critical Path Approximation: 
        # Jobs with more remaining work are more critical.
        # We want to prioritize operations that reduce the makespan of the critical job.
        
        # Heuristic: 
        # 1. Identify the machine that minimizes the completion time of this operation.
        #    Completion time = max(current_time, machine_available) + duration
        # 2. Among ready operations, pick the one that minimizes the "Regret" or maximizes urgency.
        
        # Let's try a "Critical Path" based dispatch.
        # Priority = Remaining Work of Job / (1 + min_duration) ?
        # Or simply: Pick the operation that finishes earliest on its best machine, 
        # but weighted by the job's remaining work.
        
        # Actually, a common effective rule for FJSP is "Longest Remaining Processing Time" (LRPT) 
        # or "Most Work Remaining".
        # Let's combine:
        # Score = (Remaining Work of Job) * (1 / (1 + min_completion_time))
        # This prioritizes jobs with lots of work left, but favors operations that can finish quickly.
        
        min_completion_time = float('inf')
        best_m_for_op = None
        
        for m, dur in op["machine_options"].items():
            m_avail = _machine_avail.get(m, 0.0)
            start_time = max(time, m_avail)
            completion_time = start_time + dur
            
            if completion_time < min_completion_time:
                min_completion_time = completion_time
                best_m_for_op = m
        
        # If this operation is for a job with high remaining work, it's more important.
        # Let's use a weighted score.
        # High remaining work -> High priority.
        # Low completion time -> High priority (frees up machine/job sooner).
        
        # Score = remaining_work / (1 + min_completion_time)
        # We want to maximize this score.
        
        score = remaining_work / (1.0 + min_completion_time)
        
        if score > best_score:
            best_score = score
            best_index = i
            best_machine = best_m_for_op
            
    # Update machine availability for the chosen operation
    # Note: We assume the simulator will respect this assignment.
    # We update the global state to reflect that this machine is now busy.
    if best_machine is not None:
        op = ready_operations[best_index]
        dur = op["machine_options"][best_machine]
        m_avail = _machine_avail.get(best_machine, 0.0)
        start_time = max(time, m_avail)
        _machine_avail[best_machine] = start_time + dur
        
    return (best_index, best_machine)

# EVOLVE-BLOCK-END
