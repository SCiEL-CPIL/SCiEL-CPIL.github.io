"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

# Global state to track machine availability
# Key: Machine ID (int), Value: Time when the machine becomes free (float)
_machine_free_time = {}

def dispatch(ready_operations, time):
    """
    Dispatch rule: Look-ahead bottleneck heuristic.
    1. Update global machine availability state based on the current simulation time.
    2. For each ready operation, estimate the completion time if assigned to its best available machine.
    3. Select the operation that minimizes this estimated completion time.
    4. Return the index of the selected operation.
    """
    if not ready_operations:
        return 0

    # Deadline guard: If we are extremely close to the limit, return a safe default
    # (Though this function is O(N*M) which is tiny, we include it per directive)
    start_time = time.time()
    deadline = 145.0 # 5s margin from 150s limit

    # 1. Update machine availability
    # The 'time' parameter is the current simulation time.
    # We assume that if a machine was scheduled to be free at T, and current time >= T, it is free now.
    # We don't explicitly track "busy until" for *future* assignments in this simple model, 
    # but we do track when they become free based on previous dispatches.
    
    # Actually, to be more robust, we should track the *next free time* for each machine.
    # When we dispatch an operation to a machine, we update its free time.
    # Here, we just ensure that if current time has passed the recorded free time, we treat it as available now (0 load relative to now).
    
    for m in list(_machine_free_time.keys()):
        if _machine_free_time[m] < time:
            _machine_free_time[m] = time

    best_index = -1
    best_score = float('inf')
    best_machine_for_best_index = None

    # 2. Evaluate each ready operation
    for i, op in enumerate(ready_operations):
        if time.time() > deadline:
            # Fallback: return first valid index if we hit deadline
            return i if best_index == -1 else best_index
            
        machine_options = op["machine_options"]
        if not machine_options:
            continue

        # Estimate the best completion time for this operation
        # We want to find the machine that allows this op to finish earliest.
        # Finish Time = max(current_time, machine_free_time[m]) + duration
        
        min_finish_time = float('inf')
        chosen_machine = None
        
        for m, duration in machine_options.items():
            # Get the time when this machine is next available
            # If machine is not in _machine_free_time, it's free now (time)
            m_free = _machine_free_time.get(m, time)
            
            # Start time for this op on machine m
            start_time_m = max(time, m_free)
            finish_time_m = start_time_m + duration
            
            if finish_time_m < min_finish_time:
                min_finish_time = finish_time_m
                chosen_machine = m

        # Score: We want to minimize the makespan impact.
        # A simple proxy is the finish time of this operation.
        # However, we also need to consider the job's remaining work to avoid starving critical jobs.
        # Let's combine: Score = min_finish_time * (1 + remaining_work_factor)
        # Or simpler: Just pick the op that finishes earliest on the least congested machine.
        # Let's try a weighted sum:
        # Score = min_finish_time + alpha * remaining_work
        
        # To balance "finishing soon" vs "reducing total work", we can use:
        # If remaining_work is high, we might want to prioritize it even if it finishes slightly later?
        # Actually, for makespan, starting the longest bottleneck early is good.
        # But "min_finish_time" already accounts for machine load.
        
        # Let's use a composite score:
        # Primary: Minimize the estimated completion time of this specific operation (relieves immediate bottleneck).
        # Secondary: Tie-breaker by remaining_work (higher is better, so we subtract it or use it in a way that favors it).
        
        # Let's stick to the "bottleneck" idea: The operation that can be completed soonest on the current least-loaded machine is the best candidate to unblock the system.
        
        # However, if we always pick the one that finishes soonest, we might ignore a job with huge remaining work that *could* be started now but takes long.
        # A better heuristic for FJSP makespan is often "Critical Path" or "Longest Remaining Processing Time" (LRPT) for the job.
        
        # Let's refine the score:
        # Score = min_finish_time - beta * remaining_work
        # We want to minimize score. So high remaining_work lowers the score (good).
        # But min_finish_time is in time units, remaining_work is in time units (sum of durations).
        # They are comparable.
        
        # Let's try: Score = min_finish_time * 0.5 + remaining_work * 0.5
        # This balances immediate completion with total job load.
        
        remaining_work = op.get("remaining_work", 0)
        score = min_finish_time * 0.5 + remaining_work * 0.5
        
        if score < best_score:
            best_score = score
            best_index = i
            best_machine_for_best_index = chosen_machine

    # 3. Update global state with the chosen machine's new free time
    if best_index != -1 and best_machine_for_best_index is not None:
        chosen_op = ready_operations[best_index]
        chosen_duration = chosen_op["machine_options"][best_machine_for_best_index]
        m_free = _machine_free_time.get(best_machine_for_best_index, time)
        new_free_time = max(time, m_free) + chosen_duration
        _machine_free_time[best_machine_for_best_index] = new_free_time

    if best_index == -1:
        return 0

    return int(best_index)

# EVOLVE-BLOCK-END
