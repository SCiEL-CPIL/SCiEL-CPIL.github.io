"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

# Stateful tracking for the online CPM heuristic
# We track the earliest time each machine will be free (assuming we can schedule immediately)
# and the last known remaining work for each job to estimate criticality.
_machine_free_times = {}
_job_remaining_work_cache = {}
_start_time = None
_dispatch_count = 0

def dispatch(ready_operations, time):
    global _start_time, _dispatch_count
    
    if _start_time is None:
        _start_time = _time.time()
        
    # Deadline guard: If we are running too long, fall back to a very fast heuristic
    # to ensure we finish within the 120s limit.
    if _time.time() - _start_time > 105:
        # Fallback: SPT on the machine that is free earliest
        best_idx = 0
        best_key = (float('inf'), float('inf'))
        for i, op in enumerate(ready_operations):
            options = op["machine_options"]
            if not options:
                continue
            min_completion = float('inf')
            for m, dur in options.items():
                free_t = _machine_free_times.get(m, 0)
                completion = max(time, free_t) + dur
                if completion < min_completion:
                    min_completion = completion
            if (min_completion, 0) < best_key:
                best_key = (min_completion, 0)
                best_idx = i
        return int(best_idx)

    n_ops = len(ready_operations)
    if n_ops == 0:
        return 0

    _dispatch_count += 1
    
    # Strategy: Online Critical Path Method (CPM)
    # 1. Estimate the "criticality" of each job based on its remaining work.
    #    Jobs with more remaining work are more likely to be on the critical path.
    # 2. Estimate the "urgency" of each operation based on machine availability.
    #    Operations on machines that will be free soon are good candidates.
    # 3. Combine these to pick the operation that balances job criticality and machine efficiency.
    
    # Update job remaining work cache if not present or if we suspect it changed (it only decreases)
    # Since we don't get the full state, we rely on the 'remaining_work' field in the op dict.
    # However, 'remaining_work' in the op dict is for that specific operation's job?
    # The prompt says: "each ready operation is a dict with ... remaining_work".
    # This usually means the total remaining work for the *job* containing this operation.
    
    best_score = -float('inf')
    best_idx = 0
    best_machine = None

    for i, op in enumerate(ready_operations):
        job_id = op["job"]
        options = op["machine_options"]
        if not options:
            continue
            
        # Get remaining work for this job. 
        # If the dict 'remaining_work' is per-job, all ops for the same job should have the same value.
        # We take the max just in case, or assume it's consistent.
        job_rem_work = op["remaining_work"]
        
        # Find the machine that allows this operation to finish earliest.
        # This is a proxy for "least loaded" or "best available".
        min_completion = float('inf')
        best_m_for_op = None
        
        for m, dur in options.items():
            # Estimate when this machine will be free.
            # _machine_free_times[m] is the time we *think* the machine is free.
            # In a real simulation, if we dispatch this op, the machine becomes busy until max(time, free_time) + dur.
            # We use the current 'time' and our tracked free times.
            
            # Note: Our tracking of _machine_free_times is an approximation.
            # We update it when we dispatch. If the simulator assigns the machine to another job 
            # that we didn't predict, our tracking drifts. However, since we are the dispatcher,
            # we assume we control the assignment if we return (index, machine).
            # If we return just index, the simulator's assignment might differ.
            # The interface allows (index, machine). Let's use it to maintain state consistency.
            
            free_t = _machine_free_times.get(m, 0)
            start_t = max(time, free_t)
            completion_t = start_t + dur
            
            if completion_t < min_completion:
                min_completion = completion_t
                best_m_for_op = m

        if best_m_for_op is None:
            continue

        # Heuristic Score:
        # We want to prioritize jobs with high remaining work (critical path) 
        # and operations that can be started soon (low completion time).
        
        # Normalization: 
        # remaining_work can be large. completion_time is relative to current time.
        # Let's use a weighted sum.
        # High remaining work = Good.
        # Low completion time = Good.
        
        # A common CPM-like dispatch rule is:
        # Priority = Remaining Work of Job / (1 + Completion Time) ? No.
        
        # Let's try: Score = job_rem_work - alpha * min_completion
        # This favors jobs with lots of work left, but penalizes long delays.
        
        # Alternatively, "Longest Processing Time" (LPT) on the critical path is often good.
        # But SPT is good for clearing.
        
        # Let's stick to a robust CPM approximation:
        # 1. Identify the job with the most remaining work (Critical Job).
        # 2. Among operations for that job (or top 2), pick the one that finishes earliest.
        
        # To make it a single pass:
        # Score = job_rem_work * 1.0 - 0.1 * min_completion
        
        # Actually, a better CPM proxy is:
        # If a job has high remaining work, we want to keep its operations moving.
        # If a machine is free soon, we want to use it.
        
        score = job_rem_work - 0.5 * min_completion
        
        if score > best_score:
            best_score = score
            best_idx = i
            best_machine = best_m_for_op

    # Update state for the selected operation
    if best_machine is not None:
        # We assume the operation starts at max(time, _machine_free_times[best_machine])
        # and finishes at start + duration.
        # We need the duration on the selected machine.
        sel_op = ready_operations[best_idx]
        dur = sel_op["machine_options"][best_machine]
        start_t = max(time, _machine_free_times.get(best_machine, 0))
        finish_t = start_t + dur
        _machine_free_times[best_machine] = finish_t
        
        # Update job remaining work cache? 
        # The 'remaining_work' in the next call will be updated by the simulator.
        # We don't need to cache it unless we want to smooth out noise, but the field is provided.

    # Return (index, machine) to ensure our state tracking matches the simulator's action.
    # If the simulator ignores the machine choice, our state tracking will drift, 
    # but the heuristic (job_rem_work) remains valid.
    # The prompt says "optionally (index, machine)". Returning the tuple is safer for stateful logic.
    
    return (int(best_idx), best_machine)

# EVOLVE-BLOCK-END
