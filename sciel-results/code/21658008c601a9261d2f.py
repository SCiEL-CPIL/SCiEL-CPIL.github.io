import time as _time

_START_TIME = _time.time()
_DEADLINE = _START_TIME + 140.0

# Stateful tracking to potentially enhance urgency calculation if needed,
# though the primary signal is derived from the operation dict itself.
# We keep a simple counter of calls to ensure we don't exceed time limits 
# in pathological cases, but the logic is O(N*M) which is fast.
_call_count = 0

def dispatch(ready_operations, time):
    global _call_count
    _call_count += 1
    
    if not ready_operations:
        return 0
    
    # Deadline guard
    if _time.time() > _DEADLINE:
        return 0

    best_idx = 0
    best_urgency = -1
    best_machine = None
    best_completion_time = float('inf')

    n = len(ready_operations)

    # Strategy: WSPT-like Urgency
    # 1. Calculate urgency for each job: remaining_work / min_duration
    #    High urgency means the job has a lot of work left relative to how fast 
    #    it can be processed. Delaying these jobs increases makespan significantly.
    # 2. Among operations with high urgency, pick the one that minimizes 
    #    the completion time on the chosen machine (load balancing as tie-breaker).
    
    # First pass: Determine the max urgency among ready ops to normalize or just pick max
    # We iterate through all ops.
    
    # To handle ties in urgency better, we compute a composite score:
    # Score = urgency * 1000 - normalized_completion_time
    # However, simple lexicographic is often more robust in JSS:
    # Primary: Max Urgency (remaining_work / min_duration)
    # Secondary: Min Projected Completion Time (machine_load + duration)
    
    # We need machine loads for the secondary criterion. 
    # Since we don't have a perfect global state of machine availability times 
    # (only 'time' passed in, which is current sim time, not machine free time),
    # we approximate machine load using a static map updated by our choices.
    # This is an approximation but better than nothing for the tie-breaker.
    
    # Actually, the prompt says `dispatch` is called when a machine becomes free.
    # The `time` is the current simulation time.
    # The `machine_options` gives durations.
    # If we assume the machine is free at `time`, then completion = time + duration.
    # But other machines might be busy.
    # The current program tracks `_machine_loads`. Let's stick to that for the tie-breaker
    # but make URGENCY the primary driver.
    
    # Let's re-evaluate the stateful load. 
    # In a standard event-driven sim, if `dispatch` is called, it's often because 
    # a specific machine just freed up, OR it's called to assign a job to a free machine.
    # The interface returns (index, machine). This implies we can choose ANY machine 
    # from `machine_options`.
    
    # If we choose a machine, we assume it's available now? 
    # Usually, in these dispatch rules, `ready_operations` are ops whose predecessor 
    # is done. The machine might still be busy. 
    # However, without explicit machine availability times in the input dict, 
    # we must infer or approximate.
    # The current program uses `_machine_loads` as a proxy for "how busy this machine is".
    
    # Let's use the WSPT ratio as the primary key.
    # WSPT = remaining_work / min_duration_in_options
    
    # We need to find the best (index, machine) pair.
    # Primary: Maximize WSPT of the job.
    # Secondary: Minimize the time the machine is busy (projected completion).
    
    # To make it robust, we'll calculate the urgency for each op.
    # If urgencies are close (within epsilon), we fall back to load balancing.
    
    urgency_epsilon = 0.01
    
    best_op_urgency = -1
    candidates = []
    
    for i, op in enumerate(ready_operations):
        options = op['machine_options']
        if not options:
            continue
        
        min_dur = min(options.values())
        if min_dur == 0:
            min_dur = 1e-9
            
        remaining_work = op.get('remaining_work', sum(options.values()))
        urgency = remaining_work / min_dur
        
        candidates.append((i, urgency, options))
        
        if urgency > best_op_urgency:
            best_op_urgency = urgency

    # Filter candidates that are within epsilon of the best urgency
    # This creates a set of "equally urgent" jobs.
    # Among these, we pick the one that minimizes makespan contribution.
    
    best_score = float('inf')
    best_idx_final = candidates[0][0]
    best_machine_final = None
    
    for i, urgency, options in candidates:
        if abs(urgency - best_op_urgency) > urgency_epsilon:
            continue
            
        # For this job, which machine minimizes the "cost"?
        # Cost = projected completion time.
        # We use the stateful _machine_loads as a proxy for current busy time.
        # If we assume the machine is free at `time`, completion = time + duration.
        # But if it's busy, completion = max(time, machine_free_time) + duration.
        # Since we don't have machine_free_time, we use _machine_loads as a heuristic 
        # for "how much work is already on this machine".
        
        for machine, duration in options.items():
            current_load = _machine_loads.get(machine, 0)
            # Heuristic: The "effective" start time is influenced by current load.
            # If load is high, it's likely busy.
            # Let's use: projected_completion = current_load + duration
            projected_completion = current_load + duration
            
            # We want to minimize this.
            if projected_completion < best_score:
                best_score = projected_completion
                best_idx_final = i
                best_machine_final = machine

    # Fallback if no candidates (shouldn't happen if ready_operations not empty)
    if best_machine_final is None:
        best_idx_final = 0
        best_machine_final = list(ready_operations[0]['machine_options'].keys())[0]
        best_score = ready_operations[0]['machine_options'][best_machine_final]

    # Update stateful load
    _machine_loads[best_machine_final] = _machine_loads.get(best_machine_final, 0) + \
                                         ready_operations[best_idx_final]['machine_options'][best_machine_final]

    return (best_idx_final, best_machine_final)
