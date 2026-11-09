import time

def dispatch(ready_operations, time):
    """
    Dispatches the next operation using a Lookahead Machine Impact strategy.
    
    Strategy:
    - For each ready operation, we evaluate the cost of starting it on its best machine.
    - Cost = Immediate Duration + Estimated Delay to Other Ready Ops on that Machine.
    - By simulating the machine being busy for the candidate's duration, we estimate how much
      other waiting operations (that could use that machine) would be delayed.
    - This favors operations that are short OR that use machines with few competing ready operations.
    - It avoids "locking up" a busy machine with a long job if a short job could have used it,
      or vice versa, by considering the global state of ready ops.
    
    Rationale:
    - Current strategies (SPT, Critical Path) are myopic. They don't account for the fact
      that choosing one operation might block several others that are ready to go.
    - This lookahead approach approximates a 1-step greedy optimization of machine utilization.
    """
    if not ready_operations:
        return -1

    # Pre-process ready operations to get their best machine and duration
    # and a list of all machines they can use.
    # We assume the "time" passed is the current simulation time, but we only care about relative durations.
    
    n_ops = len(ready_operations)
    
    # To speed up, we'll pre-compute for each operation:
    # 1. The best machine (min duration)
    # 2. The duration on that best machine
    # 3. The set of machines it can use (for counting conflicts)
    
    op_details = []
    for op in ready_operations:
        options = op['machine_options']
        if not options:
            op_details.append((float('inf'), -1, set()))
            continue
        
        best_mach = None
        best_dur = float('inf')
        for m, d in options.items():
            if d < best_dur:
                best_dur = d
                best_mach = m
        
        # Get set of machines for conflict checking
        # We only care about the best machine for the "impact" calculation to keep it O(N^2) small
        # or we can check all machines? 
        # Let's stick to the best machine for the primary impact, as that's where the job *would* run.
        # If we pick this job, it occupies best_mach.
        # Other jobs that also list best_mach as an option might be delayed if they preferred it.
        
        op_details.append((best_dur, best_mach, set(options.keys())))

    best_index = 0
    best_cost = float('inf')

    for i in range(n_ops):
        dur_i, mach_i, machines_i = op_details[i]
        
        if dur_i == float('inf'):
            continue

        # Cost component 1: The duration itself (SPT preference)
        # We scale it slightly to not dominate the impact term completely, 
        # or just use it as is. Let's use it as is first.
        cost = dur_i
        
        # Cost component 2: Impact on other ready operations
        # If we start job i on mach_i, how much does it hurt others?
        # We assume that if another job j could use mach_i, and its best machine is mach_i,
        # it will be delayed by dur_i.
        # If its best machine is NOT mach_i, but it *can* use mach_i, it might switch, 
        # incurring a penalty of (dur_on_mach_i - dur_on_best).
        
        # Simplified Impact: 
        # Sum over all j != i:
        #   If j's best machine is mach_i: Penalty = dur_i (j is blocked)
        #   Else if j can use mach_i: Penalty = max(0, dur_j_on_mach_i - dur_j_best) * 0.5 (discounted switching cost)
        #   Else: 0
        
        # To keep it fast, we pre-compute durations for all pairs? 
        # N is small (10 jobs, maybe 10-20 ready ops). N^2 is fine.
        
        impact = 0.0
        for j in range(n_ops):
            if i == j:
                continue
            
            dur_j, mach_j, machines_j = op_details[j]
            
            if dur_j == float('inf'):
                continue
            
            # Case 1: j's best machine is mach_i. It is directly blocked.
            if mach_j == mach_i:
                impact += dur_i
            # Case 2: j's best machine is NOT mach_i, but j can use mach_i.
            # If j switches to mach_i, it might be slower. 
            # We estimate the penalty as the extra time j would take on mach_i compared to its best.
            elif mach_i in machines_j:
                # We need the duration of j on mach_i.
                # We didn't store the full dict, so we have to look it up from ready_operations[j]
                # This is a bit slow inside the loop, but N is small.
                dur_j_on_mach_i = ready_operations[j]['machine_options'].get(mach_i, float('inf'))
                if dur_j_on_mach_i < float('inf'):
                    # Penalty is the extra time incurred if j switches
                    # We discount this by 0.5 because j might not switch, or the scheduler might handle it.
                    # Also, if dur_j_on_mach_i is much larger, the penalty is high.
                    extra = dur_j_on_mach_i - dur_j
                    if extra > 0:
                        impact += extra * 0.5
                    # If extra <= 0, j would actually prefer mach_i, so blocking it is less bad? 
                    # Actually, if j prefers mach_i, blocking it is very bad (Case 1 logic applies strongly).
                    # But if mach_j != mach_i, it means dur_j < dur_j_on_mach_i. So extra is always positive here.
            
            # Optimization: Early break if cost + impact already exceeds best_cost?
            # impact is only positive, so if cost + impact > best_cost, we can break.
            if cost + impact > best_cost:
                break

        total_cost = cost + impact
        
        if total_cost < best_cost:
            best_cost = total_cost
            best_index = i

    # Return the best index and its best machine
    best_dur, best_mach, _ = op_details[best_index]
    if best_mach == -1:
        return -1
        
    return (best_index, best_mach)
