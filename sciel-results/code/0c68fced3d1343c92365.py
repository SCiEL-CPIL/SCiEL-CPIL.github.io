import time as _time
import heapq

# Global state to track machine availability and job status across dispatch calls
# This allows for a stateful, event-driven strategy rather than a stateless per-call heuristic.
_machine_available_time = {}
_job_remaining_work = {}
_job_remaining_ops = {}
_last_dispatch_time = 0.0

# Deadline guard
_START_TIME = _time.time()
_DEADLINE = _START_TIME + 140.0

def dispatch(ready_operations, time):
    """
    Dispatch rule: MUJ-EAM (Most Urgent Job on Earliest Available Machine)
    
    Strategy:
    1. Update global state (machine availability, job remaining work) based on current time.
    2. For each ready operation, determine the earliest possible start time 
       (max(current_time, machine_available_time[machine])).
    3. Select the operation that minimizes the "Completion Time Estimate" 
       (earliest_start + duration) among all possible (op, machine) pairs,
       with a tie-breaker favoring jobs with more remaining work (urgency).
       
    This is structurally different from the previous static scoring because it:
    - Maintains persistent state of machine busy times.
    - Optimizes for actual completion time rather than a heuristic score.
    - Considers the temporal aspect of machine availability directly.
    """
    global _machine_available_time, _job_remaining_work, _job_remaining_ops, _last_dispatch_time
    
    if not ready_operations:
        return 0

    # Check deadline
    if _time.time() > _DEADLINE:
        return 0

    n = len(ready_operations)
    
    # 1. Update global state
    # We don't know exactly which machines finished when, but we can infer from the ready list.
    # However, a more robust approach for a stateful dispatcher in a black-box interface is to
    # update machine availability when we see an operation become ready.
    # Actually, the `time` parameter is the current simulation time.
    # We need to know when machines became free. 
    # Since we don't get explicit "machine_free" events, we must estimate.
    # BUT, the ready_operations list contains ops that are ready NOW.
    # If an op is ready, its job's previous op has finished.
    
    # Let's refine the state update:
    # We will track the earliest time each machine *could* be free.
    # Initially, all machines are free at t=0.
    # When we assign an op to a machine, we set its availability to finish_time.
    # The problem is: how do we know if a machine was busy with a *non-ready* op that just finished?
    # The `ready_operations` list only tells us about ready ops.
    
    # Alternative Structural Change: **Pure Local Search with Lookahead Simulation**
    # Instead of trying to maintain perfect global state (which is hard with this interface),
    # let's use the available data (ready_operations, time) to perform a shallow simulation
    # of the next few steps for each candidate, picking the one that leads to the best
    # "system state" (e.g., minimum max completion time among simulated ops).
    
    # This is a "Rollout" or "Beam Search" approach of depth 1-2.
    
    best_idx = 0
    best_machine = None
    best_score = float('inf')
    
    # Pre-compute some stats for tie-breaking
    # remaining_work is in the op dict
    
    for i, op in enumerate(ready_operations):
        options = op['machine_options']
        if not options:
            continue
        
        job_id = op['job']
        remaining_work = op.get('remaining_work', 0)
        remaining_ops = op.get('remaining_ops', 0)
        
        # For each machine option, estimate the "cost" of starting this op on this machine.
        # Cost = (time + duration)  # Completion time
        # But we also need to consider congestion.
        
        # Let's try a different structural idea: **Critical Path Method (CPM) Approximation**
        # The makespan is determined by the critical path.
        # An operation is critical if it lies on the longest path from start to finish.
        # We can approximate the criticality of an op by its "remaining work" and "remaining ops".
        
        # Heuristic: 
        # 1. Calculate the "slack" of the machine. 
        #    If we assign op to machine m, how much does it delay the machine?
        # 2. Calculate the "urgency" of the job.
        #    Jobs with more remaining work should be prioritized to keep the critical path short.
        
        # Let's combine:
        # Score = (Completion Time on Machine) / (1 + Urgency Factor)
        # Lower is better.
        
        # Urgency Factor: normalized remaining work.
        # We don't have the total work for all jobs, so let's use remaining_work directly.
        
        for m, d in options.items():
            # Estimate completion time
            # We assume the machine is free at `time` for simplicity, or we can track it.
            # Let's track machine availability globally to be more accurate.
            
            if m not in _machine_available_time:
                _machine_available_time[m] = 0.0
            
            # If the machine was scheduled to be free earlier than now, it's free now.
            # If it was scheduled to be free later, it's still busy.
            earliest_start = max(time, _machine_available_time[m])
            completion_time = earliest_start + d
            
            # Penalty for waiting: if earliest_start > time, the op has to wait.
            # This is bad for makespan if the job is urgent.
            
            # Urgency: remaining_work / remaining_ops (average work per op)
            # Or just remaining_work.
            
            # Let's use a weighted sum:
            # Primary: Completion Time (lower is better)
            # Secondary: Urgency (higher is better, so we subtract or divide)
            
            # Score = completion_time - alpha * remaining_work
            # We want to minimize this.
            
            # But remaining_work and time have different scales.
            # Let's normalize.
            
            # Simpler: 
            # If two ops finish at the same time, pick the one with more remaining work.
            # If two ops have the same remaining work, pick the one that finishes earlier.
            
            # Let's try: Score = completion_time * (1 + 1.0 / (remaining_work + 1))
            # This slightly penalizes long completion times for less urgent jobs? No.
            
            # Let's stick to a robust heuristic:
            # 1. Find the machine with the earliest completion time for this op.
            # 2. Among all ops, pick the one with the earliest completion time.
            # 3. Tie-break: Higher remaining work.
            
            # Let's implement this "Earliest Completion Time" (ECT) rule.
            
            # We need to update _machine_available_time if we pick this op.
            # But we are just evaluating.
            
            # Let's calculate the ECT for this (op, machine) pair.
            ect = completion_time
            
            # Tie-breaker: prefer higher remaining work
            # To combine into a single score where lower is better:
            # score = ect - 0.1 * remaining_work
            # This means: if remaining_work is high, the score is lower (better).
            
            score = ect - 0.01 * remaining_work
            
            if score < best_score:
                best_score = score
                best_idx = i
                best_machine = m

    # Update global state for the selected op
    if best_machine is not None:
        # The op will finish at best_score + 0.01 * remaining_work (approx)
        # Actually, we should store the actual completion time.
        # Let's re-calculate the completion time for the best choice to update state correctly.
        
        # Find the best op and machine again to get the exact duration
        best_op = ready_operations[best_idx]
        best_dur = best_op['machine_options'][best_machine]
        
        if best_machine not in _machine_available_time:
            _machine_available_time[best_machine] = 0.0
            
        actual_earliest_start = max(time, _machine_available_time[best_machine])
        actual_completion = actual_earliest_start + best_dur
        
        _machine_available_time[best_machine] = actual_completion
        
        # Update job remaining work? 
        # We don't know the exact work of the current op, but we can subtract it if we had it.
        # The op dict has 'machine_options' which gives duration, but not work.
        # 'remaining_work' is the total remaining work for the job *including* this op?
        # Usually, yes.
        # So, after dispatching, the job's remaining work decreases by the work of this op.
        # But we don't have the work of this op, only the duration on the machine.
        # Work = duration * parallelism? No, it's a single machine.
        # So Work = Duration? 
        # In job-shop, "work" is usually the processing time.
        # So we can subtract the duration from the job's remaining work.
        
        job_id = best_op['job']
        if job_id in _job_remaining_work:
            _job_remaining_work[job_id] -= best_dur
            _job_remaining_ops[job_id] -= 1
        else:
            # First time seeing this job? 
            # Initialize with the current remaining_work minus the current op's work.
            # But we don't know the current op's work separately.
            # Let's assume remaining_work includes the current op.
            _job_remaining_work[job_id] = best_op['remaining_work'] - best_dur
            _job_remaining_ops[job_id] = best_op['remaining_ops'] - 1

    return (best_idx, best_machine)
