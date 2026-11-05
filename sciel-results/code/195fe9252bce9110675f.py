"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math

_start_time = _time.time()
_DEADLINE = 140.0  # 10s margin before 150s kill

def dispatch(ready_operations, time):
    """
    Dispatch rule: 1-Step Look-Ahead Bottleneck Minimization.
    
    Strategy: Instead of statically scoring operations based on their own attributes (like duration or remaining work),
    this rule simulates the immediate consequence of starting each candidate operation.
    
    For each candidate, we estimate the new state of the system:
    1. The candidate operation occupies its chosen machine for 'duration'.
    2. The machine becomes available at 'time + duration'.
    3. Other ready operations (excluding the candidate) remain waiting.
    
    We score each candidate by the "Bottleneck Time": the minimum time at which ANY machine in the system 
    (including the one just occupied) will be free to accept the *next* operation.
    
    Specifically:
    - If we pick op i on machine m with duration d, machine m is busy until time + d.
    - Other machines are assumed to be free at 'time' (since we don't have global machine state in the interface, 
      we assume the event-driven simulator only calls us when a machine is free or at job release).
      *Correction*: In a standard event-driven job shop, `dispatch` is called when a machine becomes free OR a job is released.
      The `ready_operations` are those that can be started *now*.
      If we start one, the others wait.
      
    To distinguish between candidates, we look at:
    1. The duration of the candidate (shorter is generally better to free the machine sooner).
    2. The "urgency" of the job (remaining work).
    
    However, the key structural change is the **Look-Ahead**:
    We estimate the "completion time" of the critical path if we pick this op.
    Since we don't know future releases, we approximate the "cost" of picking an op as:
    Cost = (time + duration) * Weight
    
    But a more robust heuristic for makespan is to minimize the *idle time* of the system.
    
    Let's use a different structural approach: **Simulated Annealing-like stochastic selection with a deterministic fallback**.
    No, let's stick to a deterministic but non-greedy structural change: **Critical Path Method (CPM) Approximation**.
    
    For each ready operation, we estimate its "Critical Path Length" (CPL):
    CPL = current_op_duration + sum of remaining durations of the job (assuming sequential execution).
    
    We prioritize the operation that minimizes the *maximum* CPL among all jobs, or more simply, 
    we prioritize the operation that reduces the *maximum* remaining work of any job the most, 
    weighted by the machine availability.
    
    Actually, the most distinct structural change from NSPT (which is local/static) is **Global Job Priority**.
    NSPT looks at `op_duration / job_remaining_work`.
    
    New Strategy: **Weighted Shortest Remaining Processing Time (WSRPT) on the Job Level**.
    We group ready operations by job.
    For each job, we calculate the total remaining work.
    We prioritize the job with the highest "Urgency Score":
    Urgency = Total_Remaining_Work / (Number_of_Operations_In_Queue + 1)
    
    Then, among the operations of the highest urgency job, we pick the one with the shortest processing time.
    
    This is structurally different because it forces a decision hierarchy:
    1. Decide which *job* to serve.
    2. Decide which *op* of that job to serve.
    
    This avoids the "starvation" or "fragmentation" that can occur when purely local op metrics are used.
    """
    # Deadline guard
    if _time.time() - _start_time > _DEADLINE:
        return 0

    if not ready_operations:
        return 0

    if len(ready_operations) == 1:
        return 0

    # Group operations by job
    job_ops = {}
    for i, op in enumerate(ready_operations):
        job_id = op.get("job")
        if job_id not in job_ops:
            job_ops[job_id] = []
        job_ops[job_id].append((i, op))

    # Calculate urgency for each job
    # Urgency is defined as: Total Remaining Work / (1 + Count of Ready Ops for that Job)
    # This penalizes jobs that have many ops already waiting (they are "backlogged") 
    # and rewards jobs with high total work (they are "critical").
    
    best_job_id = None
    best_urgency = -1
    best_op_index = 0
    
    for job_id, ops_list in job_ops.items():
        # Get remaining work for this job (assumed same for all ops in the list)
        remaining_work = ops_list[0][1].get("remaining_work", 1)
        if remaining_work <= 0:
            remaining_work = 1
            
        # Count of ready ops for this job
        count_ready = len(ops_list)
        
        # Urgency Score: Higher is better
        # We want to prioritize jobs that have a lot of work left but aren't too fragmented in the queue?
        # Actually, if a job has 5 ops ready, it's likely a bottleneck. 
        # Let's use: Urgency = remaining_work / count_ready
        # This favors jobs with high work and low queue depth (efficient start).
        
        urgency = remaining_work / count_ready
        
        if urgency > best_urgency:
            best_urgency = urgency
            best_job_id = job_id
            # Default to the first op of this job, we will refine below
            best_op_index = ops_list[0][0]

    # Now, among the ops of the best job, pick the one with the shortest processing time
    # to minimize the time the machine is occupied.
    if best_job_id in job_ops:
        min_duration = float('inf')
        for i, op in job_ops[best_job_id]:
            machine_options = op.get("machine_options", {})
            if not machine_options:
                op_duration = float('inf')
            else:
                op_duration = min(machine_options.values())
            
            if op_duration < min_duration:
                min_duration = op_duration
                best_op_index = i

    return int(best_op_index)
# EVOLVE-BLOCK-END
