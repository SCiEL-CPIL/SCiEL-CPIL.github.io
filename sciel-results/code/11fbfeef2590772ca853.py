"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math

_start_time = _time.time()
_DEADLINE = 140.0  # 10s margin before 150s kill

def dispatch(ready_operations, time):
    """
    Dispatch rule: Critical Ratio (CR) Heuristic.
    
    Strategy: 
    The Critical Ratio (CR) is a dynamic scheduling heuristic that prioritizes jobs/operations 
    based on their "urgency" relative to their remaining workload. 
    CR = Slack / Remaining_Work
    
    In the absence of explicit due dates, we approximate the "due date" or "deadline pressure" 
    by assuming that the job's completion should ideally align with its remaining workload 
    if it had exclusive access to resources, but since we are in a competitive environment, 
    we use the current time as a reference point for "elapsed time" relative to the job's 
    total remaining effort.
    
    A more robust interpretation for job-shop without explicit due dates:
    We treat the "due date" as the time when the job *would* finish if it started now and 
    ran sequentially without interference. However, a common proxy in literature when due dates 
    are absent is to use the remaining processing time itself as a baseline for "expected 
    completion" and compare the current time to a target.
    
    Let's refine the CR definition for this context:
    Slack = (Estimated Completion Time if uninterrupted) - Current Time
    But since we don't know future machine contention, we approximate:
    
    For each ready operation:
    1. Calculate the minimum possible duration for this operation (min over machine options).
    2. Estimate the job's total remaining processing time (sum of min durations for all remaining ops).
       Note: 'remaining_work' in the input dict is likely the sum of durations for remaining ops.
       Let's use 'remaining_work' as the total remaining processing time for the job.
    
    The Critical Ratio for a job at time `t` is often defined as:
    CR = (Due Date - Current Time) / Remaining Processing Time
    
    Since we don't have Due Dates, we can use a heuristic where we assume the "ideal" 
    completion time is `Current Time + Remaining Work`. This makes CR = 1 always, which is useless.
    
    Alternative CR interpretation for competitive scheduling:
    We want to prioritize jobs that are "behind schedule". 
    If we assume a target completion time for all jobs is roughly proportional to their total work, 
    we can't distinguish them.
    
    Let's look at the "Slack" definition provided in the directive:
    "slack is estimated by subtracting the current time from the sum of remaining durations for all unfinished operations in that job"
    Wait, that would be: Slack = Remaining_Work - Current_Time.
    This implies that as time increases, slack decreases (becomes more negative), increasing urgency.
    
    Let's define:
    Remaining_Work (RW) = op['remaining_work']
    Current_Time (t) = time
    Slack = RW - t  (This seems odd dimensionally if t is large, but let's follow the directive's hint)
    
    Actually, a more standard CR without due dates in simulation contexts often uses:
    CR = (Current Time + Estimated Remaining Time) / Current Time? No.
    
    Let's stick to the directive's specific hint: 
    "prioritizes the operation with the lowest ratio of slack time to remaining processing time, 
    where slack is estimated by subtracting the current time from the sum of remaining durations"
    
    So:
    Slack = Remaining_Work - Current_Time
    Ratio = Slack / Remaining_Work
    
    We want the LOWEST ratio.
    If Remaining_Work is small and Current_Time is large, Slack becomes negative, making the ratio negative and very low. 
    This would prioritize jobs with little remaining work that have been waiting a long time? 
    Or does it prioritize jobs with large remaining work?
    
    Let's analyze:
    If RW = 10, t = 5, Slack = 5, Ratio = 0.5
    If RW = 100, t = 5, Slack = 95, Ratio = 0.95
    If RW = 2, t = 5, Slack = -3, Ratio = -1.5
    
    The lowest ratio is -1.5 (RW=2). This prioritizes short remaining jobs when time is large.
    This is essentially a Shortest Remaining Processing Time (SRPT) variant that gets more aggressive over time.
    
    However, if t is small (start of simulation), e.g., t=0:
    RW=10 -> Ratio = 1.0
    RW=100 -> Ratio = 1.0
    RW=2 -> Ratio = 1.0
    All are equal at t=0.
    
    As time progresses, jobs with smaller RW will have lower (more negative) ratios, thus getting prioritized.
    This is a dynamic form of SRPT.
    
    Is this a good heuristic for makespan?
    SRPT is known to minimize mean flow time, which often correlates with good makespan in job-shop if it prevents long jobs from blocking short ones that are part of critical paths.
    
    Let's implement this:
    1. For each ready operation, get `remaining_work` (RW).
    2. Calculate `slack = RW - time`.
    3. Calculate `cr = slack / RW`.
    4. Select the operation with the minimum `cr`.
    
    Tie-breaking: If CRs are equal, pick the one with the shortest immediate operation duration to free the machine quickly.
    """
    # Deadline guard
    if _time.time() - _start_time > _DEADLINE:
        return 0

    if not ready_operations:
        return 0

    if len(ready_operations) == 1:
        return 0

    best_index = 0
    best_cr = float('inf')
    best_min_duration = float('inf')

    for i, op in enumerate(ready_operations):
        rw = op.get("remaining_work", 1.0)
        if rw <= 0:
            rw = 1.0
        
        # Calculate Critical Ratio based on directive
        slack = rw - time
        cr = slack / rw
        
        # Tie-breaker: shortest immediate processing time
        machine_options = op.get("machine_options", {})
        if machine_options:
            min_dur = min(machine_options.values())
        else:
            min_dur = float('inf')
            
        if cr < best_cr or (cr == best_cr and min_dur < best_min_duration):
            best_cr = cr
            best_min_duration = min_dur
            best_index = i

    return int(best_index)
# EVOLVE-BLOCK-END
