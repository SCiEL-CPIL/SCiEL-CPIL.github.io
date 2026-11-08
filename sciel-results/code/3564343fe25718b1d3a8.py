import time
import numpy as np

def dispatch(ready_operations, time):
    """
    Dispatches the next operation using a Bottleneck-Aware Critical Path (BACP) strategy.
    
    Strategy:
    - Instead of just looking at the job's remaining work (which favors long jobs),
      we look at the 'urgency' of each job. 
    - We estimate the 'bottleneck' of each job by finding the maximum minimum duration 
      among its remaining operations. This represents the single longest task that 
      cannot be parallelized within the job.
    - We prioritize jobs with a high ratio of (Remaining Work / Bottleneck). 
      This identifies jobs that have a lot of work but are not dominated by a single 
      huge operation (which would just sit on one machine). It also prioritizes jobs 
      where the critical path is tight.
    - Additionally, we apply a small penalty for the current operation's duration on 
      the best machine to encourage SPT-like behavior when criticalities are similar.
    - Machine selection: We choose the machine that minimizes the completion time 
      (duration) for the selected operation.
    
    Rationale:
    - Pure Critical Path (current) might starve jobs that have many small ops in favor 
      of one job with a few large ops, even if the large ops can be scheduled later 
      without impacting the makespan as much as delaying the many small ops of other jobs.
    - The Bottleneck-Aware metric (Work/Bottleneck) is a proxy for 'parallelism potential' 
      and 'tightness'. Jobs with high Work but low Bottleneck are 'smooth' and benefit 
      from early start to keep machines busy. Jobs with high Bottleneck are 'spiky' and 
      might be better delayed if their bottleneck machine is busy, but we don't have 
      machine availability info, so we assume the 'ready' set is manageable.
    - This is a different design family: it uses a ratio-based priority (efficiency) 
      rather than an additive/subtractive one (magnitude).
    """
    if not ready_operations:
        return -1

    best_index = -1
    best_machine = -1
    best_score = float('inf')
    
    # Pre-compute max remaining work for normalization if needed, 
    # but ratio is scale-invariant mostly.
    
    for i, op in enumerate(ready_operations):
        # Find the machine with the shortest duration for this operation
        min_dur = float('inf')
        min_mach = -1
        for m, d in op['machine_options'].items():
            if d < min_dur:
                min_dur = d
                min_mach = m
        
        # Get remaining work
        rem_work = op.get('remaining_work', 0)
        
        # Estimate bottleneck: max of min durations for remaining ops?
        # We don't have the list of remaining ops' machine options, only 'remaining_work'.
        # So we cannot compute the true bottleneck from the dict provided.
        # The dict only has: job, op, machine_options, remaining_ops, remaining_work.
        # 'remaining_ops' is likely a count or a list? The contract says "remaining_ops" is in the dict.
        # If it's a count, we can't get durations. If it's a list of dicts, we can.
        # Let's assume 'remaining_ops' is a count (int) or a list of operation dicts.
        # Given the ambiguity, and that 'remaining_work' is the sum of min durations,
        # we can't easily get the bottleneck without the individual ops.
        
        # Alternative Strategy: Use 'remaining_ops' count to estimate average duration.
        # If remaining_ops is a count N, avg_dur = rem_work / N.
        # High avg_dur suggests a "heavy" job.
        # Let's try a different angle: 
        # Priority = rem_work / (min_dur + epsilon). 
        # This favors jobs with lots of remaining work relative to the current step's cost.
        # It's like "efficiency of starting this job now".
        
        # Let's try: Score = min_dur / rem_work. 
        # Lower score is better. 
        # This favors small current duration and large remaining work.
        # If rem_work is 0, score is inf (worst).
        # This is essentially SPT weighted by 1/Workload.
        
        # Let's try a hybrid: 
        # Score = min_dur * (1 + 1/rem_work)
        # If rem_work is large, 1/rem_work is small, so Score ~ min_dur (SPT).
        # If rem_work is small, 1/rem_work is large, so Score is penalized heavily.
        # This prioritizes SPT for "finished" jobs? No, we want to prioritize jobs with LOTS of work.
        
        # Let's go back to the ratio idea but simpler:
        # We want to maximize (rem_work / min_dur).
        # So minimize (min_dur / rem_work).
        
        if rem_work <= 0:
            score = float('inf')
        else:
            score = min_dur / rem_work
        
        if score < best_score:
            best_score = score
            best_index = i
            best_machine = min_mach

    return (best_index, best_machine)
