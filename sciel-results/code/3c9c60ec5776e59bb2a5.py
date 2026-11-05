"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Dispatch rule based on Critical Ratio (CR) heuristic.
    
    This implementation uses a Critical Ratio approach adapted for job-shop 
    scheduling without explicit due dates. The heuristic prioritizes operations 
    from jobs with the least remaining work, as these jobs are closest to 
    completion and finishing them quickly frees up resources and reduces 
    the overall makespan.
    
    The Critical Ratio is defined as:
        CR = (Slack Time) / (Remaining Processing Time)
    
    Where:
    - Remaining Processing Time is the `remaining_work` of the job.
    - Slack Time is estimated as `remaining_work - current_time`. 
      (Note: This is a proxy for "time available" relative to the work 
      required, assuming a baseline where work should have started at t=0).
    
    Minimizing the CR ratio (i.e., picking the most negative or smallest 
    positive value) effectively prioritizes jobs with smaller `remaining_work`, 
    because `time` is constant for all candidates in a single dispatch step.
    
    Tie-breaking:
    - If CR values are equal (or very close), we prefer the operation with 
      the shortest minimum processing time (SPT) to minimize machine blockage.
    """
    if not ready_operations:
        return 0
    
    best_idx = 0
    best_cr = float('inf')
    best_duration = float('inf')
    
    for i, op in enumerate(ready_operations):
        remaining_work = op["remaining_work"]
        
        # Get the minimum duration across all machine options for tie-breaking
        machine_options = op["machine_options"]
        if machine_options:
            min_duration = min(machine_options.values())
        else:
            min_duration = 1.0
        
        # Calculate Critical Ratio
        # Avoid division by zero
        denom = max(remaining_work, 0.1)
        slack = remaining_work - time
        cr = slack / denom
        
        # We want the LOWEST CR (most urgent/least slack relative to work)
        # If CRs are equal, we want the LOWEST duration (SPT)
        
        if cr < best_cr:
            best_cr = cr
            best_idx = i
            best_duration = min_duration
        elif cr == best_cr:
            if min_duration < best_duration:
                best_idx = i
                best_duration = min_duration
                
    return best_idx
# EVOLVE-BLOCK-END
