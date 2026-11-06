"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

def dispatch(ready_operations, time):
    """
    Dispatch rule using Load-Aware Critical Path (LACP).
    
    Strategy:
    1. Calculate the 'Critical Path' for each job: the sum of the minimum 
       processing times for all remaining operations in that job.
    2. Determine the 'Preferred Machine' for the current ready operation: 
       the machine with the shortest duration for this specific op.
    3. Estimate 'Contention': the number of other ready operations that 
       also prefer the same machine. This approximates machine load/queue.
    4. Score = Critical Path - (alpha * Contention).
    5. Select the operation with the highest score.
    
    This balances the need to prioritize long jobs (makespan reduction) 
    against the need to avoid bottlenecks on heavily contended machines 
    (throughput/flow time reduction).
    """
    if not ready_operations:
        return 0

    n = len(ready_operations)
    
    # Deadline guard: This calculation is O(N^2) in the worst case if we 
    # count contention for each op against all others. 
    # N is typically small (<=10 jobs, so <=10 ready ops). 
    # 10*10 = 100 operations. Very fast.
    
    # Step 1: Precompute Critical Path and Preferred Machine for each ready op
    job_critical_paths = {}
    preferred_machines = []
    
    for i, op in enumerate(ready_operations):
        job_id = op['job']
        
        # Calculate critical path for this job
        # remaining_ops contains the ops remaining in the job.
        # We assume remaining_work is a reliable sum, but to be safe and 
        # consistent with 'min duration' logic, we sum min durations if possible.
        # If remaining_ops is a list of dicts with 'machine_options', we use that.
        # Otherwise, we fall back to remaining_work.
        
        rem_ops = op.get('remaining_ops', [])
        if rem_ops and isinstance(rem_ops, list) and len(rem_ops) > 0 and isinstance(rem_ops[0], dict):
            # Sum min durations for all remaining ops
            cp = 0
            for r_op in rem_ops:
                if 'machine_options' in r_op:
                    cp += min(r_op['machine_options'].values())
            # Ensure current op is included if not in remaining_ops?
            # Usually 'remaining_ops' includes the current one or the ones after.
            # The prompt says 'remaining_ops'. Let's assume it covers the tail.
            # If remaining_work is provided and seems more comprehensive, we might use it.
            # But 'min duration' is a better proxy for 'earliest possible finish'.
            job_critical_paths[job_id] = cp
        else:
            # Fallback to remaining_work
            job_critical_paths[job_id] = op['remaining_work']
            
        # Determine preferred machine for the CURRENT op
        machine_options = op['machine_options']
        # Find machine with min duration
        pref_m = min(machine_options, key=machine_options.get)
        preferred_machines.append(pref_m)

    # Step 2: Calculate Contention for each ready op
    # Contention for op i is the number of OTHER ready ops that prefer the same machine.
    # We can do this by counting frequencies of preferred_machines.
    
    machine_counts = {}
    for m in preferred_machines:
        machine_counts[m] = machine_counts.get(m, 0) + 1

    # Step 3: Calculate Score for each ready op
    # Score = Critical Path - alpha * Contention
    # We want to maximize Score.
    # If Contention is high, we penalize.
    # If Critical Path is high, we reward.
    
    # Tuning alpha: 
    # If alpha is too high, we ignore critical path and just pick least congested (SPT-like).
    # If alpha is 0, we just pick Longest Critical Path (MWR-like).
    # Let's try alpha = 0.5 or 1.0.
    # Since durations are integers, contention is integer.
    # Let's normalize? No, raw values should be okay for small N.
    
    alpha = 0.5
    
    scores = []
    for i, op in enumerate(ready_operations):
        job_id = op['job']
        cp = job_critical_paths.get(job_id, 0)
        pref_m = preferred_machines[i]
        contention = machine_counts.get(pref_m, 0)
        
        # We subtract contention to avoid congested machines.
        # Note: The current op itself is counted in contention. 
        # Should we subtract 1? 
        # If I'm the only one wanting Machine A, contention=1.
        # If I'm one of 5, contention=5.
        # The "extra" load is contention - 1.
        # Let's use (contention - 1) as the penalty for *other* jobs.
        penalty = (contention - 1) * alpha
        
        score = cp - penalty
        scores.append(score)

    # Return index of max score
    return int(np.argmax(scores))

# EVOLVE-BLOCK-END
