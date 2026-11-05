"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import math

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    
    best_index = 0
    best_score = -float('inf')
    best_machine = None
    
    for i, op in enumerate(ready_operations):
        remaining_work = op["remaining_work"]
        remaining_ops = op.get("remaining_ops", 1)
        machine_options = op["machine_options"]
        
        if not machine_options:
            continue
        
        # Find the machine with the shortest duration (best machine for this op)
        min_duration = float('inf')
        min_machine = None
        for m, d in machine_options.items():
            if d < min_duration:
                min_duration = d
                min_machine = m
        
        # Soft-min surrogate: compute a smooth estimate of the "effective" duration
        # using log-sum-exp to avoid hard min discontinuities
        durations = list(machine_options.values())
        if len(durations) > 1:
            # Soft-min with temperature: lower temp → closer to hard min
            # soft_min = -log(sum(exp(-d/tau))) * tau
            tau = 1.0  # temperature
            neg_d_over_tau = np.array([-d / tau for d in durations])
            max_val = np.max(neg_d_over_tau)
            soft_min_val = -tau * (max_val + math.log(np.sum(np.exp(neg_d_over_tau - max_val))))
            soft_min_duration = -soft_min_val
        else:
            soft_min_duration = min_duration
        
        num_options = len(machine_options)
        
        # Heuristic Score:
        # 1. Prioritize jobs with more remaining work (urgency) - MWRF
        # 2. Prefer operations with shorter processing times (efficiency) - SPT
        # 3. Prefer operations with fewer machine options (most constrained first)
        # 4. Slight preference for jobs with fewer remaining operations (closer to done)
        #    but this is secondary to remaining_work
        
        # Weights tuned for makespan minimization:
        # - remaining_work dominates: jobs with more work should start early
        # - duration matters: shorter ops reduce idle time
        # - constraint: fewer options = more constrained, schedule earlier
        score = (
            remaining_work * 8.0
            - soft_min_duration * 1.5
            + num_options * 0.2
            - remaining_ops * 0.1
        )
        
        if score > best_score:
            best_score = score
            best_index = i
            best_machine = min_machine
    
    # Return the index and the chosen machine to allow for optimal machine selection
    return (best_index, best_machine)
# EVOLVE-BLOCK-END
