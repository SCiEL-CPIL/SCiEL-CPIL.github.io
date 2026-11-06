"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Dispatch rule: greedy LPT approximation for makespan minimization.
    
    Selects the ready operation that best balances machine loads by:
      1. Prioritizing operations with large minimum processing times (LPT)
      2. Considering remaining work to avoid idle machines
      3. Accounting for machine flexibility to reduce scheduling conflicts
    """
    if not ready_operations:
        return 0
    
    if len(ready_operations) == 1:
        return 0
    
    scores = []
    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        
        durations = list(machine_options.values())
        if not durations:
            min_duration = 0
            max_duration = 0
            n_machines = 0
        else:
            min_duration = min(durations)
            max_duration = max(durations)
            n_machines = len(machine_options)
        
        # Primary: LPT - prioritize long operations to balance loads early
        # Use min_duration as the guaranteed processing time on any machine
        lpt_score = min_duration
        
        # Secondary: Remaining work urgency
        # Jobs with more remaining work should be progressed to prevent
        # them from becoming bottlenecks
        if remaining_ops > 0:
            urgency = remaining_work / remaining_ops
        else:
            urgency = 0
        
        # Tertiary: Flexibility penalty
        # Operations with many machine options are more flexible and can
        # be scheduled later; prioritize less flexible operations
        flexibility_penalty = 0.05 * n_machines
        
        # Combine: LPT dominates, then urgency, then flexibility
        # Weight LPT heavily to maintain load balancing
        score = lpt_score + 0.3 * urgency - flexibility_penalty
        
        scores.append(score)
    
    scores = np.array(scores)
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
