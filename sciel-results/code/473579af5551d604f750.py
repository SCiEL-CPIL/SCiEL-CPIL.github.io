"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Dispatch rule using a Flexibility-Aware Critical Path Heuristic:
    Score = (Remaining Work / Min Duration) * Number of Machine Options
    
    This approach prioritizes operations that are both critical (high remaining work)
    and constrained (fewer machine options). By multiplying by the number of machine
    options, we effectively penalize flexible operations (many options) and boost
    constrained ones (few options). This ensures that jobs with limited routing
    choices are scheduled early to avoid blocking specific machines later in the
    schedule, which is a common source of makespan increases in job-shop problems.
    """
    if not ready_operations:
        return 0
    
    if len(ready_operations) == 1:
        return 0
    
    n = len(ready_operations)
    best_score = -1e18
    best_index = 0
    
    for i in range(n):
        op = ready_operations[i]
        remaining_work = op["remaining_work"]
        machine_options = op["machine_options"]
        remaining_ops = op["remaining_ops"]
        
        # Calculate minimum duration across available machines
        if machine_options:
            min_duration = min(machine_options.values())
            num_options = len(machine_options)
        else:
            min_duration = 1.0
            num_options = 1.0
            
        # Prevent division by zero
        if min_duration < 1e-9:
            min_duration = 1e-9
            
        # Base score: Critical Path Urgency
        base_score = remaining_work / min_duration
        
        # Flexibility Factor: Constrained jobs (low num_options) should be prioritized?
        # Wait, if num_options is 1, it's most constrained.
        # If we multiply by num_options, a job with 10 options gets 10x score?
        # That would prioritize FLEXIBLE jobs. We want to prioritize CONSTRAINED jobs.
        # So we should DIVIDE by num_options? Or use (1/num_options)?
        # If num_options = 1, factor = 1.
        # If num_options = 10, factor = 0.1.
        # This boosts constrained jobs.
        
        flexibility_factor = 1.0 / num_options
        
        total_score = base_score * flexibility_factor
        
        # Tie-breaker: Prefer more remaining operations (depth)
        tie_breaker = remaining_ops * 1e-4
        
        total_score += tie_breaker
        
        if total_score > best_score:
            best_score = total_score
            best_index = i
            
    return best_index
# EVOLVE-BLOCK-END
