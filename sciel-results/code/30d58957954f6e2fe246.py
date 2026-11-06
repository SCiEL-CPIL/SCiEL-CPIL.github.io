"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

# Global deadline guard variables
_START_TIME = time.time()
_TIME_LIMIT = 120.0
_SAFETY_MARGIN = 10.0

def _time_exceeded():
    """Check if we are approaching the time limit."""
    return (time.time() - _START_TIME) > (_TIME_LIMIT - _SAFETY_MARGIN)

def dispatch(ready_operations, time):
    # Explicit deadline guard
    if _time_exceeded():
        if not ready_operations:
            return 0
        return 0

    if not ready_operations:
        return 0
        
    best_score = float('-inf')
    best_index = 0
    best_machine = None
    
    for i, op in enumerate(ready_operations):
        if _time_exceeded():
            break

        remaining_work = op.get("remaining_work", 0.0)
        machine_options = op.get("machine_options", {})
        remaining_ops = op.get("remaining_ops", 0)
        
        if not machine_options:
            score = 0.0
            best_machine_local = 0
        else:
            # For each eligible machine, compute a composite score
            # that balances:
            #   1) Critical Ratio (CR) urgency: lower CR = more urgent
            #   2) Machine efficiency: shorter duration on that machine is better
            #   3) Job completion proximity: fewer remaining ops = closer to done
            
            min_duration = float('inf')
            best_machine_local = next(iter(machine_options.keys()))
            
            # First pass: find the fastest machine for this operation
            for m, dur in machine_options.items():
                if dur < min_duration:
                    min_duration = dur
                    best_machine_local = m
            
            if min_duration <= 0:
                min_duration = 1.0
            
            # Critical Ratio: (time until due) / (remaining work)
            # We don't have explicit due dates, so we use remaining_ops as a proxy.
            # Operations with fewer remaining ops are "closer to completion" and
            # should be prioritized to reduce makespan.
            
            # Urgency score: higher when remaining_work is small relative to remaining_ops
            # This favors operations that are close to finishing their job
            if remaining_ops > 0:
                urgency = remaining_work / (remaining_ops * min_duration + 1e-6)
            else:
                urgency = 0.0
            
            # Efficiency score: shorter processing time on the chosen machine
            efficiency = 1.0 / (min_duration + 1e-6)
            
            # Combined score: weight urgency and efficiency
            # Higher score = more likely to be dispatched
            score = 0.6 * urgency + 0.4 * efficiency
            
            # Small bonus for operations that are the last in their job
            if remaining_ops <= 1:
                score += 0.05

        if score > best_score:
            best_score = score
            best_index = i
            best_machine = best_machine_local
            
    return (int(best_index), best_machine)
# EVOLVE-BLOCK-END
