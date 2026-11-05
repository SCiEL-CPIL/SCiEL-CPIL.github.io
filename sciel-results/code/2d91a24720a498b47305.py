"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time


def dispatch(ready_operations, time):
    # Deadline guard
    start_time = time.time()
    # The evaluator kills after 150s. We are inside a single dispatch call.
    # A single dispatch call should be fast. 0.1s is plenty.
    deadline = start_time + 0.1
    
    best_idx = 0
    best_score = -float('inf')
    
    for i, op in enumerate(ready_operations):
        if time.time() > deadline:
            break
        
        # Calculate minimum duration for this operation
        # machine_options is {machine: duration}
        if not op["machine_options"]:
            continue
            
        min_duration = min(op["machine_options"].values())
        
        # Strategy: Critical Path Slack
        # Slack = Remaining Work - Minimum Completion Time of this Op
        # Min Completion Time = current_time + min_duration
        # Since current_time is constant for all ops in this batch,
        # we maximize (remaining_work - min_duration)
        score = op["remaining_work"] - min_duration
        
        if score > best_score:
            best_score = score
            best_idx = i
            
    return best_idx
# EVOLVE-BLOCK-END
