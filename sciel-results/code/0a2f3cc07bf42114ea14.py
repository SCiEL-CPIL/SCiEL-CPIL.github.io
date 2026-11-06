import time
import numpy as np

# Global deadline guard
_START_TIME = time.time()
_DEADLINE_SECONDS = 115.0 # Safety margin before 120s/150s limit

def _is_deadline_exceeded():
    return (time.time() - _START_TIME) > _DEADLINE_SECONDS

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0 # Fallback, though shouldn't be called with empty list

    # If we are close to the deadline, return a simple heuristic immediately
    if _is_deadline_exceeded():
        # Fallback to simple Most Work Remaining
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    best_index = 0
    best_score = -float('inf')
    best_machine = None

    for i, op in enumerate(ready_operations):
        # 1. Determine the best machine for this operation (Shortest Processing Time)
        machine_options = op["machine_options"]
        if not machine_options:
            continue # Should not happen for a ready op
        
        # Find machine with minimum duration
        min_duration = float('inf')
        best_m_for_op = None
        for m, d in machine_options.items():
            if d < min_duration:
                min_duration = d
                best_m_for_op = m
        
        if min_duration == float('inf') or min_duration <= 0:
            # Fallback if no valid machine or zero duration
            min_duration = 1.0 
            best_m_for_op = list(machine_options.keys())[0] if machine_options else 0

        # 2. Calculate Priority Score
        # WSJF Variant: Urgency (Remaining Ops) / Cost (Min Duration)
        # This balances job criticality (longer chains) with efficiency (shorter steps)
        urgency = op["remaining_ops"]
        if urgency <= 0:
            urgency = 1.0 # Avoid division issues, though remaining_ops should be >= 1
        
        # Add a small component of remaining_work to break ties or add weight to total size
        # The prompt suggested "Remaining Ops as urgency". Let's stick to that primarily.
        # However, pure ops count might ignore huge single operations.
        # Let's try a blend: (remaining_ops * 1.0 + remaining_work * 0.1) / min_duration
        # Or just remaining_ops / min_duration.
        
        # The prompt's measured success used "Remaining Ops as urgency".
        score = urgency / min_duration
        
        # Tie-breaking: If scores are very close, prefer higher remaining_work?
        # Or just stick to the score.
        
        if score > best_score:
            best_score = score
            best_index = i
            best_machine = best_m_for_op

    # Return index and the selected machine
    # The interface allows (index, machine)
    return (int(best_index), best_machine)
