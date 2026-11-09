import numpy as np
import time as time_module

def dispatch(ready_operations, time):
        if not ready_operations:
            return 0 # Fallback, though simulator should not call if empty
        
        # Calculate scores
        # We want to minimize makespan.
        # Heuristic: Shortest Processing Time (SPT) is generally good for open shop.
        # Also, finishing jobs with fewer remaining ops is good.
        
        scores = []
        for op in ready_operations:
            # Get duration
            # Assuming machine_options has one entry or we need to pick the best machine?
            # If machine_options has multiple, we might need to choose the machine too?
            # The return value is "index of the ready operation".
            # If we can choose the machine, we should pick the machine with the shortest duration for that op.
            
            if op["machine_options"]:
                # Pick the machine with the shortest duration for this operation
                min_duration = min(op["machine_options"].values())
            else:
                min_duration = 0
            
            # Job urgency: fewer remaining ops is better
            urgency = op["remaining_ops"]
            
            # Combine: We want small duration and small urgency.
            # Simple sum: min_duration + urgency * weight
            # Let's try weight 1.0 for now, or normalize.
            # Durations and ops counts are small integers.
            score = min_duration + urgency * 0.5
            
            scores.append(score)
        
        # Pick the index with the minimum score
        min_idx = int(np.argmin(scores))
        return min_idx
