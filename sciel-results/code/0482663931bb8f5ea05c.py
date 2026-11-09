import numpy as np

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    scores = []
    for op in ready_operations:
        R = op["remaining_work"]
        durations = list(op["machine_options"].values())
        d_min = min(durations) if durations else 0.0
        rem_ops = op["remaining_ops"]
        
        # Component 1: Work-to-time ratio
        score1 = R / (d_min + 1.0)
        
        # Component 2: Proximity to completion (higher if fewer ops left)
        # Using 1/(rem_ops+1) to avoid div by zero and scale
        score2 = 1.0 / (rem_ops + 1.0)
        
        # Combine
        # Weights: The ratio is likely dominant. 
        # Let's try equal weight first, or scale score2.
        # R can be large (e.g., 100), d_min small (1). score1 ~ 50.
        # rem_ops can be 10. score2 ~ 0.09.
        # So score1 dominates. That's probably fine.
        score = score1 + score2
        scores.append(score)
        
    return int(np.argmax(scores))
