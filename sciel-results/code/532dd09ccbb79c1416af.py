import numpy as np

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    
    try:
        scores = []
        for op in ready_operations:
            durations = list(op["machine_options"].values())
            if not durations:
                scores.append(0.0)
                continue
            
            min_dur = min(durations)
            sum_dur = sum(durations)
            remaining_work = op["remaining_work"]
            
            # Handle zero duration
            if min_dur <= 0:
                min_dur = 1e-6
                
            # Heuristic: Most Work Remaining * Efficiency Factor
            # Efficiency Factor = Sum / Min. High when Min is small relative to Sum.
            # This favors operations that can be completed quickly on the best machine.
            score = remaining_work * (sum_dur / min_dur)
            scores.append(score)
            
        return int(np.argmax(scores))
    except Exception:
        return 0
