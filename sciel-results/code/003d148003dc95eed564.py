import numpy as np

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # Pre-compute primary contention metric and secondary urgency metrics
    contention_scores = np.zeros(n)
    urgency_scores = np.zeros(n)
    
    max_rw = 0.0
    max_ro = 0
    
    for i, op in enumerate(ready_operations):
        mo = op.get("machine_options", {})
        if not mo:
            # If no machine options, we can't schedule it effectively, give low score
            contention_scores[i] = 0.0
        else:
            durs = list(mo.values())
            md = min(durs)
            sd = sum(durs)
            
            if sd > 0:
                # High ratio means durations are similar (less dependency on a specific fast machine)
                # Low ratio means one machine is much faster (high dependency/contention risk)
                contention_scores[i] = md / sd
            else:
                contention_scores[i] = 1.0 # Zero duration is ideal
        
        rw = op.get("remaining_work", 0.0)
        ro = op.get("remaining_ops", 0)
        
        # Normalize urgency for tie-breaking
        if rw > max_rw:
            max_rw = rw
        if ro > max_ro:
            max_ro = ro
            
        urgency_scores[i] = 0.0 # Placeholder, will update after max found
        
    if max_rw == 0.0:
        max_rw = 1.0
    if max_ro == 0:
        max_ro = 1.0
        
    # Second pass to compute normalized urgency
    for i, op in enumerate(ready_operations):
        rw = op.get("remaining_work", 0.0)
        ro = op.get("remaining_ops", 0)
        # Combine remaining work and ops count for a robust urgency signal
        urgency_scores[i] = (rw / max_rw + ro / max_ro) / 2.0

    # Primary driver: Contention Avoidance (md/sd)
    # Secondary driver: Urgency (to prevent starvation and handle ties)
    # The weight of urgency is kept very low to strictly adhere to the contention-focused directive
    alpha = 0.01
    
    final_scores = contention_scores + alpha * urgency_scores
    
    best_idx = int(np.argmax(final_scores))
    return best_idx
