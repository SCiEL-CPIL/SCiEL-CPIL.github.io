import numpy as np

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    scores = np.zeros(n)
    
    for i in range(n):
        op = ready_operations[i]
        mo = op.get("machine_options", {})
        
        if not mo:
            scores[i] = 0.0
            continue
            
        durs = list(mo.values())
        md = min(durs)
        sd = sum(durs)
        
        if sd <= 0:
            scores[i] = 0.0
            continue
            
        # Contention avoidance score: Higher ratio means less dependence on a single fast machine
        ratio = md / sd
        
        # Speed tie-breaker: Prefer shorter operations
        speed_bonus = 1.0 / (md + 1.0)
        
        # Weighted combination: Contention is primary
        scores[i] = 10.0 * ratio + 1.0 * speed_bonus
        
    best_idx = int(np.argmax(scores))
    return best_idx
