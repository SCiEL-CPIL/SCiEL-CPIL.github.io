import numpy as np

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # Pre-compute per-operation features
    min_durs = []
    sum_durs = []
    n_machines = []
    remaining_works = []
    remaining_ops_counts = []
    
    max_rw = 0.0
    max_ro = 0
    
    for op in ready_operations:
        mo = op.get("machine_options", {})
        if not mo:
            min_durs.append(float('inf'))
            sum_durs.append(float('inf'))
            n_machines.append(1)
            rw = op.get("remaining_work", 0.0)
            remaining_works.append(rw)
            ro = op.get("remaining_ops", 0)
            remaining_ops_counts.append(ro)
            if rw > max_rw:
                max_rw = rw
            if ro > max_ro:
                max_ro = ro
            continue
        
        durs = list(mo.values())
        md = min(durs)
        sd = sum(durs)
        nm = len(durs)
        
        min_durs.append(md)
        sum_durs.append(sd)
        n_machines.append(nm)
        
        rw = op.get("remaining_work", 0.0)
        remaining_works.append(rw)
        ro = op.get("remaining_ops", 0)
        remaining_ops_counts.append(ro)
        
        if rw > max_rw:
            max_rw = rw
        if ro > max_ro:
            max_ro = ro
    
    if max_rw == 0.0:
        max_rw = 1.0
    if max_ro == 0:
        max_ro = 1
    
    scores = np.zeros(n)
    
    for i in range(n):
        md = min_durs[i]
        sd = sum_durs[i]
        nm = n_machines[i]
        rw = remaining_works[i]
        ro = remaining_ops_counts[i]
        
        # Machine-contention avoidance: ratio of min duration to sum of durations
        # Higher ratio means more uniform processing times across machines,
        # suggesting less dependence on a single fast (potentially busy) machine
        if sd > 0 and md < float('inf'):
            contention_score = md / sd
        else:
            contention_score = 0.0
        
        # Speed component: prefer shorter minimum processing time
        if md < float('inf'):
            speed_score = 1.0 / (md + 1.0)
        else:
            speed_score = 0.0
        
        # Constraint: operations with fewer machine options are more constrained
        constraint_score = 1.0 / nm if nm > 0 else 0.0
        
        # Load: how much remaining work this job has
        load_score = rw / max_rw
        
        # Urgency: how many operations remain in this job
        urgency_score = ro / max_ro
        
        # Weighted combination
        # Emphasize contention avoidance and speed, with moderate weight on constraint
        # and load to break ties and handle edge cases
        scores[i] = (
            3.0 * contention_score +
            2.0 * speed_score +
            1.0 * constraint_score +
            0.5 * load_score +
            0.5 * urgency_score
        )
    
    best_idx = int(np.argmax(scores))
    return best_idx
