"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    # Pre-compute job remaining works
    job_works = {}
    for op in ready_operations:
        j = op["job"]
        job_works[j] = op["remaining_work"]
        
    best_score_tuple = (float('inf'), float('inf'), float('inf'))
    best_idx = 0
    best_machine = None

    for i, op in enumerate(ready_operations):
        j = op["job"]
        current_work = op["remaining_work"]
        machine_options = op["machine_options"]
        
        # Select machine with Shortest Processing Time (SPT)
        # This aligns with the observed improvement in p40
        min_dur = float('inf')
        target_machine = None
        for m, d in machine_options.items():
            if d < min_dur:
                min_dur = d
                target_machine = m
                
        new_work_j = current_work - min_dur
        
        # Calculate resulting max job work
        # Max of all other jobs' works and this job's new work
        max_other_work = 0
        for j_other, w_other in job_works.items():
            if j_other != j:
                if w_other > max_other_work:
                    max_other_work = w_other
                    
        resulting_max = max(max_other_work, new_work_j)
        
        # Score tuple: 
        # 1. Minimize resulting max job work (Min-Max Degree on Jobs)
        # 2. Minimize remaining work on this job (Finish jobs faster)
        # 3. Minimize duration (SPT, keep machines free)
        
        score_tuple = (resulting_max, new_work_j, min_dur)
        
        if score_tuple < best_score_tuple:
            best_score_tuple = score_tuple
            best_idx = i
            best_machine = target_machine

    return (best_idx, best_machine)

# EVOLVE-BLOCK-END
