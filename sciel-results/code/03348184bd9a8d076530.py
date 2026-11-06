"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    n_tasks = len(times)
    
    # Build successor list
    # predecessors[i] is a list of task indices that must precede i
    successors = [[] for _ in range(n_tasks)]
    
    for i in range(n_tasks):
        for p in predecessors[i]:
            successors[p].append(i)
            
    # Calculate Ranked Position Weight (RPW)
    # RPW[i] = times[i] + sum(times[j] for j in successors[i])
    # This approach prioritizes tasks that have heavy immediate successors,
    # which is a different heuristic family than Bottom Level (Max-based).
    
    scores = np.zeros(n_tasks, dtype=float)
    
    for i in range(n_tasks):
        base_time = times[i]
        succ_time_sum = 0.0
        for j in successors[i]:
            succ_time_sum += times[j]
        scores[i] = base_time + succ_time_sum
        
    return scores
# EVOLVE-BLOCK-END
