"""Assembly line balancing with bottleneck slack priority scoring."""
# EVOLVE-BLOCK-START
import numpy as np

def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    
    n = len(times)
    if n == 0:
        return np.array([])
    
    # Calculate number of immediate successors for each task
    num_successors = np.zeros(n, dtype=float)
    for i in range(n):
        for pred in predecessors[i]:
            num_successors[pred] += 1
    
    # Calculate slack: cycle_time - time_i
    slack = cycle_time - times
    
    # Calculate bottleneck ratio: slack / (1 + num_successors)
    # We want to prioritize tasks with the SMALLEST ratio.
    # To make this a "highest priority" score, we invert it.
    # Add a small epsilon to avoid division by zero and handle negative slack gracefully.
    epsilon = 1e-9
    ratios = slack / (1.0 + num_successors)
    
    # Priority is inverse of ratio
    # Tasks with small slack and many successors get high priority.
    priority_scores = 1.0 / (ratios + epsilon)
    
    return priority_scores
# EVOLVE-BLOCK-END
