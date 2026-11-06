"""Assembly line balancing with bottleneck slack priority scoring."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict

def priority(instance):
    """
    Scores each assembly task based on a 'bottleneck slack' heuristic.
    
    The heuristic prioritizes tasks that consume a large fraction of the cycle time
    relative to the number of immediate successors they unlock.
    Score = -((cycle_time - time_i) / (1 + num_successors_i))
    Higher scores indicate higher priority.
    """
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    
    n = len(times)
    if n == 0:
        return np.array([])
    
    # Calculate number of immediate successors for each task
    num_successors = np.zeros(n, dtype=int)
    for i in range(n):
        for pred in predecessors[i]:
            num_successors[pred] += 1
    
    # Calculate bottleneck slack
    # Small slack means the task is "tight" or consumes a lot of capacity relative to branching
    # We want to prioritize tasks with SMALL slack, so we negate it for the priority score
    # Adding a small tie-breaker: prefer larger processing times if slacks are equal
    slack = (cycle_time - times) / (1.0 + num_successors)
    
    # Priority: Negated slack. 
    # We also add a small component of processing time to break ties in favor of larger tasks,
    # which is generally beneficial in ALB.
    # Normalization: Times are typically small integers. Slack is similar scale.
    # Let's just use -slack as the primary driver.
    # To ensure unique-ish scores and stable sorting, we can add a tiny fraction of time.
    # However, the greedy algorithm handles ties by picking the first one found in the eligible set.
    # The order of the eligible set depends on how we iterate. 
    # The prompt says "highest-priority eligible task". 
    # If we return -slack, tasks with the most "critical" position get picked first.
    
    # Let's refine: 
    # Priority = -slack + 0.01 * times 
    # This ensures that if two tasks have the same slack, the one taking longer is picked first.
    
    priority_scores = -slack + 0.01 * times
    
    return priority_scores
# EVOLVE-BLOCK-END
