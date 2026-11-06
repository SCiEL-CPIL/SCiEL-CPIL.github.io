"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from functools import lru_cache


def _count_transitive_successors(n, pred_list):
    """Count transitive successors for each task using the predecessor structure."""
    # Build successor list (reverse of predecessors)
    succ = [[] for _ in range(n)]
    for i in range(n):
        for p in pred_list[i]:
            succ[p].append(i)
    
    # Count transitive successors via DFS/BFS
    count_succ = [0] * n
    for i in range(n):
        visited = set()
        stack = list(succ[i])
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                count_succ[i] += 1
                stack.extend(succ[node])
    return count_succ


def _total_time_of_successors(n, times, pred_list):
    """Total time of all transitive successors for each task."""
    succ = [[] for _ in range(n)]
    for i in range(n):
        for p in pred_list[i]:
            succ[p].append(i)
    
    total_time_succ = [0.0] * n
    for i in range(n):
        visited = set()
        stack = list(succ[i])
        total = 0.0
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                total += times[node]
                stack.extend(succ[node])
        total_time_succ[i] = total
    return total_time_succ


def _compute_priority(instance):
    """
    Compute priority scores for assembly tasks.
    
    Strategy: Combine multiple heuristics to prioritize tasks that:
    1. Have many transitive successors (bottleneck tasks that block many others)
    2. Have long processing times (hard to fit)
    3. Have many immediate successors
    
    This should reduce the number of stations by making critical tasks eligible early.
    """
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    cycle_time = instance.get("cycle_time", 40)
    
    # Get predecessors list
    if "predecessors" in instance:
        pred_list = instance["predecessors"]
    elif "preds" in instance:
        pred_list = instance["preds"]
    else:
        # Default: no predecessors
        pred_list = [[] for _ in range(n)]
    
    # Ensure pred_list is a list of lists of ints
    if isinstance(pred_list, np.ndarray):
        pred_list = pred_list.tolist()
    pred_list = [list(p) for p in pred_list]
    
    # Count transitive successors
    count_succ = _count_transitive_successors(n, pred_list)
    count_succ = np.asarray(count_succ, dtype=float)
    
    # Total time of transitive successors
    total_time_succ = _total_time_of_successors(n, times, pred_list)
    total_time_succ = np.asarray(total_time_succ, dtype=float)
    
    # Number of immediate successors
    succ_count = np.zeros(n, dtype=float)
    for i in range(n):
        for p in pred_list[i]:
            succ_count[p] += 1
    
    # Normalize each component
    def normalize(x):
        mn = np.min(x)
        mx = np.max(x)
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)
    
    norm_times = normalize(times)
    norm_count_succ = normalize(count_succ)
    norm_total_time_succ = normalize(total_time_succ)
    norm_succ_count = normalize(succ_count)
    
    # Weighted combination
    # Prioritize tasks with many transitive successors (they unlock many other tasks)
    # and long processing times (hard to fit in remaining station capacity)
    weights = {
        'count_succ': 0.4,
        'times': 0.3,
        'total_time_succ': 0.2,
        'succ_count': 0.1
    }
    
    priority_scores = (
        weights['count_succ'] * norm_count_succ +
        weights['times'] * norm_times +
        weights['total_time_succ'] * norm_total_time_succ +
        weights['succ_count'] * norm_succ_count
    )
    
    return priority_scores


def priority(instance):
    """
    Score each assembly task; stations are filled with the highest-priority 
    eligible task that fits the cycle time.
    
    Returns: one score per task (higher = higher priority)
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    try:
        result = _compute_priority(instance)
        
        # If we have time, we could try to refine, but the current approach
        # is deterministic and fast, so just return
        return result
        
    except Exception as e:
        # Fallback: just use times as priority
        times = np.asarray(instance.get("times", [1.0]), dtype=float)
        return times
# EVOLVE-BLOCK-END
