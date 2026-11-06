"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(instance):
    """
    Score each assembly task by its positional criticality (Longest Path from Source).
    
    This heuristic prioritizes tasks based on their Earliest Start Time (EST),
    which is the longest path from any source node to the task. This identifies
    tasks that are deep in the dependency chain and must wait for many predecessors.
    By giving them high priority, we ensure they are scheduled as soon as they
    become eligible, preventing them from being delayed and potentially reducing
    the total number of stations by keeping the critical path tight.
    
    The score is the ratio of the Earliest Start Time to the cycle time.
    """
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    
    # Build predecessor list and in-degree
    in_degree = np.zeros(n, dtype=int)
    pred_list = [[] for _ in range(n)]
    for i in range(n):
        preds = predecessors[i] if isinstance(predecessors[i], (list, tuple, np.ndarray)) else [predecessors[i]]
        for p in preds:
            pred_list[i].append(p)
            in_degree[i] += 1
    
    # Compute topological order using Kahn's algorithm
    from collections import deque
    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)
    
    topo_order = []
    in_deg_copy = in_degree.copy()
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        # Find successors by scanning all nodes? No, we need successor list for efficiency in topo sort.
        # But we only built pred_list. Let's build successor list too for topo sort.
        pass 
        
    # Re-do graph building to have both pred and succ lists for proper topo sort
    successors = [[] for _ in range(n)]
    in_degree = np.zeros(n, dtype=int)
    pred_list = [[] for _ in range(n)]
    for i in range(n):
        preds = predecessors[i] if isinstance(predecessors[i], (list, tuple, np.ndarray)) else [predecessors[i]]
        for p in preds:
            pred_list[i].append(p)
            successors[p].append(i)
            in_degree[i] += 1
            
    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)
            
    topo_order = []
    in_deg_copy = in_degree.copy()
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            in_deg_copy[succ] -= 1
            if in_deg_copy[succ] == 0:
                queue.append(succ)
                
    # Compute ES (Earliest Start Time) in topological order
    # ES[i] = max(ES[p] + times[p] for p in pred_list[i])
    # If no predecessors, ES[i] = 0
    ES = np.zeros(n, dtype=float)
    
    for node in topo_order:
        if pred_list[node]:
            max_pred_es = 0.0
            for p in pred_list[node]:
                val = ES[p] + times[p]
                if val > max_pred_es:
                    max_pred_es = val
            ES[node] = max_pred_es
        else:
            ES[node] = 0.0
            
    # Normalize by cycle time as per directive
    if cycle_time > 0:
        scores = ES / cycle_time
    else:
        scores = ES
        
    return scores
# EVOLVE-BLOCK-END
