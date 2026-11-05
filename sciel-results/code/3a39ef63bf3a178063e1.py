"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

def priority(instance):
    """
    Compute priority scores for assembly line balancing tasks.
    
    Uses a composite score of Positional Weight, Successor Count, and Task Time.
    Positional Weight (PW) captures the critical path length from the task.
    Successor Count (SC) captures how many tasks are unlocked by this task.
    Task Time (T) captures the immediate load.
    
    The strategy up-weights high successor count tasks (bottlenecks) to keep
    the graph open, unlike the seed which down-weighted them.
    
    Returns: array of priority scores (higher = more urgent to schedule)
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    cycle_time = instance["cycle_time"]
    
    if n == 0:
        return np.array([])
        
    # Parse predecessors
    # predecessors[i] = list of task indices that must come before task i
    predecessors = instance["predecessors"]
    
    # Build successor lists
    successors = [[] for _ in range(n)]
    for i in range(n):
        preds = predecessors[i] if i < len(predecessors) else []
        for p in preds:
            successors[p].append(i)
    
    # Compute out-degree (number of immediate successors)
    out_degree = np.array([len(successors[i]) for i in range(n)], dtype=float)
    
    # Compute in-degree for topological sort
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        preds = predecessors[i] if i < len(predecessors) else []
        in_degree[i] = len(preds)
    
    # Topological sort using Kahn's algorithm
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)
    
    # If not all nodes are in topo_order, there's a cycle - fall back to times
    if len(topo_order) < n:
        return times.copy()
    
    # Compute positional weights in reverse topological order
    # pos_weight[i] = times[i] + sum(pos_weight[j] for j in successors[i])
    pos_weight = np.zeros(n, dtype=float)
    for node in reversed(topo_order):
        pos_weight[node] = times[node]
        for succ in successors[node]:
            pos_weight[node] += pos_weight[succ]
            
    # Check time limit before final normalization
    if time.time() > deadline:
        # Fallback to positional weight if we are running out of time
        return pos_weight.copy()

    # Normalize components
    max_pw = np.max(pos_weight) if np.max(pos_weight) > 0 else 1.0
    max_sc = np.max(out_degree) if np.max(out_degree) > 0 else 1.0
    max_t = np.max(times) if np.max(times) > 0 else 1.0
    
    # Composite Priority:
    # 1. Positional Weight (0.6): Primary driver for critical path
    # 2. Successor Count (0.3): Up-weighting bottlenecks to unlock graph
    # 3. Task Time (0.1): Minor boost for long tasks
    
    norm_pw = pos_weight / max_pw
    norm_sc = out_degree / max_sc
    norm_t = times / max_t
    
    priority_scores = 0.6 * norm_pw + 0.3 * norm_sc + 0.1 * norm_t
    
    return priority_scores
# EVOLVE-BLOCK-END
