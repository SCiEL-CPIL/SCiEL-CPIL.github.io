"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

def priority(instance):
    """
    Calculates priority scores for assembly tasks.
    
    Uses a composite heuristic based on:
    1. Rank (Longest Path to end): Primary criticality measure.
    2. Total Downstream Time: Captures total volume of work dependent on the task.
    3. Number of Successors: Prioritizes tasks that unlock more subsequent work.
    4. Task Duration: Prioritizes longer tasks to ensure they fit (LPT-like heuristic).
    
    Secondary terms are normalized and scaled by cycle time to provide a smooth
    adjustment to the primary Rank score.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    times = np.array(instance["times"], dtype=float)
    n_tasks = len(times)
    preds = instance["predecessors"]
    cycle_time = instance.get("cycle_time", 40.0)
    
    # Build adjacency list for successors
    successors = [[] for _ in range(n_tasks)]
    in_degree = [0] * n_tasks
    
    for i in range(n_tasks):
        if i < len(preds):
            for p in preds[i]:
                # p is a predecessor of i, so i is a successor of p
                successors[p].append(i)
                in_degree[i] += 1
                
    # Topological Sort (Kahn's Algorithm)
    queue = deque()
    indeg = in_degree[:]
    for i in range(n_tasks):
        if indeg[i] == 0:
            queue.append(i)
            
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
                
    # Calculate Rank, Total Downstream, and Successor Count
    rank = np.zeros(n_tasks, dtype=float)
    total_down = np.zeros(n_tasks, dtype=float)
    num_succ = np.zeros(n_tasks, dtype=float)
    
    for i in reversed(topo_order):
        num_succ[i] = len(successors[i])
        
        if successors[i]:
            succ_ranks = [rank[j] for j in successors[i]]
            max_succ_rank = max(succ_ranks)
            
            succ_downs = [total_down[j] for j in successors[i]]
            sum_succ_down = sum(succ_downs)
        else:
            max_succ_rank = 0.0
            sum_succ_down = 0.0
        
        rank[i] = times[i] + max_succ_rank
        total_down[i] = times[i] + sum_succ_down
    
    # Check deadline
    if time.time() > deadline:
        return rank.copy()
    
    # Normalization
    # We normalize secondary features to [0, 1] and scale by cycle_time
    # to make their contribution comparable to the Rank scale (which is in time units).
    
    max_total_down = np.max(total_down)
    max_num_succ = np.max(num_succ)
    max_time = np.max(times)
    
    if max_total_down == 0: max_total_down = 1.0
    if max_num_succ == 0: max_num_succ = 1.0
    if max_time == 0: max_time = 1.0
    
    norm_total_down = total_down / max_total_down
    norm_num_succ = num_succ / max_num_succ
    norm_time = times / max_time
    
    # Composite Score
    # Primary: Rank (Critical Path)
    # Secondary: Total Downstream (Work Volume)
    # Tertiary: Successor Count (Branching/Unblocking)
    # Quaternary: Duration (Packing Efficiency)
    
    # Weights are chosen to ensure Rank is dominant but secondary factors
    # can break ties and guide packing.
    w_total_down = 0.25
    w_num_succ = 0.15
    w_time = 0.15
    
    scores = rank + \
             w_total_down * cycle_time * norm_total_down + \
             w_num_succ * cycle_time * norm_num_succ + \
             w_time * cycle_time * norm_time
    
    return scores

# EVOLVE-BLOCK-END
