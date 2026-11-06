"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq
from collections import deque

def priority(instance):
    """
    Calculates priority scores for assembly tasks using a refined heuristic.
    
    The primary signal is the 'Rank' (Longest Path to end), which identifies tasks
    on the critical path. Secondary signals include total downstream work and
    connectivity to break ties and identify bottlenecks that rank alone might miss.
    
    This version adds 'Total Downstream Successors' count to better capture
    structural bottlenecks that unlock many parallel paths.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    times = np.array(instance["times"], dtype=float)
    n_tasks = len(times)
    preds = instance["predecessors"]
    
    # Build adjacency list for successors
    successors = [[] for _ in range(n_tasks)]
    in_degree = [0] * n_tasks
    
    for i in range(n_tasks):
        if i < len(preds):
            for p in preds[i]:
                # p is a predecessor of i, so i is a successor of p
                successors[p].append(i)
                in_degree[i] += 1
                
    # Topological Sort (Kahn's Algorithm) using a heap for deterministic order
    # We use a min-heap on indices to ensure deterministic processing order
    # This doesn't change the validity of the topo order but makes it reproducible
    heap = []
    indeg = in_degree[:]
    for i in range(n_tasks):
        if indeg[i] == 0:
            heapq.heappush(heap, i)
            
    topo_order = []
    while heap:
        u = heapq.heappop(heap)
        topo_order.append(u)
        for v in successors[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(heap, v)
                
    # If graph is cyclic or malformed, topo_order might be incomplete. Assume valid DAG.
    # If not all tasks are in topo_order, we need a fallback.
    if len(topo_order) != n_tasks:
        # Fallback: return times if graph is invalid
        return times.copy()

    # Calculate Rank (Longest Path), Total Downstream Time, and Total Downstream Successors
    rank = np.zeros(n_tasks, dtype=float)
    total_down_time = np.zeros(n_tasks, dtype=float)
    total_down_succ = np.zeros(n_tasks, dtype=float)
    
    # Iterate backwards through topological order
    for i in reversed(topo_order):
        # For rank: longest path through successors
        max_succ_rank = 0.0
        # For total_down_time: sum of all downstream work times
        sum_succ_down_time = 0.0
        # For total_down_succ: sum of all downstream successor counts
        sum_succ_down_succ = 0.0
        
        if successors[i]:
            succ_ranks = [rank[j] for j in successors[i]]
            max_succ_rank = max(succ_ranks)
            
            succ_down_times = [total_down_time[j] for j in successors[i]]
            sum_succ_down_time = sum(succ_down_times)
            
            succ_down_succs = [total_down_succ[j] for j in successors[i]]
            sum_succ_down_succ = sum(succ_down_succs)
        
        rank[i] = times[i] + max_succ_rank
        total_down_time[i] = times[i] + sum_succ_down_time
        total_down_succ[i] = 1.0 + sum_succ_down_succ # Count self + downstream
        
    # Immediate successors count
    num_succ = np.array([len(successors[i]) for i in range(n_tasks)], dtype=float)
    
    # Check deadline
    if time.time() > deadline:
        # Return a simple fallback if we ran out of time (unlikely for n=30)
        return rank.copy()
    
    # Composite Score:
    # Primary: Rank (critical path) - most important for minimizing stations
    # Secondary: Total downstream time - identifies true work bottlenecks
    # Tertiary: Total downstream successors - identifies structural bottlenecks (unlocking many paths)
    # Quaternary: Number of immediate successors - breaks ties
    
    # Weights:
    # Rank is the dominant factor.
    # Total Downstream Time is scaled down because it can be large (sum of many tasks).
    # Total Downstream Successors is scaled down as it's a count.
    
    # Tuned weights:
    # 1.0 * rank: Ensures critical path is respected.
    # 0.2 * total_down_time: Rewards tasks with heavy downstream work.
    # 0.1 * total_down_succ: Rewards tasks that unlock many subsequent tasks.
    # 0.05 * num_succ: Small tie-breaker for immediate connectivity.
    
    scores = 1.0 * rank + 0.2 * total_down_time + 0.1 * total_down_succ + 0.05 * num_succ
    
    return scores

# EVOLVE-BLOCK-END
