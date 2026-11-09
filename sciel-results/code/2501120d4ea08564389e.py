"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

def priority(instance):
    start_time = time.time()
    deadline = 115  # Safety margin before 120s limit
    
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    n = len(times)
    
    if n == 0:
        return np.array([])
        
    # 1. Build successor graph
    successors = [[] for _ in range(n)]
    
    for j in range(n):
        preds = predecessors[j]
        for i in preds:
            successors[i].append(j)
            
    # Correct In-Degree Calculation for Kahn's Topological Sort
    # in_degree[i] = number of predecessors of i
    in_degree = [0] * n
    for j in range(n):
        for i in predecessors[j]:
            in_degree[j] += 1
            
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    # Handle potential cycles gracefully by appending remaining nodes
    if len(topo_order) < n:
        remaining = [i for i in range(n) if i not in topo_order]
        topo_order.extend(remaining)
        
    # 2. Calculate Positional Weights
    # w(i) = t(i) + sum(w(j) for j in successors(i))
    # Process in reverse topological order (sinks to sources)
    pos_weights = times.copy()
    
    for u in reversed(topo_order):
        for v in successors[u]:
            pos_weights[u] += pos_weights[v]
            
    # 3. Calculate Immediate Successor Count (ISC)
    # Tasks with fewer immediate successors are often "tighter" in the end
    # and should be prioritized to prevent bottlenecks at the end of the line.
    isc = np.array([len(successors[i]) for i in range(n)], dtype=float)
    
    # 4. Combine Heuristics
    # Primary: Positional Weight (Longest Path)
    # Secondary: Immediate Successor Count (Fewer is better, so we negate or subtract)
    # We want to maximize the score.
    # High positional weight is good.
    # Low immediate successor count is good.
    # So: Score = pos_weights - alpha * isc
    # We choose a small alpha so that positional weight dominates,
    # but ties are broken by fewer successors.
    
    # Normalize isc to be in a similar range to pos_weights to avoid scaling issues?
    # Or just use a small constant.
    # pos_weights can be large (sum of times).
    # isc is small (integer count).
    # Let's try subtracting a small fraction of the max time or just a small constant.
    # Actually, a common heuristic is "Rank" = (pos_weight, -isc).
    # Since we return a single float, we can do:
    # priority = pos_weights - 0.01 * isc
    # This ensures that if two tasks have very close positional weights,
    # the one with fewer successors is preferred.
    
    # Refined Heuristic:
    # 1. Base: Positional Weight (Longest Path to End)
    # 2. Tie-breaker 1: Task Duration (LPT-like). Larger tasks are more constrained by Cycle Time.
    #    Adding a fraction of the task's own time to the positional weight emphasizes
    #    the "size" of the task, which helps avoid stranding large tasks at the end.
    # 3. Tie-breaker 2: Immediate Successor Count. Tasks with more successors unlock
    #    more downstream work, so they are slightly more critical.
    #
    # Score = pos_weights + 0.5 * times + 0.1 * isc
    #
    # Note: pos_weights already includes times. So this is effectively:
    # Score = 1.5 * times + sum(successor positional weights) + 0.1 * isc
    #
    # The 0.5 coefficient ensures that if two tasks have the same downstream path
    # weight, the one with the larger own-time is preferred.
    # The 0.1 coefficient for isc is small enough to be a secondary tie-breaker
    # but significant enough to break ties when times are identical.
    
    final_priority = pos_weights + 0.5 * times + 0.1 * isc
    
    # Check time
    if time.time() - start_time > deadline - 5:
         # Fallback to simple LPT if somehow slow (unlikely)
         return times
         
    return final_priority
# EVOLVE-BLOCK-END
