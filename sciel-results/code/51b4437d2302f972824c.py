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
    pos_weights = times.copy()
    
    for u in reversed(topo_order):
        for v in successors[u]:
            pos_weights[u] += pos_weights[v]
            
    # 3. Calculate Immediate Successor Count (ISC)
    isc = np.array([len(successors[i]) for i in range(n)], dtype=float)
    
    # 4. Combine Heuristics
    # Primary: Positional Weight (Higher is better)
    # Secondary: Task Time (Higher is better, for tie-breaking)
    # Tertiary: Immediate Successor Count (Lower is better, for tie-breaking)
    
    # We want to maximize the score.
    # Scale PW to dominate.
    # Scale Time to be secondary.
    # Scale ISC to be tertiary (negated).
    
    # Max ISC to normalize? Or just use raw?
    # ISC is small (0-30). Time is small (0-40). PW is large (0-1200).
    
    # Priority = PW * 1000 + Time - ISC
    # This ensures PW is the main driver.
    # If PW is equal, larger Time wins.
    # If PW and Time are equal, smaller ISC wins.
    
    final_priority = pos_weights * 1000.0 + times - isc
    
    # Check time
    if time.time() - start_time > deadline - 5:
         # Fallback to simple LPT if somehow slow (unlikely)
         return times
         
    return final_priority
