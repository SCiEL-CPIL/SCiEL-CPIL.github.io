"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def priority(instance):
    """
    Computes priority for each task based on Critical Path Density and Fanout.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin from 150s limit

    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    predecessors = instance["predecessors"]
    
    # 1. Build successor lists
    # predecessors[i] is a list of indices j such that j -> i
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
    
    # 2. Compute Longest Path Length (L_i) for each task
    # L_i = t_i + max(L_j for j in successors[i])
    # We need topological order.
    # Since it's a DAG, we can use Kahn's algorithm or DFS.
    # Given n=30, even O(n^2) is fine, but let's do it properly.
    
    # Calculate in-degrees for topological sort (or just use the structure)
    # Actually, we can compute L_i by processing tasks in reverse topological order.
    # Let's find topological order.
    
    in_degree = [0] * n
    for i in range(n):
        for p in predecessors[i]:
            in_degree[i] += 1
            
    queue = [i for i in range(n) if in_degree[i] == 0]
    topo_order = []
    while queue:
        # Pop node
        node = queue.pop(0) 
        topo_order.append(node)
        for s in successors[node]:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)
                
    # If topo_order length != n, there's a cycle (shouldn't happen in ALB)
    # Fallback to LTF if cycle detected or error
    if len(topo_order) != n:
        return times.copy()

    # Compute L_i in reverse topological order
    L = [0.0] * n
    for node in reversed(topo_order):
        max_succ_L = 0.0
        for s in successors[node]:
            if L[s] > max_succ_L:
                max_succ_L = L[s]
        L[node] = times[node] + max_succ_L
        
    # 3. Compute Successor Counts
    c = [len(successors[i]) for i in range(n)]
    
    # 4. Compute Priorities
    # Strategy: Ratio (t_i / L_i) combined with Fanout (c_i)
    # Priority_i = (t_i / L_i) * (1 + c_i)
    
    priorities = np.zeros(n, dtype=float)
    for i in range(n):
        if L[i] > 0:
            ratio = times[i] / L[i]
        else:
            ratio = 0.0
        priorities[i] = ratio * (1 + c[i])
        
    # Check time
    if time.time() > deadline:
        # Return best effort (which is already computed and fast)
        pass
        
    return priorities
# EVOLVE-BLOCK-END
