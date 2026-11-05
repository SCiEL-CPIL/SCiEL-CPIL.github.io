import numpy as np
from collections import defaultdict, deque

def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    # predecessors is likely a list of lists, where predecessors[i] is list of indices of tasks that must precede i
    # Or it's a dict? The seed uses instance["times"], instance["predecessors"].
    # Standard format in these tasks: predecessors[i] is a list of task indices.
    
    n = len(times)
    
    # Build successor graph
    # succ[i] is list of tasks that come after i
    succ = [[] for _ in range(n)]
    in_degree = [0] * n
    
    for i in range(n):
        if isinstance(predecessors, dict):
            preds = predecessors.get(i, [])
        else:
            preds = predecessors[i]
            
        for p in preds:
            succ[p].append(i)
            in_degree[i] += 1
            
    # Topological Sort (Kahn's algorithm) to get order
    # We need reverse topological order to compute ranks (from end to start)
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    temp_in_degree = in_degree[:]
    
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in succ[u]:
            temp_in_degree[v] -= 1
            if temp_in_degree[v] == 0:
                queue.append(v)
                
    # If topo_order length < n, there is a cycle. 
    # ALB instances are DAGs. If cycle, return times as fallback.
    if len(topo_order) < n:
        return times

    # Compute Ranks (Longest path from node to sink, inclusive)
    # Process in reverse topological order
    rank = [0.0] * n
    for u in reversed(topo_order):
        max_succ_rank = 0.0
        for v in succ[u]:
            if rank[v] > max_succ_rank:
                max_succ_rank = rank[v]
        rank[u] = times[u] + max_succ_rank
        
    # Compute Successor Counts
    succ_count = [len(succ[i]) for i in range(n)]
    
    # Compute Priority Score
    # Strategy: Critical Path Density = times[i] / rank[i]
    # Combined with Successor Count
    # Score = Density + SuccCount
    # To avoid division by zero (though rank > 0 if times > 0), add small epsilon
    eps = 1e-9
    density = times / (np.array(rank) + eps)
    
    # We want to prioritize high density and high fanout.
    # Let's try a weighted sum.
    # If rank is very large, density is small. 
    # Is this the right direction?
    # "Prioritize tasks that are... bottlenecks... and high-fanout"
    # High density = task is a big part of its path.
    # High fanout = task unlocks many.
    
    # Let's stick to the prompt's explicit combination.
    scores = density + np.array(succ_count, dtype=float)
    
    return scores
