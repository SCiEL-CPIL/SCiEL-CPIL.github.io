"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    n_tasks = len(times)
    
    # Build successor list and in-degrees for topological sort
    successors = [[] for _ in range(n_tasks)]
    in_degree = np.zeros(n_tasks, dtype=int)
    
    for i in range(n_tasks):
        for p in predecessors[i]:
            successors[p].append(i)
            in_degree[i] += 1
            
    # Calculate Rank (Longest Path from Start)
    # Rank[i] = times[i] + max(Rank[p] for p in predecessors[i])
    # We need to process tasks in topological order (sources first).
    
    queue = [i for i in range(n_tasks) if in_degree[i] == 0]
    rank = np.zeros(n_tasks, dtype=float)
    
    # Use a simple queue (list) for BFS-like topological sort
    # Since we need max of predecessors, and predecessors are "upstream",
    # we process upstream first.
    
    # Initialize rank for sources
    for i in queue:
        rank[i] = times[i]
        
    processed_count = 0
    # We need to be careful with the queue. A standard topological sort
    # pops a node, processes it, and decrements in-degree of successors.
    # If in-degree of a successor becomes 0, we push it.
    # At that point, all its predecessors have been processed, so we can compute its rank.
    
    while queue:
        u = queue.pop(0) # Pop from front for BFS-like (order doesn't strictly matter for correctness of max, as long as all preds are done)
        processed_count += 1
        
        for v in successors[u]:
            # Update rank of v if this predecessor u gives a longer path
            # rank[v] = max(rank[v], rank[u] + times[v]) ? 
            # No, rank[v] = times[v] + max(rank[p] for all p in preds[v]).
            # We can accumulate the max predecessor rank.
            
            # Let's maintain a 'max_pred_rank' array or just update rank[v] incrementally?
            # rank[v] is not fully determined until all preds are processed.
            # We can store the max rank seen so far for v.
            
            if rank[u] > rank[v] - times[v]: # This logic is messy.
                pass
                
    # Let's restart the logic clearly.
    # We need an array to track the max rank of processed predecessors for each node.
    max_pred_rank = np.zeros(n_tasks, dtype=float)
    
    # Re-initialize queue
    queue = [i for i in range(n_tasks) if in_degree[i] == 0]
    # For sources, max_pred_rank is 0, so rank = times.
    for i in queue:
        rank[i] = times[i] # This is correct for sources.
        # But we need to propagate.
        
    # Let's use a different structure.
    # topological_order = []
    # q = deque([i for i in range(n_tasks) if in_degree[i] == 0])
    # while q:
    #    u = q.popleft()
    #    topological_order.append(u)
    #    for v in successors[u]:
    #        in_degree[v] -= 1
    #        if in_degree[v] == 0:
    #            q.append(v)
            
    # Then calculate rank in topological_order.
    # rank[i] = times[i] + max(rank[p] for p in predecessors[i])
    
    # This is O(N+E) for topo sort + O(E) for rank calc.
    
    # Let's implement this cleanly.
    from collections import deque
    
    q = deque([i for i in range(n_tasks) if in_degree[i] == 0])
    topo_order = []
    in_deg_copy = in_degree.copy()
    
    while q:
        u = q.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_deg_copy[v] -= 1
            if in_deg_copy[v] == 0:
                q.append(v)
                
    # Calculate Rank
    for u in topo_order:
        max_pred = 0.0
        for p in predecessors[u]:
            if rank[p] > max_pred:
                max_pred = rank[p]
        rank[u] = times[u] + max_pred
        
    return rank
# EVOLVE-BLOCK-END
