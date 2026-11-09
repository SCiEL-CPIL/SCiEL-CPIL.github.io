"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(instance):
    """
    Computes priority scores for assembly tasks based on the longest path to the end.
    Higher score means higher priority.
    """
    start_time = time.time()
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    n = len(times)
    
    # If the instance is very small or time is critical, fallback to simple LPT
    # But for N=30, the graph algorithm is extremely fast.
    if n == 0:
        return np.array([])
        
    # Build successor list
    successors = [[] for _ in range(n)]
    in_degree = np.zeros(n, dtype=int)
    
    # predecessors is a list of lists or similar structure.
    # Assume predecessors[i] is a list of indices of tasks that must be done before i.
    for i in range(n):
        preds = predecessors[i]
        for p in preds:
            successors[p].append(i)
            in_degree[i] += 1
            
    # Topological Sort (Kahn's algorithm) to get order
    # We need to process from end to start to calculate "rank" (longest path to end)
    # Alternatively, compute "depth" from start, but "rank" from end is better for scheduling.
    
    # Let's compute the longest path from each node to any sink.
    # This requires processing nodes in reverse topological order.
    
    # Get topological order
    queue = [i for i in range(n) if in_degree[i] == 0]
    topological_order = []
    temp_in_degree = in_degree.copy()
    
    # Standard Kahn's
    while queue:
        # Use a simple list as queue, pop from end for stack-like behavior or start for queue
        # For deterministic behavior, we can just iterate
        node = queue.pop(0) 
        topological_order.append(node)
        for s in successors[node]:
            temp_in_degree[s] -= 1
            if temp_in_degree[s] == 0:
                queue.append(s)
                
    # If topological order is not full length, there's a cycle (shouldn't happen in ALBP)
    # Fallback to LPT if cycle detected
    if len(topological_order) != n:
        return times.copy()

    # Calculate rank: longest path from node to end
    # rank[i] = times[i] + max(rank[s] for s in successors[i])
    # If no successors, rank[i] = times[i]
    
    ranks = np.zeros(n, dtype=float)
    
    # Process in reverse topological order
    for i in reversed(topological_order):
        max_succ_rank = 0.0
        for s in successors[i]:
            if ranks[s] > max_succ_rank:
                max_succ_rank = ranks[s]
        ranks[i] = times[i] + max_succ_rank
        
    # Tie-breaking: if ranks are equal, prefer longer tasks?
    # The greedy algorithm picks highest priority first.
    # If two tasks have the same rank, picking the one with larger time might help fill the station better.
    # We can combine them: priority = rank * (1 + epsilon) + time?
    # Or simply: if rank is the primary key, and time is secondary.
    # Since the evaluator picks "highest priority", we can just return the ranks.
    # However, to break ties in favor of longer tasks (which might be harder to fit later),
    # we can add a small fraction of the time to the rank.
    
    # Let's try: priority = rank + 0.1 * time
    # This ensures rank is dominant, but time breaks ties.
    
    priorities = ranks + 0.1 * times
    
    # Time guard check (though this is O(N+E) and very fast)
    if time.time() - start_time > 140:
        # Fallback to simple LPT if something went wrong and took too long
        return times.copy()

    return priorities

# EVOLVE-BLOCK-END
