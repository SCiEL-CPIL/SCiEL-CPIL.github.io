"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(instance):
    """
    Score each assembly task for the greedy station-filling heuristic.
    
    The heuristic fills stations by repeatedly picking the highest-priority
    eligible task that fits in the remaining cycle time. We want to minimize
    the number of stations.
    
    Strategy: Use a combination of:
    1. Longest processing time (LPT) as a base - larger tasks are harder to place.
    2. Position in the precedence graph - tasks with many successors should be 
       prioritized to avoid blocking.
    3. A "critical path" like measure - tasks that are on long chains.
    
    We compute a score that combines these factors. Higher score = higher priority.
    
    Key insight: The greedy algorithm picks tasks one by one. To minimize stations,
    we want to:
    - Place large tasks early (they're harder to fit later)
    - Respect precedence (can't place a task before its predecessors)
    - Pack stations tightly
    
    A good heuristic is to prioritize tasks by:
    - Their own processing time (larger = higher priority)
    - The total work that depends on them (successors) - tasks that unlock a lot 
      of work should be placed early
    - Their position in the critical path
    """
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]  # list of lists, preds[i] = list of task indices that must come before i
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Build successor graph
    succs = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            succs[p].append(i)
    
    # Compute longest path from each task to any sink (task with no successors)
    # This gives us the "remaining critical path length" from each task
    # Use memoized DFS or topological order
    
    # Topological sort
    in_degree = [len(preds[i]) for i in range(n)]
    from collections import deque
    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)
    
    topo_order = []
    in_deg_copy = in_degree[:]
    q = deque()
    for i in range(n):
        if in_deg_copy[i] == 0:
            q.append(i)
    
    while q:
        u = q.popleft()
        topo_order.append(u)
        for v in succs[u]:
            in_deg_copy[v] -= 1
            if in_deg_copy[v] == 0:
                q.append(v)
    
    # Compute longest path from each node to a sink
    # longest_from[i] = times[i] + max(longest_from[j] for j in succs[i]), or times[i] if no successors
    longest_from = np.zeros(n)
    # Process in reverse topological order
    for i in reversed(topo_order):
        max_succ = 0.0
        for j in succs[i]:
            if longest_from[j] > max_succ:
                max_succ = longest_from[j]
        longest_from[i] = times[i] + max_succ
    
    # Compute longest path from any source to each node
    # longest_to[i] = times[i] + max(longest_to[j] for j in preds[i]), or times[i] if no predecessors
    longest_to = np.zeros(n)
    for i in topo_order:
        max_pred = 0.0
        for j in preds[i]:
            if longest_to[j] > max_pred:
                max_pred = longest_to[j]
        longest_to[i] = times[i] + max_pred
    
    # Number of successors (direct and transitive) - approximate by direct successors
    # Actually, let's compute the total number of transitive successors
    # This is expensive for large n, but n=30 so it's fine
    num_succs = np.zeros(n, dtype=int)
    # Use BFS/DFS from each node
    for i in range(n):
        visited = set()
        stack = list(succs[i])
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                stack.extend(succs[node])
        num_succs[i] = len(visited)
    
    # Combine scores:
    # Base: processing time (larger tasks first)
    # Add: longest_from (tasks that start long chains)
    # Add: num_succs (tasks that unlock a lot of work)
    
    # Normalize components
    if n > 0:
        times_norm = times / times.max() if times.max() > 0 else times
        lf_norm = longest_from / longest_from.max() if longest_from.max() > 0 else longest_from
        ns_norm = num_succs.astype(float) / num_succs.max() if num_succs.max() > 0 else num_succs.astype(float)
        
        # Weighted combination
        # Emphasize processing time and critical path
        score = 1.0 * times_norm + 0.8 * lf_norm + 0.3 * ns_norm
    else:
        score = np.array([])
    
    return score
# EVOLVE-BLOCK-END
