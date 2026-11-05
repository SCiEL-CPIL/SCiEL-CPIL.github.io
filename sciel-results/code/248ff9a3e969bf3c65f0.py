import numpy as np


def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    n = len(times)
    
    # Build successor list
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        for p in preds[i]:
            successors[p].append(i)
            in_degree[i] += 1
    
    # Topological sort (Kahn's algorithm) to get order
    from collections import deque
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for s in successors[node]:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)
    
    # Compute rank: longest path from node to end
    rank = np.zeros(n)
    # Process in reverse topological order
    for node in reversed(topo_order):
        if successors[node]:
            rank[node] = times[node] + max(rank[s] for s in successors[node])
        else:
            rank[node] = times[node]
    
    # Add small bonus for number of successors to break ties beneficially
    succ_counts = np.array([len(successors[i]) for i in range(n)], dtype=float)
    
    # Priority: rank dominates, successor count as tie-breaker
    priority_scores = rank + 0.01 * succ_counts
    
    return priority_scores
