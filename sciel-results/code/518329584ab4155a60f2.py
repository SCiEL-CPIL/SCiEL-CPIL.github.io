"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(instance):
    """
    Score each assembly task by its positional criticality (Longest Path to Sink).
    
    The goal is to minimize the number of stations. A standard heuristic is to
    prioritize tasks that lie on the critical path, specifically those with the
    longest remaining duration downstream. This is calculated as the longest
    path from the task to any sink (successor), including the task's own duration.
    
    By placing these high-priority tasks first, we ensure that the most
    time-consuming chains are started early, which helps in packing the
    remaining tasks into the cycle time limits more efficiently and reducing
    the total number of stations.
    """
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    predecessors = instance["predecessors"]
    
    # Build successor list and in-degree
    in_degree = np.zeros(n, dtype=int)
    successors = [[] for _ in range(n)]
    for i in range(n):
        preds = predecessors[i] if isinstance(predecessors[i], (list, tuple, np.ndarray)) else [predecessors[i]]
        for p in preds:
            successors[p].append(i)
            in_degree[i] += 1
    
    # Compute topological order using Kahn's algorithm
    from collections import deque
    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)
    
    topo_order = []
    in_deg_copy = in_degree.copy()
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            in_deg_copy[succ] -= 1
            if in_deg_copy[succ] == 0:
                queue.append(succ)
    
    # Compute LS (Longest path to Sink) in reverse topological order
    # LS[i] = times[i] + max(LS[s] for s in successors[i])
    # If no successors, LS[i] = times[i]
    LS = np.zeros(n, dtype=float)
    
    for node in reversed(topo_order):
        max_succ_ls = 0.0
        for succ in successors[node]:
            if LS[succ] > max_succ_ls:
                max_succ_ls = LS[succ]
        LS[node] = times[node] + max_succ_ls
    
    # The priority is the LS value. Higher is better.
    return LS
# EVOLVE-BLOCK-END
