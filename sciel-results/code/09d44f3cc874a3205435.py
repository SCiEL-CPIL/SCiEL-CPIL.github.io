"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(instance):
    """
    Compute priority scores for each assembly task using a Critical Path Density metric.
    
    The score combines:
    1. The ratio of the task's duration to the total length of its longest downstream path
       (higher means the task is a larger fraction of its critical path, i.e., a bottleneck).
    2. The count of immediate successors (higher means more branching points downstream).
    
    Tasks that are both bottlenecks in the critical path and high-fanout branching points
    are prioritized to reduce downstream fragmentation and station count.
    """
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    n = len(times)
    
    # Build successor lists from predecessors
    # predecessors[i] is a list of task indices that must precede task i
    successors = [[] for _ in range(n)]
    for i in range(n):
        if i in predecessors:
            for pred in predecessors[i]:
                successors[pred].append(i)
    
    # Compute longest downstream path (including self) for each task
    # Use memoization with topological order
    # First, compute in-degrees to do a topological sort
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        if i in predecessors:
            for pred in predecessors[i]:
                in_degree[i] += 1
    
    # Topological sort using Kahn's algorithm
    from collections import deque
    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)
    
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)
    
    # Compute longest downstream path in reverse topological order
    longest_downstream = np.zeros(n, dtype=float)
    for node in reversed(topo_order):
        if not successors[node]:
            longest_downstream[node] = times[node]
        else:
            max_succ = max(longest_downstream[s] for s in successors[node])
            longest_downstream[node] = times[node] + max_succ
    
    # Compute immediate successor count
    succ_count = np.array([len(successors[i]) for i in range(n)], dtype=float)
    
    # Critical Path Density: ratio of task duration to longest downstream path
    # Avoid division by zero
    cpd = np.zeros(n, dtype=float)
    nonzero_mask = longest_downstream > 0
    cpd[nonzero_mask] = times[nonzero_mask] / longest_downstream[nonzero_mask]
    
    # Combine: weight CPD heavily, add successor count as a tiebreaker/secondary factor
    # Normalize successor count to [0, 1] range if possible
    max_succ = succ_count.max() if n > 0 else 1
    if max_succ > 0:
        normalized_succ = succ_count / max_succ
    else:
        normalized_succ = succ_count
    
    # Final priority: CPD is the primary metric, successor count is secondary
    priority_scores = cpd + 0.1 * normalized_succ
    
    return priority_scores
# EVOLVE-BLOCK-END
