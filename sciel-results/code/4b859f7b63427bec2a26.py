"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(instance):
    """
    Compute priority scores for assembly line balancing.
    
    Strategy: Use a combination of:
    1. Task time (longer tasks first)
    2. Number of successors (tasks with more successors first)
    3. Position in the precedence graph (earlier tasks first)
    4. Weighted combination to balance station utilization
    
    The idea is to prioritize tasks that:
    - Are long (to reduce total stations)
    - Have many successors (critical path tasks)
    - Can be placed early to unlock more flexibility
    """
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    n = len(times)
    
    # Build successor lists and counts
    successors = [[] for _ in range(n)]
    succ_counts = np.zeros(n, dtype=int)
    for i in range(n):
        preds = predecessors[i]
        for p in preds:
            successors[p].append(i)
            succ_counts[p] += 1
    
    # Compute in-degree for topological sort
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for p in predecessors[i]:
            in_degree[i] += 1
    
    # Topological sort using Kahn's algorithm
    from collections import deque
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    in_deg = in_degree.copy()
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for j in successors[node]:
            in_deg[j] -= 1
            if in_deg[j] == 0:
                queue.append(j)
    
    # Compute longest path from each task to the end
    # longest_path[i] = max time path starting at i and going forward
    longest_path = np.zeros(n, dtype=float)
    # Process in reverse topological order
    for node in reversed(topo_order):
        max_succ_path = 0.0
        for j in successors[node]:
            max_succ_path = max(max_succ_path, longest_path[j])
        longest_path[node] = times[node] + max_succ_path
    
    # Compute position in topological order (earlier is better)
    pos_in_topo = np.zeros(n, dtype=float)
    for idx, node in enumerate(topo_order):
        pos_in_topo[node] = idx
    
    # Normalize each component
    time_norm = times / (times.max() + 1e-9)
    succ_norm = succ_counts / (succ_counts.max() + 1e-9)
    lp_norm = longest_path / (longest_path.max() + 1e-9)
    pos_norm = 1.0 - (pos_in_topo / (n + 1e-9))  # Earlier is higher
    
    # Weighted combination
    # Emphasize task time and longest path (critical path)
    score = (0.5 * time_norm + 
             0.3 * lp_norm + 
             0.1 * succ_norm + 
             0.1 * pos_norm)
    
    return score
# EVOLVE-BLOCK-END
