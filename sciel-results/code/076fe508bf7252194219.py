"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(instance):
    """
    Assign a priority score to each task for assembly line balancing.
    
    The strategy combines:
    1. Task duration (longer tasks first)
    2. Number of transitive successors (tasks that unblock many others first)
    3. A small tie-breaker based on task index for stability
    
    The evaluator fills stations with the highest-priority eligible task that fits
    within the cycle time. We want to minimize the number of stations used.
    
    Key insight: tasks with many transitive successors are "bottlenecks" — scheduling
    them early makes their successors eligible sooner, improving packing.
    """
    t0 = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill
    
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    
    # Get predecessors (list of lists or similar structure)
    preds = instance.get("predecessors", None)
    if preds is None:
        # No predecessors, just use time-based priority
        return times.copy()
    
    # Build adjacency: for each task, its direct successors
    # predecessors[i] = list of tasks that must come before i
    # So if j is in predecessors[i], then j -> i (j is predecessor of i)
    # We need transitive successors of each task.
    
    # Build direct successor graph
    # succ[j] = list of i such that j is a direct predecessor of i
    succ = [[] for _ in range(n)]
    for i in range(n):
        if preds is not None:
            pred_list = preds[i]
            if pred_list is not None:
                for j in pred_list:
                    succ[j].append(i)
    
    # Compute transitive successors count for each task using BFS/DFS
    # For each node, count how many nodes are reachable from it
    trans_successors = [0] * n
    
    for start in range(n):
        if time.time() > DEADLINE:
            # Fallback: just use time
            return times.copy()
        # BFS from start to count reachable nodes
        visited = [False] * n
        visited[start] = True
        stack = [start]
        count = 0
        while stack:
            node = stack.pop()
            for neighbor in succ[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    count += 1
                    stack.append(neighbor)
        trans_successors[start] = count
    
    # Also compute "critical path length" from each task to the end
    # This is the longest path from task i to any sink
    # DP: cpl[i] = times[i] + max(cpl[j] for j in successors of i), or times[i] if no successors
    # We need topological order. Let's compute in-degree and do topo sort.
    
    in_degree = [0] * n
    for i in range(n):
        if preds is not None and preds[i] is not None:
            in_degree[i] = len(preds[i])
    
    from collections import deque
    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)
    
    topo_order = []
    in_deg_copy = in_degree[:]
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for neighbor in succ[node]:
            in_deg_copy[neighbor] -= 1
            if in_deg_copy[neighbor] == 0:
                queue.append(neighbor)
    
    # If topo_order doesn't have all nodes, there's a cycle; handle gracefully
    if len(topo_order) < n:
        # Cycle detected, just use time
        return times.copy()
    
    # Compute longest path from each node to sink (reverse topo order)
    cpl = [0.0] * n
    # Process in reverse topological order
    for node in reversed(topo_order):
        max_succ_cpl = 0.0
        for neighbor in succ[node]:
            if cpl[neighbor] > max_succ_cpl:
                max_succ_cpl = cpl[neighbor]
        cpl[node] = times[node] + max_succ_cpl
    
    # Normalize each component to [0, 1] range for combination
    max_time = np.max(times) if n > 0 else 1.0
    max_ts = max(trans_successors) if n > 0 else 1
    max_cpl = max(cpl) if n > 0 else 1.0
    
    if max_time < 1e-9:
        max_time = 1.0
    if max_ts < 1e-9:
        max_ts = 1.0
    if max_cpl < 1e-9:
        max_cpl = 1.0
    
    time_norm = times / max_time
    ts_norm = np.array(trans_successors, dtype=float) / max_ts
    cpl_norm = np.array(cpl, dtype=float) / max_cpl
    
    # Combine: weight transitive successors heavily, then CPL, then time
    # Tasks with many successors should be scheduled first to unlock others
    # Tasks with long downstream paths (CPL) are critical
    # Longer tasks take more time so they should be placed early
    
    # Weighted combination
    w_ts = 2.0   # transitive successors
    w_cpl = 1.5  # critical path length
    w_time = 1.0 # task duration
    
    scores = w_ts * ts_norm + w_cpl * cpl_norm + w_time * time_norm
    
    # Add tiny tie-breaker to ensure deterministic ordering
    # Use negative index so earlier indices get slightly higher priority on ties
    scores = scores - np.arange(n) * 1e-9
    
    return scores
# EVOLVE-BLOCK-END
