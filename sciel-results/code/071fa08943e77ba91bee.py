"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(instance):
    """
    Score each assembly task using a combination of:
    1. Longest downstream path (critical path length from task to end)
    2. Task duration
    3. Number of immediate successors (fan-out)
    
    This 'Critical Path Density' approach prioritizes tasks that are
    bottlenecks on the critical path and also high-fanout branching points,
    which should reduce downstream fragmentation and station count.
    """
    start_time = time.time()
    DEADLINE = 120.0  # Safety margin before 150s kill
    
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    cycle_time = instance.get("cycle_time", 40)
    predecessors = instance.get("predecessors", [])
    
    # Build adjacency: successors[i] = list of tasks that depend on i
    successors = [[] for _ in range(n)]
    in_degree = np.zeros(n, dtype=int)
    
    for i, preds in enumerate(predecessors):
        for p in preds:
            successors[p].append(i)
            in_degree[i] += 1
    
    # Compute longest path from each task to any sink (downstream critical path length)
    # Use topological order via Kahn's algorithm
    from collections import deque
    
    # Topological sort
    topo_order = []
    in_deg = in_degree.copy()
    queue = deque([i for i in range(n) if in_deg[i] == 0])
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            in_deg[succ] -= 1
            if in_deg[succ] == 0:
                queue.append(succ)
    
    # If not all nodes are in topo_order, there's a cycle; fall back to simple ordering
    if len(topo_order) < n:
        # Fallback: just use task duration as priority
        return times.copy()
    
    # Compute longest downstream path for each node (reverse topological order)
    longest_downstream = np.zeros(n, dtype=float)
    for node in reversed(topo_order):
        max_succ = 0.0
        for succ in successors[node]:
            if longest_downstream[succ] > max_succ:
                max_succ = longest_downstream[succ]
        longest_downstream[node] = times[node] + max_succ
    
    # Compute number of immediate successors (fan-out)
    fan_out = np.array([len(successors[i]) for i in range(n)], dtype=float)
    
    # Compute number of immediate predecessors (fan-in)
    fan_in = np.array([len(predecessors[i]) if i < len(predecessors) else 0 for i in range(n)], dtype=float)
    
    # Check time budget
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 5:
        # Return simple priority if we're running out of time
        return longest_downstream.copy()
    
    # Priority score: combine critical path length, task duration, and fan-out
    # Higher critical path length = more important to schedule early
    # Higher fan-out = more branching points, scheduling early reduces fragmentation
    # We want to prioritize tasks that are on the critical path AND have high fan-out
    
    # Normalize components
    max_ld = np.max(longest_downstream) if n > 0 else 1.0
    max_time = np.max(times) if n > 0 else 1.0
    max_fanout = np.max(fan_out) if n > 0 else 1.0
    
    # Avoid division by zero
    if max_ld == 0: max_ld = 1.0
    if max_time == 0: max_time = 1.0
    if max_fanout == 0: max_fanout = 1.0
    
    norm_ld = longest_downstream / max_ld
    norm_time = times / max_time
    norm_fanout = fan_out / max_fanout
    
    # Combined score: weighted sum with emphasis on critical path and fan-out
    # Weighting: critical path length is most important, then task duration, then fan-out
    score = 1.0 * norm_ld + 0.5 * norm_time + 0.3 * norm_fanout
    
    return score
# EVOLVE-BLOCK-END
