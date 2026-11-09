"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(instance):
    """
    Score each assembly task for greedy station filling.
    Higher priority = should be placed earlier in the sequence.
    
    Uses a composite score based on:
    1. Task duration (longer tasks are harder to fit)
    2. Downstream critical path length (tasks with more downstream work 
       are more critical bottlenecks)
    3. Number of successors (fewer successors = more constrained, should go earlier)
    """
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    cycle_time = instance.get("cycle_time", 40)
    
    # Get predecessor relationships
    # predecessors: list of lists, predecessors[i] = list of task indices that must come before i
    preds = instance.get("predecessors", [list(range(i)) for i in range(n)])
    
    # Build successor lists
    succs = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            succs[p].append(i)
    
    # Compute downstream longest path (longest path from each node to any sink)
    # Use topological order (reverse)
    # First, compute in-degrees for topological sort
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for p in preds[i]:
            in_degree[i] += 1
    
    # Topological sort using Kahn's algorithm
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
        for s in succs[node]:
            in_deg_copy[s] -= 1
            if in_deg_copy[s] == 0:
                queue.append(s)
    
    # If topological sort didn't include all nodes, add remaining (cycle handling)
    if len(topo_order) < n:
        remaining = set(range(n)) - set(topo_order)
        topo_order.extend(sorted(remaining))
    
    # Compute downstream longest path in reverse topological order
    # downstream[i] = times[i] + max(downstream[s] for s in succs[i]) if succs else times[i]
    downstream = np.zeros(n, dtype=float)
    # Process in reverse topological order
    for i in reversed(topo_order):
        if succs[i]:
            max_down = max(downstream[s] for s in succs[i])
            downstream[i] = times[i] + max_down
        else:
            downstream[i] = times[i]
    
    # Compute upstream longest path (longest path from any source to each node)
    upstream = np.zeros(n, dtype=float)
    for i in topo_order:
        if preds[i]:
            max_up = max(upstream[p] for p in preds[i])
            upstream[i] = max_up + times[i]
        else:
            upstream[i] = times[i]
    
    # Count successors
    succ_count = np.array([len(succs[i]) for i in range(n)], dtype=float)
    
    # Composite priority score:
    # - Higher downstream work = more critical, should be scheduled earlier
    # - Longer duration = harder to fit
    # - Fewer successors = more constrained
    # 
    # Normalize components to [0, 1] range for combination
    max_down = np.max(downstream) if n > 0 else 1
    max_up = np.max(upstream) if n > 0 else 1
    max_time = np.max(times) if n > 0 else 1
    max_succ = np.max(succ_count) if n > 0 and np.max(succ_count) > 0 else 1
    
    # Downstream criticality (most important - tasks with lots of downstream work)
    down_norm = downstream / max_down if max_down > 0 else np.zeros(n)
    
    # Duration importance
    time_norm = times / max_time if max_time > 0 else np.zeros(n)
    
    # Constrainedness: tasks with fewer successors are more constrained
    # Invert so fewer successors = higher score
    if max_succ > 0:
        constrained = 1.0 - (succ_count / max_succ)
    else:
        constrained = np.ones(n)
    
    # Combine: weight downstream heavily, then duration, then constrainedness
    # The idea: prioritize tasks that are bottlenecks for the remaining work
    priority_score = 0.5 * down_norm + 0.3 * time_norm + 0.2 * constrained
    
    return priority_score
# EVOLVE-BLOCK-END
