"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def priority(instance):
    """
    Returns a priority score for each activity.
    The serial schedule-generation scheme starts the highest-priority eligible activity.
    Lower makespan is better.
    
    Strategy: 
    1. Compute topological order.
    2. Forward pass: earliest start times (ES) and earliest finish times (EF) ignoring resources.
    3. Backward pass: latest start times (LS) and latest finish times (LF) ignoring resources.
    4. Compute slack for each activity.
    5. Priority = -slack (activities with less slack are higher priority).
    6. Tie-breaking: higher duration first, then higher resource demand.
    
    This is a standard critical path heuristic which tends to perform well for RCPSP.
    """
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 110s limit, leaving 10s margin for evaluator overhead
    
    n = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    predecessors = instance.get("predecessors", None)
    
    if predecessors is None:
        # No precedence constraints, just prioritize by duration
        return -durations
    
    # Build adjacency list
    # predecessors[i] is a list of indices of activities that must finish before i starts
    # We need to build successor list for forward pass and use predecessors for backward pass
    
    # Ensure predecessors is a list of lists
    if isinstance(predecessors, np.ndarray):
        if predecessors.dtype == object:
            preds = [list(p) for p in predecessors]
        else:
            # If it's a 2D array or similar, convert appropriately
            preds = [list(p) for p in predecessors]
    elif isinstance(predecessors, list):
        preds = [list(p) if isinstance(p, (list, tuple, np.ndarray)) else ([p] if p is not None else []) for p in predecessors]
    else:
        preds = [[] for _ in range(n)]
    
    # Build successor list
    succs = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            if p >= 0 and p < n:
                succs[p].append(i)
    
    # Topological sort (Kahn's algorithm)
    in_degree = [0] * n
    for i in range(n):
        for p in preds[i]:
            if p >= 0 and p < n:
                in_degree[i] += 1
    
    from collections import deque
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in succs[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # If not all nodes are in topo_order, there's a cycle; fall back to duration-based
    if len(topo_order) != n:
        return -durations
    
    # Forward pass: ES and EF
    ES = np.zeros(n, dtype=float)
    EF = np.zeros(n, dtype=float)
    for u in topo_order:
        if preds[u]:
            ES[u] = max(EF[p] for p in preds[u] if p >= 0 and p < n)
        EF[u] = ES[u] + durations[u]
    
    # Project horizon
    makespan_no_res = max(EF) if n > 0 else 0
    
    # Backward pass: LF and LS
    LF = np.full(n, makespan_no_res, dtype=float)
    LS = np.full(n, makespan_no_res, dtype=float)
    for u in reversed(topo_order):
        if succs[u]:
            LF[u] = min(LS[s] for s in succs[u])
        else:
            LF[u] = makespan_no_res
        LS[u] = LF[u] - durations[u]
    
    # Slack
    slack = LS - ES
    
    # Priority: negative slack (lower slack = higher priority)
    # Tie-break: higher duration, then higher max demand
    priority_scores = -slack
    
    # Add small tie-breakers
    if demands is not None:
        try:
            demands_arr = np.asarray(demands, dtype=float)
            if demands_arr.ndim == 2:
                max_demand = demands_arr.max(axis=1)
            else:
                max_demand = demands_arr
            # Normalize to avoid scale issues
            if max_demand.max() > 0:
                priority_scores = priority_scores - 0.001 * max_demand
        except:
            pass
    
    # Final tie-break by duration
    priority_scores = priority_scores - 0.0001 * durations
    
    return priority_scores
# EVOLVE-BLOCK-END
