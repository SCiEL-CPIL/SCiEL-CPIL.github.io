"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_positional_weight(times, successors):
    """
    Compute positional weight for each task: sum of its own time plus all successor times.
    This captures the total volume of work unlocked by completing the task.
    """
    n = len(times)
    # Topological order (Kahn's algorithm)
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for j in successors[i]:
            in_degree[j] += 1

    from collections import deque
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    # Process in reverse topological order to compute positional weights
    pw = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        pw[u] = times[u] + sum(pw[v] for v in successors[u])

    return pw


def _compute_rank(times, predecessors, successors):
    """
    Compute a rank for each task based on the longest path from the task to any successor.
    This is the standard rank function used in many ALB heuristics.
    """
    n = len(times)
    # Topological order
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for j in successors[i]:
            in_degree[j] += 1

    from collections import deque
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    # Process in reverse topological order
    rank = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        if successors[u]:
            rank[u] = times[u] + max(rank[v] for v in successors[u])
        else:
            rank[u] = times[u]

    return rank


def _compute_successor_count(successors):
    """Count the number of direct successors for each task."""
    n = len(successors)
    return np.array([len(succ) for succ in successors], dtype=float)


def priority(instance):
    """
    Score each assembly task for the greedy station-filling algorithm.
    Higher score means higher priority.
    
    Strategy: Use positional weight (total downstream work) as the dominant signal,
    with rank and successor count as secondary tie-breakers.
    """
    t0 = time.time()
    DEADLINE = 140.0  # Safety margin before 150s limit

    times = np.asarray(instance["times"], dtype=float)
    cycle_time = instance.get("cycle_time", 40)
    
    # Build predecessor and successor lists
    n = len(times)
    predecessors = instance.get("predecessors", [])
    
    # Convert predecessors to a list of lists if needed
    if isinstance(predecessors, dict):
        pred_list = [predecessors.get(i, []) for i in range(n)]
    else:
        pred_list = list(predecessors)
    
    # Build successor list
    successors = [[] for _ in range(n)]
    for i, preds in enumerate(pred_list):
        for p in preds:
            successors[p].append(i)
    
    # Compute positional weight (dominant signal)
    pw = _compute_positional_weight(times, successors)
    
    # Compute rank (secondary signal)
    rank = _compute_rank(times, predecessors, successors)
    
    # Compute successor count (tertiary signal)
    succ_count = _compute_successor_count(successors)
    
    # Combine signals: positional weight is dominant, then rank, then successor count
    # Normalize each component to [0, 1] range for combination
    pw_min, pw_max = pw.min(), pw.max()
    rank_min, rank_max = rank.min(), rank.max()
    sc_min, sc_max = succ_count.min(), succ_count.max()
    
    pw_norm = (pw - pw_min) / (pw_max - pw_min + 1e-10)
    rank_norm = (rank - rank_min) / (rank_max - rank_min + 1e-10)
    sc_norm = (succ_count - sc_min) / (sc_max - sc_min + 1e-10)
    
    # Weighted combination: positional weight 70%, rank 20%, successor count 10%
    score = 0.7 * pw_norm + 0.2 * rank_norm + 0.1 * sc_norm
    
    # Add a small boost for tasks with no successors (sinks) to break ties
    # Sinks are important because they don't unlock further work
    sink_boost = np.where(succ_count == 0, 0.05, 0.0)
    score += sink_boost
    
    # Check deadline
    if time.time() > DEADLINE:
        return score
    
    return score
# EVOLVE-BLOCK-END
