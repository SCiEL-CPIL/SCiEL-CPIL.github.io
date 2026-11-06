"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _build_graph(n, preds):
    """Build adjacency and in-degree structures."""
    succ = [[] for _ in range(n)]
    indeg = [0] * n
    for i, plist in enumerate(preds):
        for p in plist:
            succ[p].append(i)
            indeg[i] += 1
    return succ, indeg


def _topo_order(n, succ, indeg):
    """Kahn's algorithm to get a topological order."""
    from collections import deque
    q = deque([i for i in range(n) if indeg[i] == 0])
    order = []
    indeg_copy = list(indeg)
    while q:
        u = q.popleft()
        order.append(u)
        for v in succ[u]:
            indeg_copy[v] -= 1
            if indeg_copy[v] == 0:
                q.append(v)
    return order


def _compute_rank(n, times, preds, cycle_time):
    """
    Compute priority scores using a combined heuristic:
    1. Longest path from task to any successor (backward rank)
    2. Number of immediate successors (successor count)
    3. Bottleneck slack: (cycle_time - time_i) / (1 + num_successors)
    
    We want tasks that are:
    - Long (hard to fit)
    - Have many successors (unblock more tasks)
    - Consume a large fraction of cycle time (bottleneck)
    
    The priority is designed so that higher score = higher priority.
    """
    n = len(times)
    times = np.asarray(times, dtype=float)
    
    # Build graph
    succ, indeg = _build_graph(n, preds)
    order = _topo_order(n, succ, indeg)
    
    # Backward rank: longest path from node to any sink (including itself)
    # rank[i] = time[i] + max(rank[s] for s in successors(i))
    rank = np.zeros(n, dtype=float)
    # Process in reverse topological order
    for i in reversed(order):
        max_succ_rank = 0.0
        for s in succ[i]:
            if rank[s] > max_succ_rank:
                max_succ_rank = rank[s]
        rank[i] = times[i] + max_succ_rank
    
    # Forward rank: longest path from any source to node (including itself)
    f_rank = np.zeros(n, dtype=float)
    for i in order:
        max_pred_rank = 0.0
        for p in preds[i]:
            if f_rank[p] > max_pred_rank:
                max_pred_rank = f_rank[p]
        f_rank[i] = times[i] + max_pred_rank
    
    # Successor count
    succ_count = np.array([len(succ[i]) for i in range(n)], dtype=float)
    
    # Compute priority components
    # 1. Normalized backward rank (higher = more critical path length ahead)
    max_rank = rank.max() if rank.max() > 0 else 1.0
    norm_rank = rank / max_rank
    
    # 2. Successor bonus: tasks with more successors are more valuable
    # Normalize successor count
    max_succ = succ_count.max() if succ_count.max() > 0 else 1.0
    norm_succ = succ_count / max_succ
    
    # 3. Bottleneck slack: smaller (cycle_time - time_i) means tighter fit
    # We want to prioritize tasks that are close to filling the cycle time
    slack = cycle_time - times
    # Inverse slack: higher when slack is small (task is large relative to cycle)
    # Use 1/(1+slack) to avoid division by zero
    inv_slack = 1.0 / (1.0 + slack)
    max_inv_slack = inv_slack.max() if inv_slack.max() > 0 else 1.0
    norm_inv_slack = inv_slack / max_inv_slack
    
    # 4. Critical position: tasks that are early in the DAG but have long paths ahead
    # This captures tasks that must be placed early to allow successors to fit
    
    # Combined priority: weighted sum
    # Emphasize: backward rank (critical path), successor count (unblocking), bottleneck
    priority = (
        1.0 * norm_rank +
        0.5 * norm_succ +
        0.8 * norm_inv_slack +
        0.3 * (times / (times.max() if times.max() > 0 else 1.0))
    )
    
    return priority


def priority(instance):
    """
    Score each assembly task for the greedy station-filling algorithm.
    Higher score = higher priority.
    """
    start_time = time.time()
    
    times = instance["times"]
    preds = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    
    n = len(times)
    if n == 0:
        return np.array([])
    
    # Compute the combined heuristic priority
    scores = _compute_rank(n, times, preds, cycle_time)
    
    # Safety: ensure we don't exceed time budget
    elapsed = time.time() - start_time
    if elapsed > 140.0:
        # Fallback to simple longest-task-first if we're running out of time
        return np.asarray(times, dtype=float)
    
    return scores
# EVOLVE-BLOCK-END
