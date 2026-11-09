"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _build_adjacency(predecessors, n):
    """Return list of lists: adj[i] = tasks that depend on i."""
    adj = [[] for _ in range(n)]
    if predecessors is None:
        return adj
    for i, preds in enumerate(predecessors):
        if preds:
            for p in preds:
                adj[p].append(i)
    return adj

def _compute_rank(instance):
    """
    Compute task priorities using a combination of:
    1. Task duration
    2. Successor duration (sum of immediate successors' durations)
    3. Position in the DAG (longest path from task to any sink)
    4. Number of successors (tasks with more successors are more critical)
    
    Returns an array of priority scores (higher = should be scheduled earlier).
    """
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    predecessors = instance.get("predecessors", None)
    
    if predecessors is None:
        predecessors = [[] for _ in range(n)]
    
    # Build successor lists
    successors = [[] for _ in range(n)]
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
            in_degree[i] += 1
    
    # Topological sort (Kahn's algorithm)
    from collections import deque
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    indeg = in_degree.copy()
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            indeg[succ] -= 1
            if indeg[succ] == 0:
                queue.append(succ)
    
    # If topo sort didn't include all nodes (cycle), fall back to original order
    if len(topo_order) != n:
        topo_order = list(range(n))
    
    # Compute longest path from each node to any sink (bottom rank)
    # Process in reverse topological order
    bottom_rank = np.zeros(n, dtype=float)
    for node in reversed(topo_order):
        if successors[node]:
            max_succ_rank = max(bottom_rank[s] for s in successors[node])
            bottom_rank[node] = times[node] + max_succ_rank
        else:
            bottom_rank[node] = times[node]
    
    # Compute number of successors (direct and indirect)
    succ_count = np.zeros(n, dtype=int)
    for node in reversed(topo_order):
        if successors[node]:
            succ_count[node] = 1 + max(succ_count[s] for s in successors[node])
        # else stays 0
    
    # Compute topological depth (longest path from source to node)
    top_rank = np.zeros(n, dtype=float)
    for node in topo_order:
        if predecessors[node]:
            max_pred_rank = max(top_rank[p] for p in predecessors[node])
            top_rank[node] = max_pred_rank + times[node]
        else:
            top_rank[node] = times[node]
    
    # Normalize each component
    def normalize(arr):
        arr = arr.astype(float)
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.ones_like(arr)
        return (arr - mn) / (mx - mn)
    
    # Weighted combination:
    # - bottom_rank (most important: tasks that lead to long chains)
    # - task duration
    # - successor count (bottleneck tasks)
    # - top_rank (slight preference for later-stage tasks)
    
    w_bottom = 0.45
    w_time = 0.30
    w_succ = 0.15
    w_top = 0.10
    
    priority_scores = (
        w_bottom * normalize(bottom_rank) +
        w_time * normalize(times) +
        w_succ * normalize(succ_count.astype(float)) +
        w_top * normalize(top_rank)
    )
    
    return priority_scores

def priority(instance):
    """
    Score each assembly task for the greedy scheduler.
    Returns one score per task; higher score = higher priority.
    """
    start_time = time.time()
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    
    try:
        scores = _compute_rank(instance)
    except Exception:
        scores = times.copy()
    
    # Ensure we're within time budget
    elapsed = time.time() - start_time
    if elapsed > 110:
        return times.copy()
    
    return scores
# EVOLVE-BLOCK-END
