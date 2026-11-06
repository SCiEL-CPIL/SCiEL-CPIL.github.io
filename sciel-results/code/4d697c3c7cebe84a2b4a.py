"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque


def _compute_resource_contention(times, predecessors, cycle_time):
    """
    Compute resource contention metrics:
    - For each task, the sum of durations of all transitive predecessors (upstream time)
    - For each task, the sum of durations of all transitive successors (downstream time)
    - Earliest start time (longest path from sources)
    - Latest finish time (longest path to sinks)
    - Transitive predecessor and successor counts
    
    Returns all these metrics as numpy arrays.
    """
    n = len(times)
    # Build adjacency: pred -> successors
    succ = [[] for _ in range(n)]
    pred_list = []
    for i in range(n):
        preds = predecessors[i]
        pred_list.append(preds)
        for p in preds:
            succ[p].append(i)
    
    # Topological order via Kahn's algorithm
    in_degree = [len(pred_list[i]) for i in range(n)]
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo = []
    while queue:
        u = queue.popleft()
        topo.append(u)
        for v in succ[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # If topo doesn't include all nodes (cycle), handle gracefully
    if len(topo) < n:
        # Fallback: just use times
        return (np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n),
                np.zeros(n), np.zeros(n))
    
    # Earliest start: longest path from sources
    earliest = np.zeros(n, dtype=float)
    for u in topo:
        for v in succ[u]:
            val = earliest[u] + times[u]
            if val > earliest[v]:
                earliest[v] = val
    
    # Latest finish: longest path from task to sinks (reverse)
    # Build reverse adjacency
    pred_adj = [[] for _ in range(n)]
    for i in range(n):
        for p in pred_list[i]:
            pred_adj[i].append(p)
    
    latest_finish = np.zeros(n, dtype=float)
    for u in reversed(topo):
        for v in pred_adj[u]:
            val = latest_finish[u] + times[u]
            if val > latest_finish[v]:
                latest_finish[v] = val
    
    # Transitive predecessor time sum: sum of times of all transitive predecessors
    # trans_pred_time[i] = sum of times[j] for all j that are transitive predecessors of i
    trans_pred_time = np.zeros(n, dtype=float)
    trans_pred_count = np.zeros(n, dtype=int)
    for u in topo:
        for v in succ[u]:
            trans_pred_time[v] += times[u] + trans_pred_time[u]
            trans_pred_count[v] += 1 + trans_pred_count[u]
    
    # Transitive successor time sum: sum of times of all transitive successors
    # trans_succ_time[i] = sum of times[j] for all j that are transitive successors of i
    trans_succ_time = np.zeros(n, dtype=float)
    trans_succ_count = np.zeros(n, dtype=int)
    for u in reversed(topo):
        for v in succ[u]:
            trans_succ_time[u] += times[v] + trans_succ_time[v]
            trans_succ_count[u] += 1 + trans_succ_count[v]
    
    return (earliest, latest_finish, trans_pred_time, trans_succ_time,
            trans_pred_count, trans_succ_count)


def priority(instance):
    """
    Score each assembly task using resource contention metrics.
    
    Strategy: Each task's priority reflects the total time-weighted dependency
    structure around it. Tasks in dense, time-heavy subgraphs are prioritized
    because placing them well maximizes parallelism potential.
    
    Components:
      1. Upstream time density: normalized transitive predecessor time sum
      2. Downstream time density: normalized transitive successor time sum  
      3. Own duration: longer tasks are harder to pack
      4. Slack (cycle_time - earliest - own_time - downstream_time): less slack = more critical
    """
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = float(instance["cycle_time"])
    n = len(times)
    
    t0 = time.time()
    deadline = t0 + 10.0  # safety margin
    
    # Compute all metrics
    (earliest, latest_finish, trans_pred_time, trans_succ_time,
     trans_pred_count, trans_succ_count) = _compute_resource_contention(
        times, predecessors, cycle_time)
    
    # Check deadline
    if time.time() > deadline:
        return times.astype(float)
    
    # Normalize each component to [0, 1]
    def normalize(arr):
        mn, mx = float(arr.min()), float(arr.max())
        if mx > mn:
            return (arr - mn) / (mx - mn)
        return np.ones(n, dtype=float)
    
    # 1. Upstream time density: tasks with more upstream work are later in the chain
    #    Higher upstream time means the task is constrained by predecessors
    upstream_norm = normalize(trans_pred_time)
    
    # 2. Downstream time density: tasks with more downstream work are bottlenecks
    #    Higher downstream time means more work depends on this task
    downstream_norm = normalize(trans_succ_time)
    
    # 3. Own duration
    dur_norm = normalize(times)
    
    # 4. Slack: cycle_time - earliest_start - own_time - trans_succ_time
    #    Less slack = more critical (less flexibility in scheduling)
    slack = cycle_time - earliest - times - trans_succ_time
    # Clamp to non-negative for normalization
    slack_clamped = np.maximum(slack, 0.0)
    # Invert: less slack = higher priority
    slack_norm = 1.0 - normalize(slack_clamped)
    
    # 5. Total contention: sum of upstream + downstream + own time (total time in connected component)
    total_time = trans_pred_time + times + trans_succ_time
    total_norm = normalize(total_time)
    
    # Combine: emphasize downstream (bottleneck) and total contention,
    # with upstream and slack as secondary signals
    score = (0.30 * downstream_norm + 
             0.25 * total_norm + 
             0.20 * slack_norm + 
             0.15 * dur_norm + 
             0.10 * upstream_norm)
    
    # Check deadline
    if time.time() > deadline:
        return times.astype(float)
    
    return score
# EVOLVE-BLOCK-END
