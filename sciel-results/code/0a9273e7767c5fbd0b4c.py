"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque


def _build_successor_adjacency(n, predecessors):
    """Build successor adjacency list from predecessors."""
    succ = [[] for _ in range(n)]
    for j, preds in enumerate(predecessors):
        for i in preds:
            succ[i].append(j)
    return succ


def _compute_downstream_metrics(n, predecessors, times):
    """
    Compute:
    - longest_down[i]: longest path length from i to any sink (including own time)
    - trans_succ_count[i]: number of transitive successors
    - weighted_downstream[i]: sum of times of all transitive successors
    - rank[i]: topological depth (longest chain of successors, count of nodes)
    """
    succ = _build_successor_adjacency(n, predecessors)
    
    # Topological order via Kahn's algorithm
    in_degree = [len(preds) for preds in predecessors]
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    top_order = []
    while queue:
        node = queue.popleft()
        top_order.append(node)
        for j in range(n):
            if node in predecessors[j]:
                in_degree[j] -= 1
                if in_degree[j] == 0:
                    queue.append(j)
    
    # Process in reverse topological order
    longest_down = np.zeros(n, dtype=float)
    trans_succ_count = np.zeros(n, dtype=int)
    weighted_downstream = np.zeros(n, dtype=float)
    rank = np.zeros(n, dtype=float)
    
    for node in reversed(top_order):
        max_down = 0.0
        max_rank = 0.0
        ts_count = 0
        wd = 0.0
        for s in succ[node]:
            max_down = max(max_down, longest_down[s])
            max_rank = max(max_rank, rank[s])
            ts_count += 1 + trans_succ_count[s]
            wd += times[s] + weighted_downstream[s]
        longest_down[node] = times[node] + max_down
        rank[node] = 1.0 + max_rank
        trans_succ_count[node] = ts_count
        weighted_downstream[node] = wd
    
    return (longest_down, 
            np.array(trans_succ_count, dtype=float), 
            weighted_downstream,
            rank)


def _simulate_greedy(n, predecessors, cycle_time, priority_vec, times):
    """Simulate greedy station assignment and return number of stations."""
    assigned = [False] * n
    remaining = set(range(n))
    n_stations = 0
    
    while remaining:
        n_stations += 1
        station_load = 0.0
        # Find eligible tasks (all predecessors assigned) that fit
        eligible = []
        for task in remaining:
            if all(p not in remaining for p in predecessors[task]):
                if station_load + times[task] <= cycle_time + 1e-9:
                    eligible.append(task)
        
        if not eligible:
            # No task fits; this shouldn't happen if cycle_time >= max time
            break
        
        # Pick highest priority eligible task
        best_task = max(eligible, key=lambda t: priority_vec[t])
        remaining.discard(best_task)
        assigned[best_task] = True
        station_load += times[task]
    
    return n_stations


def _normalize(x):
    mx = np.max(x)
    if mx > 0:
        return x / mx
    return np.zeros_like(x)


def priority(instance):
    """
    Score tasks using a weighted combination of processing time, longest downstream
    path, transitive successor count, weighted downstream time, and topological rank.
    Uses internal simulation to select the best weight configuration.
    """
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill

    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    if n == 0:
        return np.array([], dtype=float)

    # Compute all downstream features
    longest_down, trans_succ, weighted_down, rank = _compute_downstream_metrics(n, preds, times)
    
    # Normalize all features to [0, 1]
    t_norm = _normalize(times)
    ld_norm = _normalize(longest_down)
    ts_norm = _normalize(trans_succ)
    wd_norm = _normalize(weighted_down)
    rk_norm = _normalize(rank)
    
    scale = np.max(times) if np.max(times) > 0 else 1.0
    
    # Weight configurations to try: (w_time, w_longest_down, w_trans_succ, w_weighted_down, w_rank)
    weight_configs = [
        (0.40, 0.25, 0.15, 0.10, 0.10),
        (0.35, 0.25, 0.15, 0.15, 0.10),
        (0.30, 0.30, 0.15, 0.10, 0.15),
        (0.35, 0.30, 0.15, 0.10, 0.10),
        (0.40, 0.20, 0.20, 0.10, 0.10),
        (0.30, 0.25, 0.20, 0.15, 0.10),
        (0.35, 0.20, 0.20, 0.15, 0.10),
        (0.40, 0.25, 0.10, 0.10, 0.15),
        (0.30, 0.30, 0.20, 0.10, 0.10),
        (0.45, 0.20, 0.15, 0.10, 0.10),
        (0.35, 0.30, 0.15, 0.15, 0.05),
        (0.40, 0.20, 0.15, 0.15, 0.10),
        (0.30, 0.25, 0.20, 0.15, 0.10),
        (0.35, 0.25, 0.20, 0.10, 0.10),
        (0.25, 0.30, 0.20, 0.15, 0.10),
        (0.30, 0.30, 0.15, 0.15, 0.10),
        (0.40, 0.20, 0.10, 0.15, 0.15),
        (0.35, 0.25, 0.10, 0.15, 0.15),
    ]
    
    best_priority = None
    best_stations = float('inf')
    
    for (wt, wld, wts, wwd, wrk) in weight_configs:
        if time.time() - start_time > DEADLINE - 10.0:
            break
            
        priority_vec = (wt * t_norm + wld * ld_norm + wts * ts_norm + wwd * wd_norm + wrk * rk_norm) * scale
        
        # Simulate to get station count
        n_stations = _simulate_greedy(n, preds, cycle_time, priority_vec, times)
        
        if n_stations < best_stations:
            best_stations = n_stations
            best_priority = priority_vec
    
    if best_priority is None:
        # Fallback: simple time-based priority
        best_priority = times.copy()
    
    return best_priority
# EVOLVE-BLOCK-END
