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
    """Compute longest downstream path, successor counts, and transitive successors."""
    succ = _build_successor_adjacency(n, predecessors)
    
    # Topological order
    in_degree = [len(preds) for preds in predecessors]
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    top_order = []
    while queue:
        node = queue.popleft()
        top_order.append(node)
        for s in succ[node]:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)
    
    # Longest path downstream (including own time)
    longest_down = [0.0] * n
    for node in reversed(top_order):
        max_down = 0.0
        for s in succ[node]:
            max_down = max(max_down, longest_down[s])
        longest_down[node] = times[node] + max_down
    
    # Direct successor count
    succ_count = [len(succ[i]) for i in range(n)]
    
    # Transitive successor count
    trans_succ = [0] * n
    for node in reversed(top_order):
        for s in succ[node]:
            trans_succ[node] += 1 + trans_succ[s]
    
    # Longest path upstream (including own time)
    longest_up = [0.0] * n
    for node in top_order:
        max_up = 0.0
        for p in predecessors[node]:
            max_up = max(max_up, longest_up[p])
        longest_up[node] = times[node] + max_up
    
    # Predecessor count
    pred_count = [len(predecessors[i]) for i in range(n)]
    
    return (np.array(longest_down, dtype=float),
            np.array(succ_count, dtype=float),
            np.array(trans_succ, dtype=float),
            np.array(longest_up, dtype=float),
            np.array(pred_count, dtype=float))


def _simulate_greedy(n, predecessors, cycle_time, priority_vec, times):
    """Simulate greedy station assignment and return number of stations."""
    remaining = set(range(n))
    n_stations = 0
    
    while remaining:
        n_stations += 1
        station_load = 0.0
        
        # Find eligible tasks that fit
        eligible = []
        for task in remaining:
            if all(p not in remaining for p in predecessors[task]):
                if station_load + times[task] <= cycle_time:
                    eligible.append(task)
        
        if not eligible:
            break
        
        # Pick highest priority eligible task
        best_task = max(eligible, key=lambda t: priority_vec[t])
        remaining.discard(best_task)
        station_load += times[best_task]
    
    return n_stations


def _compute_rank_priority(n, predecessors, times, cycle_time):
    """Compute priority based on rank position in a topological ordering weighted by time."""
    succ = _build_successor_adjacency(n, predecessors)
    
    # Topological order
    in_degree = [len(preds) for preds in predecessors]
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    top_order = []
    while queue:
        node = queue.popleft()
        top_order.append(node)
        for s in succ[node]:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)
    
    # Rank: earlier in topological order = higher priority
    rank = np.zeros(n)
    for idx, node in enumerate(top_order):
        rank[node] = idx
    
    # Normalize rank (invert so earlier = higher)
    max_rank = np.max(rank)
    if max_rank > 0:
        rank_norm = 1.0 - rank / max_rank
    else:
        rank_norm = np.ones(n)
    
    return rank_norm


def priority(instance):
    """Score each assembly task for priority-based station assignment."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    try:
        times = np.asarray(instance["times"], dtype=float)
        predecessors = instance["predecessors"]
        cycle_time = instance["cycle_time"]
        n = len(times)
        
        # Compute features
        longest_down, succ_count, trans_succ, longest_up, pred_count = _compute_downstream_metrics(n, predecessors, times)
        
        # Normalize features
        def norm(x):
            mx = np.max(x)
            if mx > 0:
                return x / mx
            return np.zeros_like(x)
        
        ld_norm = norm(longest_down)
        sc_norm = norm(succ_count)
        ts_norm = norm(trans_succ)
        lu_norm = norm(longest_up)
        pc_norm = norm(pred_count)
        t_norm = norm(times)
        
        # Rank-based priority
        rank_norm = _compute_rank_priority(n, predecessors, times, cycle_time)
        
        # Try several weight combinations and pick the best via simulation
        # Weights: (time, longest_down, succ_count, trans_succ, longest_up, pred_count, rank)
        weight_configs = [
            (0.4, 0.3, 0.1, 0.1, 0.0, 0.0, 0.1),
            (0.3, 0.3, 0.2, 0.1, 0.0, 0.0, 0.1),
            (0.3, 0.2, 0.2, 0.1, 0.0, 0.0, 0.2),
            (0.2, 0.3, 0.2, 0.1, 0.0, 0.0, 0.2),
            (0.3, 0.3, 0.1, 0.1, 0.0, 0.0, 0.2),
            (0.4, 0.2, 0.2, 0.1, 0.0, 0.0, 0.1),
            (0.3, 0.4, 0.1, 0.1, 0.0, 0.0, 0.1),
            (0.2, 0.4, 0.2, 0.1, 0.0, 0.0, 0.1),
            (0.3, 0.3, 0.3, 0.1, 0.0, 0.0, 0.0),
            (0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0),
            (0.5, 0.2, 0.1, 0.1, 0.0, 0.0, 0.1),
            (0.3, 0.2, 0.2, 0.2, 0.0, 0.0, 0.1),
            (0.2, 0.3, 0.3, 0.1, 0.0, 0.0, 0.1),
            (0.3, 0.3, 0.2, 0.2, 0.0, 0.0, 0.0),
            (0.4, 0.2, 0.3, 0.1, 0.0, 0.0, 0.0),
            (0.2, 0.4, 0.2, 0.2, 0.0, 0.0, 0.0),
            (0.3, 0.3, 0.1, 0.2, 0.0, 0.0, 0.1),
            (0.2, 0.3, 0.2, 0.2, 0.0, 0.0, 0.1),
            (0.4, 0.3, 0.1, 0.1, 0.0, 0.0, 0.1),
            (0.5, 0.3, 0.1, 0.1, 0.0, 0.0, 0.0),
        ]
        
        best_priority = None
        best_stations = float('inf')
        
        for (wt, wld, wsc, wts, wlu, wpc, wrk) in weight_configs:
            if time.time() > deadline - 5.0:
                break
            
            priority_vec = (wt * t_norm + wld * ld_norm + wsc * sc_norm + 
                          wts * ts_norm + wlu * lu_norm + wpc * pc_norm + 
                          wrk * rank_norm) * np.max(times)
            
            # Simulate to get station count
            n_stations = _simulate_greedy(n, predecessors, cycle_time, priority_vec, times)
            
            if n_stations < best_stations:
                best_stations = n_stations
                best_priority = priority_vec
        
        if best_priority is None:
            # Fallback
            best_priority = 0.5 * times + 0.3 * ld_norm * np.max(times) + 0.2 * sc_norm * np.max(times)
        
        return best_priority
    
    except Exception:
        # Fallback to simple time-based priority
        return np.asarray(instance["times"], dtype=float)
# EVOLVE-BLOCK-END
