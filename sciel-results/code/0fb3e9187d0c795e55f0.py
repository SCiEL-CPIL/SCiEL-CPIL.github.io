"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time
from collections import deque

def _simulate_stations(times, predecessors, successors, cycle_time, priority_scores, n):
    """Simulate greedy station assignment and return number of stations used."""
    assigned = [False] * n
    completed = [0] * n  # number of predecessors completed
    remaining_preds = [len(predecessors[i]) for i in range(n)]
    
    station_count = 0
    remaining = n
    
    while remaining > 0:
        station_count += 1
        remaining_time = cycle_time
        # Find eligible tasks (all predecessors assigned)
        eligible = []
        for i in range(n):
            if not assigned[i] and remaining_preds[i] == 0:
                eligible.append(i)
        
        if not eligible:
            break
        
        # Sort by priority (descending) and assign greedily
        eligible.sort(key=lambda i: priority_scores[i], reverse=True)
        
        for i in eligible:
            if not assigned[i] and times[i] <= remaining_time:
                assigned[i] = True
                remaining_time -= times[i]
                remaining -= 1
                # Update successor counts
                for v in successors[i]:
                    remaining_preds[v] -= 1
    
    return station_count

def priority(instance):
    start_time = _time.time()
    DEADLINE = 145.0  # safety margin before 150s kill
    
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    
    # Build successor list
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        preds = predecessors[i]
        for p in preds:
            successors[p].append(i)
            in_degree[i] += 1
    
    # Topological order (Kahn's algorithm)
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo = []
    indeg_copy = in_degree[:]
    while queue:
        u = queue.popleft()
        topo.append(u)
        for v in successors[u]:
            indeg_copy[v] -= 1
            if indeg_copy[v] == 0:
                queue.append(v)
    
    # If not all nodes in topo, there's a cycle; fall back to LPT
    if len(topo) != n:
        return times.copy()
    
    # Compute longest downstream path (rank position) in reverse topological order
    # rank[i] = times[i] + max(rank[j] for j in successors[i], default 0)
    rank = np.zeros(n, dtype=float)
    for u in reversed(topo):
        max_succ = 0.0
        for v in successors[u]:
            if rank[v] > max_succ:
                max_succ = rank[v]
        rank[u] = times[u] + max_succ
    
    # Also compute number of successors (out-degree)
    out_degree = np.array([len(successors[i]) for i in range(n)], dtype=float)
    
    # Multi-criteria baseline: combine rank and out-degree
    # Higher rank and more successors should get higher priority
    max_rank = np.max(rank) if n > 0 else 1.0
    max_outdeg = np.max(out_degree) if n > 0 else 1.0
    
    # Normalize
    norm_rank = rank / (max_rank + 1e-9)
    norm_outdeg = out_degree / (max_outdeg + 1e-9)
    
    # Baseline priority: weighted combination
    base_priority = 0.7 * norm_rank + 0.3 * norm_outdeg
    
    # Deadline check before local search
    if _time.time() - start_time > DEADLINE - 10.0:
        return base_priority
    
    # Local search: try to improve by adjusting priorities
    best_priority = base_priority.copy()
    best_stations = _simulate_stations(times, predecessors, successors, cycle_time, best_priority, n)
    
    # Try different weight combinations
    weight_options = [
        (0.8, 0.2), (0.6, 0.4), (0.9, 0.1), (0.5, 0.5),
        (0.75, 0.25), (0.65, 0.35), (0.85, 0.15)
    ]
    
    for w1, w2 in weight_options:
        if _time.time() - start_time > DEADLINE - 5.0:
            break
        trial_priority = w1 * norm_rank + w2 * norm_outdeg
        stations = _simulate_stations(times, predecessors, successors, cycle_time, trial_priority, n)
        if stations < best_stations:
            best_stations = stations
            best_priority = trial_priority.copy()
    
    # Try adding a small bonus for large tasks (they're harder to fit)
    if _time.time() - start_time > DEADLINE - 5.0:
        return best_priority
    
    max_time = np.max(times) if n > 0 else 1.0
    norm_time = times / (max_time + 1e-9)
    
    for time_bonus in [0.1, 0.2, 0.3]:
        if _time.time() - start_time > DEADLINE - 3.0:
            break
        trial_priority = best_priority + time_bonus * norm_time
        stations = _simulate_stations(times, predecessors, successors, cycle_time, trial_priority, n)
        if stations < best_stations:
            best_stations = stations
            best_priority = trial_priority.copy()
    
    return best_priority
# EVOLVE-BLOCK-END
