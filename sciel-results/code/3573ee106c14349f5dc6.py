"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _topological_order(times, preds):
    """Return a topological ordering (list of indices)."""
    n = len(times)
    # Build successor lists
    succ = [[] for _ in range(n)]
    in_deg = [0] * n
    for j in range(n):
        for i in preds[j]:
            succ[i].append(j)
            in_deg[j] += 1
    # Kahn's algorithm
    from collections import deque
    q = deque([i for i in range(n) if in_deg[i] == 0])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in succ[u]:
            in_deg[v] -= 1
            if in_deg[v] == 0:
                q.append(v)
    # If not all nodes were added, there's a cycle; append remaining
    if len(order) < n:
        for i in range(n):
            if i not in order:
                order.append(i)
    return order

def _compute_ranks(times, preds):
    """Compute rank of each task (longest path to sink)."""
    n = len(times)
    succ = [[] for _ in range(n)]
    for j in range(n):
        for i in preds[j]:
            succ[i].append(j)
    # Process in reverse topological order
    order = _topological_order(times, preds)
    rank = [0.0] * n
    for u in reversed(order):
        if succ[u]:
            rank[u] = times[u] + max(rank[v] for v in succ[u])
        else:
            rank[u] = times[u]
    return np.array(rank, dtype=float)

def _compute_successor_count(preds):
    """Compute number of successors for each task."""
    n = len(preds)
    succ_count = [0] * n
    for j in range(n):
        for i in preds[j]:
            succ_count[i] += 1
    return np.array(succ_count, dtype=float)

def _simulate_station_count(priority_scores, times, preds, cycle_time):
    """Simulate the station-filling process and return number of stations."""
    n = len(times)
    # Build successor and predecessor sets
    succ = [set() for _ in range(n)]
    pred = [set() for _ in range(n)]
    for j in range(n):
        for i in preds[j]:
            succ[i].add(j)
            pred[j].add(i)
    
    # Sort tasks by priority (descending)
    order = sorted(range(n), key=lambda i: priority_scores[i], reverse=True)
    
    assigned = [False] * n
    remaining = set(range(n))
    station_count = 0
    
    while remaining:
        station_count += 1
        station_load = 0.0
        # Try to assign tasks in priority order to this station
        for i in order:
            if i not in remaining:
                continue
            # Check if all predecessors are assigned
            if all(p not in remaining for p in pred[i]):
                if station_load + times[i] <= cycle_time:
                    remaining.remove(i)
                    assigned[i] = True
                    station_load += times[i]
    
    return station_count

def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Compute various features
    rank = _compute_ranks(times, preds)
    succ_count = _compute_successor_count(preds)
    
    # Primary strategy: rank-based (longest path to sink)
    # This tends to work well for assembly line balancing
    scores = rank.copy()
    
    # Add a small tie-breaker based on number of successors
    # Tasks with more successors should be prioritized slightly
    # to avoid bottlenecks downstream
    scores = scores + 0.001 * succ_count
    
    # Additional tie-breaker: slightly prefer longer tasks among same rank
    scores = scores + 0.0001 * times
    
    return scores
# EVOLVE-BLOCK-END
