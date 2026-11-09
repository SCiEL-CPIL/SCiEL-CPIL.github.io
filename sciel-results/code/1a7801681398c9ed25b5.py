"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque
import heapq
import time


def _build_graph(instance):
    """Build adjacency and in-degree arrays from the instance."""
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    n = len(times)
    
    adj = [[] for _ in range(n)]
    in_deg = np.zeros(n, dtype=int)
    
    for i in range(n):
        for p in preds[i]:
            adj[p].append(i)
            in_deg[i] += 1
    
    return times, adj, in_deg


def _topological_sort(adj, in_deg, n):
    """Return a topological ordering of tasks."""
    queue = deque()
    for i in range(n):
        if in_deg[i] == 0:
            queue.append(i)
    
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in adj[node]:
            in_deg[neighbor] -= 1
            if in_deg[neighbor] == 0:
                queue.append(neighbor)
    
    return order


def _simulate(instance, priority_scores):
    """
    Simulate the greedy list scheduling with given priority scores.
    Returns number of stations.
    """
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    adj = [[] for _ in range(n)]
    in_deg = np.zeros(n, dtype=int)
    for i in range(n):
        for p in preds[i]:
            adj[p].append(i)
            in_deg[i] += 1
    
    completed = [False] * n
    station_load = 0.0
    num_stations = 1
    remaining_in_deg = in_deg.copy()
    
    while True:
        eligible = []
        for i in range(n):
            if not completed[i] and remaining_in_deg[i] == 0:
                eligible.append(i)
        
        if not eligible:
            break
        
        eligible.sort(key=lambda i: priority_scores[i], reverse=True)
        
        placed_any = False
        for i in eligible:
            if station_load + times[i] <= cycle_time:
                completed[i] = True
                station_load += times[i]
                placed_any = True
                for j in adj[i]:
                    remaining_in_deg[j] -= 1
        
        if not placed_any:
            num_stations += 1
            station_load = 0.0
            continue
        
        if all(completed):
            break
    
    return num_stations


def priority(instance):
    """
    Score each assembly task using bottleneck density:
    priority[i] = rank_down[i] * (1 + times[i] / min_stations_downstream[i])
    
    where rank_down is the longest path from task to any sink,
    and min_stations_downstream is the minimum number of stations needed
    to complete the downstream subtree of task i.
    
    This penalizes long tasks that sit on critical paths where station
    capacity is scarce.
    """
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    cycle_time = instance["cycle_time"]
    
    preds = instance["predecessors"]
    adj = [[] for _ in range(n)]
    in_deg = np.zeros(n, dtype=int)
    for i in range(n):
        for p in preds[i]:
            adj[p].append(i)
            in_deg[i] += 1
    
    order = _topological_sort(adj, in_deg.copy(), n)
    
    # Downstream rank: longest path from task to any sink (including task's own time)
    rank_down = np.zeros(n, dtype=float)
    for i in reversed(order):
        max_succ = 0.0
        for j in adj[i]:
            max_succ = max(max_succ, rank_down[j])
        rank_down[i] = times[i] + max_succ
    
    # Compute minimum stations needed for downstream subtree of each task.
    # For a task i, the downstream work includes task i and all its successors.
    # The minimum number of stations to complete this work is at least
    # ceil(rank_down[i] / cycle_time), but we can do better by considering
    # the structure.
    
    # A better estimate: for each task, compute the "depth" in terms of
    # stations. Use a recursive formulation:
    # min_stations[i] = max(1, ceil(rank_down[i] / cycle_time))
    # But this is a lower bound. Let's use a more refined approach.
    
    # Actually, let's compute the "bottleneck density" as:
    # density[i] = times[i] / max(1, min_stations_est[i])
    # where min_stations_est[i] is an estimate of stations needed for task i's subtree.
    
    # For the subtree rooted at i (i and all descendants), the total work is rank_down[i].
    # The minimum stations is at least ceil(rank_down[i] / cycle_time).
    # But due to precedence constraints, it could be more.
    
    # Let's use: min_stations_est[i] = max(1, int(np.ceil(rank_down[i] / cycle_time)))
    # This gives a reasonable estimate.
    
    min_stations_est = np.maximum(1.0, np.ceil(rank_down / cycle_time))
    
    # Bottleneck density: ratio of task time to estimated stations for its subtree
    # Higher density means the task is a bigger bottleneck relative to available capacity
    bottleneck_density = times / min_stations_est
    
    # Combine: primary priority is downstream rank (critical path length),
    # weighted by a factor that increases with bottleneck density.
    # This ensures long tasks on tight critical paths get highest priority.
    
    # Use a smooth combination:
    # priority[i] = rank_down[i] * (1.0 + bottleneck_density[i])
    # This preserves the ordering by critical path length while boosting
    # tasks that are individually more constraining.
    
    priority_scores = rank_down * (1.0 + bottleneck_density)
    
    return priority_scores

# EVOLVE-BLOCK-END
