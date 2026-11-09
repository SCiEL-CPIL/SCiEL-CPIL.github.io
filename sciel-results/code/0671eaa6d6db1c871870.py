"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque


def priority(instance):
    """
    Compute priority scores using a refined 'bottleneck density' heuristic.
    
    Priority is defined as:
    1. Bottleneck Density: time[i] / min_stations_downstream[i]
       where min_stations_downstream[i] is estimated by the longest path from i to any sink,
       divided by cycle_time (ceiled). This captures how "tight" the schedule is downstream.
    2. Successor Pressure: Number of immediate successors. Tasks with more successors
       are bottlenecks for many other tasks and should be prioritized.
    
    We combine these to favor tasks that are long and critical.
    """
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    cycle_time = instance["cycle_time"]
    
    # Handle edge case of empty instance
    if n == 0:
        return np.array([])
        
    # Build adjacency list
    preds = instance.get("predecessors", [None] * n)
    succ = [[] for _ in range(n)]
    for i in range(n):
        if preds[i] is not None:
            for p in preds[i]:
                if p < n and p >= 0: # Safety check
                    succ[p].append(i)
    
    # Topological Sort
    in_degree = [0] * n
    for i in range(n):
        if preds[i] is not None:
            for p in preds[i]:
                if p < n and p >= 0:
                    in_degree[i] += 1
    
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    temp_in_degree = in_degree[:]
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for s in succ[node]:
            temp_in_degree[s] -= 1
            if temp_in_degree[s] == 0:
                queue.append(s)
                
    # If there's a cycle or missing nodes, fallback to simple time-based priority
    if len(topo_order) != n:
        return times.copy()

    # Compute longest path from each node to any descendant (downstream critical path)
    # longest_down[i] = times[i] + max(longest_down[j] for j in succ[i])
    # This represents the minimum time required to complete the chain starting at i.
    longest_down = np.zeros(n, dtype=float)
    for node in reversed(topo_order):
        if succ[node]:
            longest_down[node] = times[node] + max(longest_down[s] for s in succ[node])
        else:
            longest_down[node] = times[node]
            
    # Estimate minimum number of stations required for the downstream path
    # Using longest path is a tighter lower bound for parallelism constraints than total sum,
    # but for "bottleneck density" in line balancing, we often care about the critical path length
    # relative to cycle time.
    # Let's use the critical path length to estimate the "depth" of the problem.
    # min_stations = ceil(longest_down / cycle_time)
    
    min_stations = np.ceil(longest_down / cycle_time)
    # Ensure at least 1 station
    min_stations = np.maximum(1.0, min_stations)
    
    # Bottleneck Density: How much of the cycle time is consumed by this task relative to the
    # number of stations it forces us to have downstream?
    # Actually, a better heuristic for "bottleneck" is:
    # If a task is on a long critical path, it should be placed early to allow its successors
    # to be placed in subsequent stations.
    # The ratio times[i] / min_stations[i] rewards tasks that are large relative to the
    # remaining "depth" of the network.
    
    bottleneck_density = times / min_stations
    
    # Secondary metric: Successor Count
    # Tasks with many successors are "bottlenecks" in the graph structure.
    succ_counts = np.array([len(s) for s in succ], dtype=float)
    
    # Normalize successor counts to [0, 1] for stable addition
    max_succ = succ_counts.max()
    if max_succ > 0:
        succ_norm = succ_counts / max_succ
    else:
        succ_norm = np.zeros(n)
        
    # We want to maximize priority.
    # High bottleneck density is good (long task, tight downstream).
    # High successor count is good (critical hub).
    
    # Let's weight them. Bottleneck density is the primary driver.
    # Successor count helps break ties and prioritize hubs.
    
    # To ensure the greedy filler picks the right tasks, we might want to scale
    # bottleneck density more aggressively.
    
    priority_scores = bottleneck_density + 0.2 * succ_norm
    
    return priority_scores
# EVOLVE-BLOCK-END
