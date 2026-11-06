import numpy as np
import time
from collections import deque

def priority(instance):
    t_start = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    cycle_time = instance["cycle_time"]
    predecessors = instance["predecessors"]
    
    if n == 0:
        return np.array([], dtype=float)
    
    # Build successor map: succ[i] = list of tasks that depend on i
    succ = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            succ[p].append(i)
    
    # Topological sort (Kahn's algorithm)
    in_degree = [len(predecessors[i]) for i in range(n)]
    
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for s in succ[node]:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)
    
    # If not all nodes, there's a cycle - fallback
    if len(topo_order) != n:
        return times
    
    # Compute longest path from each node to a terminal (reverse topological order)
    # Use soft-max with temperature for smoother priority signal
    # lp[i] = times[i] + soft_max(lp[s] for s in succ[i]) if succ[i] else times[i]
    lp = [0.0] * n
    temp = 0.3  # temperature for soft-max (lower = closer to hard max, sharper critical path)
    for node in reversed(topo_order):
        if succ[node]:
            succ_lp = [lp[s] for s in succ[node]]
            # Soft-max: exp(x/temp) / sum(exp(x'/temp)) * max_value
            # This gives a smooth approximation of max
            m = max(succ_lp)
            exps = [np.exp((x - m) / temp) for x in succ_lp]
            soft_max_val = m + temp * np.log(sum(exps))
            lp[node] = times[node] + soft_max_val
        else:
            lp[node] = times[node]
    
    # Compute weighted precedence: sum of times of task and all transitive successors
    # wp[i] = times[i] + sum(wp[s] for s in succ[i])
    wp = [0.0] * n
    for node in reversed(topo_order):
        w = times[node]
        for s in succ[node]:
            w += wp[s]
        wp[node] = w
    
    # Compute number of immediate successors
    num_succ = [len(succ[i]) for i in range(n)]
    
    # Compute sum of immediate successor times (local packing pressure)
    succ_time_sum = [0.0] * n
    for i in range(n):
        for s in succ[i]:
            succ_time_sum[i] += times[s]
    
    # Compute depth: number of levels from node to deepest terminal
    depth = [0] * n
    for node in reversed(topo_order):
        if succ[node]:
            depth[node] = 1 + max(depth[s] for s in succ[node])
        else:
            depth[node] = 0
    
    # Combine heuristics with refined weights:
    # Primary: longest path (lp) - critical path remaining
    # Secondary: weighted precedence (wp) - total downstream work
    # Tertiary: number of successors - branching factor
    # Quaternary: depth - levels to terminal
    # Quinary: task time - larger tasks earlier to avoid fragmentation
    
    # Normalize each component to [0, 1] range for fair combination
    lp_min, lp_max = min(lp), max(lp)
    lp_range = lp_max - lp_min if lp_max > lp_min else 1.0
    
    wp_min, wp_max = min(wp), max(wp)
    wp_range = wp_max - wp_min if wp_max > wp_min else 1.0
    
    depth_min, depth_max = min(depth), max(depth)
    depth_range = depth_max - depth_min if depth_max > depth_min else 1.0
    
    time_min, time_max = min(times), max(times)
    time_range = time_max - time_min if time_max > time_min else 1.0
    
    max_num_succ = max(num_succ) if num_succ else 1
    max_succ_time = max(succ_time_sum) if succ_time_sum else 1.0
    if max_succ_time == 0:
        max_succ_time = 1.0
    
    priority_scores = np.zeros(n)
    for i in range(n):
        lp_norm = (lp[i] - lp_min) / lp_range
        wp_norm = (wp[i] - wp_min) / wp_range
        succ_norm = num_succ[i] / max_num_succ
        depth_norm = (depth[i] - depth_min) / depth_range
        time_norm = (times[i] - time_min) / time_range
        succ_time_norm = succ_time_sum[i] / max_succ_time
        
        # Refined weights: emphasize lp most (critical path), then wp (downstream work),
        # then local packing pressure (succ_time_sum), then task time
        priority_scores[i] = (0.48 * lp_norm + 
                            0.27 * wp_norm + 
                            0.08 * succ_norm + 
                            0.05 * depth_norm + 
                            0.07 * time_norm +
                            0.05 * succ_time_norm)
    
    # Check time budget
    if time.time() - t_start > deadline:
        return priority_scores
    
    # Final polish: ensure no ties by adding tiny epsilon based on topological position
    # Use topological order index for tie-breaking (earlier in topo = higher priority on tie)
    eps = 1e-12
    topo_pos = [0] * n
    for pos, node in enumerate(topo_order):
        topo_pos[node] = pos
    for i in range(n):
        priority_scores[i] += eps * (n - topo_pos[i])
    
    return priority_scores
