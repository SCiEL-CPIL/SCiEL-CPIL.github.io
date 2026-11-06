import numpy as np
import time
from collections import deque

def priority(instance):
    t_start = time.time()
    deadline = 140.0
    
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    cycle_time = instance["cycle_time"]
    predecessors = instance["predecessors"]
    
    if n == 0:
        return np.array([], dtype=float)
    
    # Build successor map
    succ = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            succ[p].append(i)
    
    # Topological sort
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
        
    if len(topo_order) != n:
        return times

    # 1. Exact Longest Path (Hard Max)
    lp = [0.0] * n
    for node in reversed(topo_order):
        if succ[node]:
            lp[node] = times[node] + max(lp[s] for s in succ[node])
        else:
            lp[node] = times[node]

    # 2. Weighted Precedence (Sum of downstream)
    wp = [0.0] * n
    for node in reversed(topo_order):
        w = times[node]
        for s in succ[node]:
            w += wp[s]
        wp[node] = w

    # 3. Bottleneck Slack
    # slack = (cycle_time - time) / (num_succ + 1)
    # We want to prioritize SMALL slack.
    slack = [0.0] * n
    for i in range(n):
        num_succ = len(succ[i])
        slack[i] = (cycle_time - times[i]) / (num_succ + 1)
        
    # Normalize components to [0, 1]
    def normalize(arr):
        mn = min(arr)
        mx = max(arr)
        if mx == mn:
            return [0.0] * len(arr)
        return [(x - mn) / (mx - mn) for x in arr]

    lp_norm = normalize(lp)
    wp_norm = normalize(wp)
    time_norm = normalize(times)
    slack_norm = normalize(slack)
    
    # Combine
    # Weights: LP dominant, WP secondary, Time tertiary, Slack as penalty for large slack
    w_lp = 0.50
    w_wp = 0.30
    w_time = 0.10
    w_slack = 0.10
    
    priority_scores = np.zeros(n)
    for i in range(n):
        # High LP, High WP, High Time are good.
        # High Slack is bad (we want small slack).
        priority_scores[i] = (w_lp * lp_norm[i] + 
                              w_wp * wp_norm[i] + 
                              w_time * time_norm[i] - 
                              w_slack * slack_norm[i])
        
    # Tie-breaking with topological position
    eps = 1e-12
    topo_pos = [0] * n
    for pos, node in enumerate(topo_order):
        topo_pos[node] = pos
    for i in range(n):
        priority_scores[i] += eps * (n - topo_pos[i])
        
    return priority_scores
