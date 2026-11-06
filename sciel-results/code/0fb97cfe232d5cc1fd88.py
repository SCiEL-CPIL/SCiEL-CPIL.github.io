import numpy as np
import time

def priority(instance):
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    # Predecessor list per activity
    preds = instance.get("predecessors", None)
    if preds is None:
        preds = [[] for _ in range(n)]
    pred_lists = []
    for i in range(n):
        p = preds[i]
        if p is None:
            p = []
        pred_lists.append(list(p))
    
    # Successor list per activity
    succ_lists = [[] for _ in range(n)]
    for i in range(n):
        for p in pred_lists[i]:
            succ_lists[p].append(i)
    
    # Backward pass: longest path from each activity to a terminal
    # (including its own duration)
    # Use topological order (reverse)
    # Compute in-degree for topological sort
    in_degree = [len(pred_lists[i]) for i in range(n)]
    from collections import deque
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in succ_lists[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # Backward pass (reverse topological order)
    backward = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        max_succ = 0.0
        for v in succ_lists[u]:
            if backward[v] > max_succ:
                max_succ = backward[v]
        backward[u] = durations[u] + max_succ
    
    # Forward pass for total float
    forward = np.zeros(n, dtype=float)
    for u in topo_order:
        max_pred = 0.0
        for p in pred_lists[u]:
            if forward[p] > max_pred:
                max_pred = forward[p]
        forward[u] = max_pred + durations[u]
    
    project_duration = max(forward) if n > 0 else 0.0
    
    # Total float = backward - forward (how much slack)
    # Lower float = more critical = higher priority
    total_float = backward - forward  # >= 0 ideally
    
    # Number of immediate successors
    num_succ = np.array([len(succ_lists[i]) for i in range(n)], dtype=float)
    
    # Resource scarcity: for each resource, total demand / capacity
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    
    resource_pressure = np.zeros(n, dtype=float)
    if demands is not None and capacities is not None:
        demands_arr = np.asarray(demands, dtype=float)  # shape (n, R)
        capacities_arr = np.asarray(capacities, dtype=float)  # shape (R,)
        R = capacities_arr.shape[0]
        # For each activity, sum over resources of (demand / capacity)
        # Higher ratio = more scarce resource usage
        if demands_arr.ndim == 2 and demands_arr.shape[1] == R:
            # Normalize: demand_i_r / capacity_r
            ratios = demands_arr / np.where(capacities_arr > 0, capacities_arr, 1.0)
            resource_pressure = np.sum(ratios, axis=1)
    
    # Combine:
    # - backward path (longer = higher priority)
    # - negative total float (less slack = higher priority)
    # - num_succ (more successors = higher priority)
    # - resource_pressure (more scarce resources = higher priority)
    
    # Normalize each component to [0, 1] range for combination
    def normalize(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    norm_backward = normalize(backward)
    norm_float = normalize(-total_float)  # invert: less float = higher
    norm_succ = normalize(num_succ)
    norm_resource = normalize(resource_pressure)
    
    # Weighted combination
    w_backward = 0.4
    w_float = 0.3
    w_succ = 0.15
    w_resource = 0.15
    
    score = (w_backward * norm_backward + 
             w_float * norm_float + 
             w_succ * norm_succ + 
             w_resource * norm_resource)
    
    # Add a small tie-breaker based on original duration
    score += 0.05 * normalize(durations)
    
    return score
