# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(instance):
    """
    Score each activity for a serial schedule-generation scheme.
    Higher score = higher priority (started earlier).
    
    Strategy: Combine critical path length (from start and to end),
    resource demand intensity, and duration to identify activities
    that are both on the critical path and resource-intensive.
    """
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill
    
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    # Get predecessors/successors
    preds = instance.get("predecessors", None)
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    
    # Build adjacency structures
    if preds is not None:
        # preds is likely a list of lists where preds[i] = list of predecessor indices
        pred_list = []
        succ_list = [[] for _ in range(n)]
        for i in range(n):
            if isinstance(preds, list):
                if i < len(preds):
                    p = preds[i]
                    if p is not None:
                        pred_list.append(list(p))
                        for j in p:
                            if 0 <= j < n:
                                succ_list[j].append(i)
                    else:
                        pred_list.append([])
                else:
                    pred_list.append([])
            else:
                pred_list.append([])
    else:
        pred_list = [[] for _ in range(n)]
        succ_list = [[] for _ in range(n)]
    
    # Compute longest path from each node to the end (backward pass)
    # slack[i] = longest path from i to any sink
    slack = np.zeros(n, dtype=float)
    in_degree = np.array([len(p) for p in pred_list], dtype=int)
    
    # Topological sort for forward pass (earliest start)
    from collections import deque
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    in_deg_copy = in_degree.copy()
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in succ_list[u]:
            in_deg_copy[v] -= 1
            if in_deg_copy[v] == 0:
                queue.append(v)
    
    # If topo sort didn't include all nodes (cycle), handle gracefully
    if len(topo_order) < n:
        # Add remaining nodes in reverse order as fallback
        seen = set(topo_order)
        for i in range(n):
            if i not in seen:
                topo_order.append(i)
    
    # Forward pass: earliest finish time
    ef = np.zeros(n, dtype=float)
    for u in topo_order:
        for v in succ_list[u]:
            if ef[u] + durations[u] > ef[v]:
                ef[v] = ef[u] + durations[u]
    
    # Backward pass: latest start time (from end)
    # Make span = max(ef)
    makespan_est = np.max(ef) if n > 0 else 0.0
    lf = np.full(n, makespan_est, dtype=float)
    for u in reversed(topo_order):
        for v in succ_list[u]:
            if lf[v] - durations[v] < lf[u]:
                lf[u] = lf[v] - durations[v]
    
    # Slack (total float)
    slack = np.maximum(0.0, lf - ef)
    
    # Critical path length from each node to end
    # This is the longest path starting at node i
    cp_to_end = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        if succ_list[u]:
            cp_to_end[u] = durations[u] + np.max([cp_to_end[v] for v in succ_list[u]])
        else:
            cp_to_end[u] = durations[u]
    
    # Critical path length from start to each node
    cp_from_start = np.zeros(n, dtype=float)
    for u in topo_order:
        if pred_list[u]:
            cp_from_start[u] = np.max([cp_from_start[p] + durations[p] for p in pred_list[u]])
        else:
            cp_from_start[u] = 0.0
    
    # Resource demand intensity
    if demands is not None and capacities is not None:
        # demands could be a list of lists (one per resource) or a 2D array
        try:
            demands_arr = np.asarray(demands, dtype=float)
            if demands_arr.ndim == 1:
                # Single resource
                demand_intensity = demands_arr / (np.max(demands_arr) + 1e-9)
            elif demands_arr.ndim == 2:
                # Multiple resources: demands_arr[i][r] = demand of activity i on resource r
                caps_arr = np.asarray(capacities, dtype=float)
                if caps_arr.ndim == 1:
                    # capacities[r] for each resource
                    # For each activity, compute max(demand/capacity) across resources
                    ratio = demands_arr / (caps_arr[np.newaxis, :] + 1e-9)
                    demand_intensity = np.max(ratio, axis=1)
                else:
                    demand_intensity = np.max(demands_arr, axis=1)
                    if np.max(demand_intensity) > 0:
                        demand_intensity = demand_intensity / np.max(demand_intensity)
        except:
            demand_intensity = np.ones(n) / n
    else:
        demand_intensity = np.ones(n) / n
    
    # Normalize components
    max_cp_end = np.max(cp_to_end) if n > 0 else 1.0
    if max_cp_end < 1e-9:
        max_cp_end = 1.0
    cp_end_norm = cp_to_end / max_cp_end
    
    max_cp_start = np.max(cp_from_start) if n > 0 else 1.0
    if max_cp_start < 1e-9:
        max_cp_start = 1.0
    cp_start_norm = cp_from_start / max_cp_start
    
    max_dur = np.max(durations) if n > 0 else 1.0
    if max_dur < 1e-9:
        max_dur = 1.0
    dur_norm = durations / max_dur
    
    # Successor count (activities with many successors are more critical)
    succ_count = np.array([len(succ_list[i]) for i in range(n)], dtype=float)
    max_succ = np.max(succ_count) if n > 0 else 1.0
    if max_succ < 1e-9:
        max_succ = 1.0
    succ_norm = succ_count / max_succ
    
    # Predecessor count (activities with many predecessors may be harder to start)
    pred_count = np.array([len(pred_list[i]) for i in range(n)], dtype=float)
    max_pred = np.max(pred_count) if n > 0 else 1.0
    if max_pred < 1e-9:
        max_pred = 1.0
    pred_norm = pred_count / max_pred
    
    # Combine scores with weights
    # Emphasize: critical path position, resource intensity, duration
    # Activities with low slack (critical) should be prioritized
    slack_inv = 1.0 / (slack + 1.0)  # Inverse slack: lower slack = higher score
    max_slack_inv = np.max(slack_inv) if n > 0 else 1.0
    if max_slack_inv < 1e-9:
        max_slack_inv = 1.0
    slack_inv_norm = slack_inv / max_slack_inv
    
    # Weighted combination
    w_cp_end = 0.30
    w_cp_start = 0.15
    w_dur = 0.15
    w_demand = 0.20
    w_slack_inv = 0.15
    w_succ = 0.05
    
    score = (w_cp_end * cp_end_norm +
             w_cp_start * cp_start_norm +
             w_dur * dur_norm +
             w_demand * demand_intensity +
             w_slack_inv * slack_inv_norm +
             w_succ * succ_norm)
    
    # Add small tiebreaker based on index to ensure deterministic ordering
    score += np.arange(n) * 1e-9
    
    return score
# EVOLVE-BLOCK-END
