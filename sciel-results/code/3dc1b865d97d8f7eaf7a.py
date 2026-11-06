"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(instance):
    """
    Priority scoring for serial schedule-generation scheme.
    
    Combines:
    - Critical path proximity (backward pass slack)
    - Resource demand pressure (demand/capacity ratio)
    - Successor count (bottleneck potential)
    - Duration
    """
    n = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)  # shape (n, num_resources)
    capacities = np.asarray(instance["capacities"], dtype=float)  # shape (num_resources,)
    
    # Build predecessor/successor lists
    predecessors = instance.get("predecessors", [[] for _ in range(n)])
    successors = [[] for _ in range(n)]
    for i in range(n):
        preds = predecessors[i] if isinstance(predecessors[i], list) else list(predecessors[i])
        for p in preds:
            successors[p].append(i)
    
    # Compute earliest start/finish times (forward pass)
    est = np.zeros(n, dtype=float)
    eft = np.zeros(n, dtype=float)
    # Topological order using Kahn's algorithm
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        preds = predecessors[i] if isinstance(predecessors[i], list) else list(predecessors[i])
        in_degree[i] = len(preds)
    
    queue = [i for i in range(n) if in_degree[i] == 0]
    topo_order = []
    while queue:
        node = queue.pop(0)
        topo_order.append(node)
        for succ in successors[node]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)
    
    # Forward pass
    for node in topo_order:
        preds = predecessors[node] if isinstance(predecessors[node], list) else list(predecessors[node])
        if preds:
            est[node] = max(eft[p] for p in preds)
        eft[node] = est[node] + durations[node]
    
    project_duration = max(eft) if n > 0 else 0
    
    # Backward pass: latest finish/start
    lft = np.full(n, project_duration, dtype=float)
    lst = np.full(n, project_duration, dtype=float)
    # Reverse topological order
    for node in reversed(topo_order):
        if successors[node]:
            lft[node] = min(lst[s] for s in successors[node])
        else:
            lft[node] = project_duration
        lst[node] = lft[node] - durations[node]
    
    # Slack = lst - est
    slack = lst - est
    
    # Resource demand pressure: for each activity, max(demand[r]/capacity[r])
    # Handle zero capacity
    safe_cap = np.where(capacities > 0, capacities, 1.0)
    demand_ratio = demands / safe_cap[np.newaxis, :]
    max_demand_ratio = np.max(demand_ratio, axis=1)
    
    # Normalize components to [0, 1] range
    # Slack: lower slack = higher priority (critical path)
    min_slack = np.min(slack)
    max_slack = np.max(slack)
    slack_range = max_slack - min_slack
    if slack_range > 1e-9:
        norm_slack = (max_slack - slack) / slack_range  # 1 for most critical, 0 for least
    else:
        norm_slack = np.ones(n, dtype=float)
    
    # Duration: longer = higher priority
    min_dur = np.min(durations)
    max_dur = np.max(durations)
    dur_range = max_dur - min_dur
    if dur_range > 1e-9:
        norm_dur = (durations - min_dur) / dur_range
    else:
        norm_dur = np.ones(n, dtype=float)
    
    # Demand ratio: higher = higher priority
    min_dr = np.min(max_demand_ratio)
    max_dr = np.max(max_demand_ratio)
    dr_range = max_dr - min_dr
    if dr_range > 1e-9:
        norm_dr = (max_demand_ratio - min_dr) / dr_range
    else:
        norm_dr = np.ones(n, dtype=float)
    
    # Successor count: more successors = higher priority
    succ_counts = np.array([len(successors[i]) for i in range(n)], dtype=float)
    min_sc = np.min(succ_counts)
    max_sc = np.max(succ_counts)
    sc_range = max_sc - min_sc
    if sc_range > 1e-9:
        norm_sc = (succ_counts - min_sc) / sc_range
    else:
        norm_sc = np.ones(n, dtype=float)
    
    # Weighted combination
    # Critical path proximity is most important, then resource pressure, then duration
    w_slack = 0.4
    w_dr = 0.3
    w_dur = 0.2
    w_sc = 0.1
    
    scores = w_slack * norm_slack + w_dr * norm_dr + w_dur * norm_dur + w_sc * norm_sc
    
    return scores
# EVOLVE-BLOCK-END
