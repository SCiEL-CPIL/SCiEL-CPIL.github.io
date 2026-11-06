"""New program for Resource-Constrained Project Scheduling (serial SGS).

Approach: CPM Slack (Minimum Slack) combined with Resource Pressure.
This approach uses the Critical Path Method to calculate the slack (float)
for each activity. Activities with lower slack are more critical and are
given higher priority. This is combined with a resource pressure term to
prioritize activities that consume scarce resources among those with similar
slack.
"""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque


def priority(instance):
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)

    # Predecessor list per activity.
    preds = instance.get("predecessors", None)
    if preds is None:
        preds = [[] for _ in range(n)]
    # Normalise predecessors to lists of ints.
    pred_lists = []
    for i in range(n):
        p = preds[i]
        if p is None:
            p = []
        if isinstance(p, (list, tuple, np.ndarray)):
            p = [int(x) for x in p]
        else:
            p = [int(p)]
        # Filter out self and out-of-range.
        p = [x for x in p if 0 <= x < n and x != i]
        pred_lists.append(p)

    # Successor list per activity.
    succ_lists = [[] for _ in range(n)]
    for i in range(n):
        for j in pred_lists[i]:
            succ_lists[j].append(i)

    # ---- Topological Sort (Kahn's Algorithm) ----
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for j in pred_lists[i]:
            in_degree[i] += 1

    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)

    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in succ_lists[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    # Handle potential cycles (should not happen in RCPSP)
    if len(topo_order) < n:
        remaining = set(range(n)) - set(topo_order)
        topo_order.extend(list(remaining))

    # ---- CPM Forward Pass: Early Start (ES) and Early Finish (EF) ----
    ES = np.zeros(n, dtype=float)
    EF = np.zeros(n, dtype=float)
    
    for i in topo_order:
        if pred_lists[i]:
            ES[i] = max(EF[j] for j in pred_lists[i])
        else:
            ES[i] = 0.0
        EF[i] = ES[i] + durations[i]

    # Project makespan (unconstrained)
    makespan_unconstrained = max(EF) if n > 0 else 0.0

    # ---- CPM Backward Pass: Late Start (LS) and Late Finish (LF) ----
    LF = np.zeros(n, dtype=float)
    LS = np.zeros(n, dtype=float)
    
    # Initialize LF for sink nodes (no successors) to makespan
    for i in reversed(topo_order):
        if not succ_lists[i]:
            LF[i] = makespan_unconstrained
        else:
            LF[i] = min(LS[j] for j in succ_lists[i])
        LS[i] = LF[i] - durations[i]

    # ---- Slack Calculation ----
    # Slack = LS - ES
    slack = LS - ES
    
    # Ensure non-negative slack (numerical stability)
    slack = np.maximum(slack, 0.0)

    # ---- Resource Pressure Calculation ----
    demands = np.asarray(instance.get("demands", None), dtype=float)
    capacities = np.asarray(instance.get("capacities", None), dtype=float)

    resource_pressure = np.zeros(n, dtype=float)
    if demands is not None and demands.size > 0:
        if demands.ndim == 1:
            demands = demands.reshape(n, 1)
        if demands.shape[0] != n:
            demands = np.zeros((n, 1), dtype=float)

        R = demands.shape[1]
        if capacities is None or np.asarray(capacities).size == 0:
            caps = np.ones(R, dtype=float)
        else:
            caps = np.asarray(capacities, dtype=float).reshape(-1)
            if caps.size == 1:
                caps = np.full(R, float(caps[0]), dtype=float)
            elif caps.size < R:
                tmp = np.ones(R, dtype=float)
                tmp[:caps.size] = caps
                caps = tmp
            else:
                caps = caps[:R]
            caps = np.maximum(caps, 1e-9)

        # Per-activity resource usage ratio
        ratio = demands / caps[np.newaxis, :]  # (n, R)

        # Weight by resource scarcity
        resource_load = np.sum(ratio, axis=0)  # (R,)
        load_max = float(np.max(resource_load)) if R > 0 else 1.0
        if load_max <= 1e-12:
            load_weight = np.ones(R, dtype=float)
        else:
            load_weight = resource_load / load_max  # (R,)

        # Activity pressure: weighted sum of ratios
        resource_pressure = np.sum(ratio * load_weight[np.newaxis, :], axis=1)  # (n,)

        # Normalize pressure to [0, 1]
        rp_max = float(np.max(resource_pressure)) if n > 0 else 1.0
        if rp_max <= 1e-12:
            resource_pressure = np.zeros(n, dtype=float)
        else:
            resource_pressure = resource_pressure / rp_max

    # ---- Composite Priority ----
    # We want to minimize slack. So priority should be inversely proportional to slack.
    # Score = -slack + weight * resource_pressure
    # To make sure slack dominates, we scale it.
    # Normalize slack to [0, 1] so we can mix it with pressure.
    slack_max = float(np.max(slack)) if n > 0 else 1.0
    if slack_max <= 1e-12:
        slack_norm = np.zeros(n, dtype=float)
    else:
        slack_norm = slack / slack_max

    # Higher score = Higher Priority
    # Low slack (0) -> High priority component (1)
    # High slack (1) -> Low priority component (0)
    criticality = 1.0 - slack_norm
    
    # Combine: 80% criticality, 20% resource pressure
    score = 0.8 * criticality + 0.2 * resource_pressure

    # Tie-breaker: Higher duration first (optional, usually not needed if slack is distinct)
    # But for ties in slack, longer tasks are often better to start earlier.
    dur_max = float(np.max(durations)) if n > 0 else 1.0
    if dur_max > 1e-12:
        dur_norm = durations / dur_max
        score += 0.01 * dur_norm

    return np.asarray(score, dtype=float)
# EVOLVE-BLOCK-END
