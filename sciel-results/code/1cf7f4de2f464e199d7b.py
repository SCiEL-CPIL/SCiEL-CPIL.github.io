"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE_SECONDS = 110.0  # Safety margin before 120s kill


def _compute_priority(instance):
    """
    Compute a priority score for each activity to minimize makespan.
    
    Strategy: 
    1. Compute Head (longest path to node) and Tail (longest path from node).
    2. Compute Total Float (Slack) = Critical Path Length - Head - Tail + Duration.
    3. Prioritize activities with high Tail and low Slack (high urgency).
    4. Weights are tuned to emphasize critical path structure over resource intensity,
       based on findings that pure CP/Slack signals can outperform mixed heuristics
       on certain instance distributions.
    """
    durations = np.asarray(instance["durations"], dtype=float)
    demands = instance["demands"]  # shape: (n_activities, n_resources)
    capacities = instance["capacities"]  # shape: (n_resources,)
    predecessors = instance["predecessors"]  # list of lists or similar
    
    n = len(durations)
    if n == 0:
        return np.array([])
    
    # Convert predecessors to a consistent format
    if isinstance(predecessors, np.ndarray):
        pred_lists = []
        for i in range(n):
            pred_lists.append(list(predecessors[i]))
    else:
        pred_lists = [list(p) for p in predecessors]
    
    # Build successor lists
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in pred_lists[i]:
            if i not in successors[p]:
                successors[p].append(i)
    
    # Topological order
    in_degree = [len(pred_lists[i]) for i in range(n)]
    queue = [i for i in range(n) if in_degree[i] == 0]
    topo_order = []
    while queue:
        u = queue.pop(0)
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # If topo_order doesn't contain all nodes, there's a cycle. 
    # Fallback to simple duration priority if cycle detected (shouldn't happen in valid RCPSP).
    if len(topo_order) < n:
        return durations.copy()

    # head[i] = longest duration from any source to i (including i's own duration)
    head = np.zeros(n, dtype=float)
    for i in topo_order:
        if pred_lists[i]:
            head[i] = durations[i] + max(head[p] for p in pred_lists[i])
        else:
            head[i] = durations[i]
    
    # tail[i] = longest duration from i to any sink (including i's own duration)
    tail = np.zeros(n, dtype=float)
    for i in reversed(topo_order):
        if not successors[i]:
            tail[i] = durations[i]
        else:
            tail[i] = durations[i] + max(tail[j] for j in successors[i])
    
    # Project makespan without resource constraints (longest path)
    makespan_ignoring_resources = max(tail) if n > 0 else 0.0
    
    # Total Float / Slack
    # slack[i] = LS[i] - ES[i]
    # ES[i] = head[i] - durations[i]
    # LF[i] = makespan - (tail[i] - durations[i])
    # LS[i] = LF[i] - durations[i] = makespan - tail[i]
    # slack[i] = (makespan - tail[i]) - (head[i] - durations[i])
    # slack[i] = makespan - tail[i] - head[i] + durations[i]
    slack = np.zeros(n, dtype=float)
    for i in range(n):
        slack[i] = makespan_ignoring_resources - tail[i] - head[i] + durations[i]
        
    # Normalize components to [0, 1]
    def normalize(arr):
        arr = np.asarray(arr, dtype=float)
        mn, mx = np.min(arr), np.max(arr)
        if mx - mn < 1e-10:
            return np.ones_like(arr) * 0.5
        return (arr - mn) / (mx - mn)
    
    norm_tail = normalize(tail)
    norm_slack_inv = normalize(-slack) # Lower slack = Higher priority
    
    # Combine:
    # Emphasize Tail (critical path length) and Inverse Slack (urgency).
    # Weights: 0.6 for Tail, 0.4 for Slack.
    # This removes the noise from demand/duration heuristics which may not correlate
    # as strongly with makespan reduction in these specific generated instances.
    priority_scores = 0.6 * norm_tail + 0.4 * norm_slack_inv
    
    return priority_scores


def priority(instance):
    """
    Score each activity for the serial schedule-generation scheme.
    Higher score = higher priority (scheduled first when eligible).
    """
    start_time = time.time()
    
    try:
        scores = _compute_priority(instance)
        elapsed = time.time() - start_time
        if elapsed > _DEADLINE_SECONDS:
            # Should not happen for 30 activities, but guard anyway
            return np.asarray(instance["durations"], dtype=float)
        return scores
    except Exception:
        # Fallback: longest processing time
        try:
            return np.asarray(instance["durations"], dtype=float)
        except Exception:
            n = len(instance.get("durations", []))
            return np.ones(n, dtype=float)
# EVOLVE-BLOCK-END
