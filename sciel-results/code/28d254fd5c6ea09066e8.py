"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 140.0  # safety margin before 150s kill


def _compute_priorities(instance):
    """Compute a priority score per task using Critical Path Density heuristic.
    
    Priority = task_time / slack, where slack = cycle_time - (task_time + downstream_time).
    This prioritizes tasks with high processing time and low scheduling flexibility.
    """
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    cycle_time = float(instance.get("cycle_time", 40.0))
    preds = instance.get("predecessors", None)

    # Build adjacency: succs[i] = list of successors of task i
    succs = [[] for _ in range(n)]
    if preds is not None:
        for i in range(n):
            p = preds[i]
            if p is not None:
                for j in p:
                    succs[j].append(i)

    # Topological sort (Kahn's algorithm)
    in_deg = [0] * n
    for i in range(n):
        if preds is not None and preds[i] is not None:
            in_deg[i] = len(preds[i])

    from collections import deque
    queue = deque([i for i in range(n) if in_deg[i] == 0])
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in succs[u]:
            in_deg[v] -= 1
            if in_deg[v] == 0:
                queue.append(v)

    # If topo order is incomplete (cycle), fall back to LPT
    if len(topo_order) != n:
        return times.copy()

    # down_time[i] = longest path from i to any sink (including i's own time)
    down_time = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        down_time[u] = times[u]
        for v in succs[u]:
            if down_time[v] > 0 or len(succs[u]) > 0:
                down_time[u] = max(down_time[u], times[u] + down_time[v])

    # up_time[i] = longest path from any source to i (including i's own time)
    up_time = np.zeros(n, dtype=float)
    for u in topo_order:
        up_time[u] = times[u]
        if preds is not None and preds[u] is not None:
            for v in preds[u]:
                up_time[u] = max(up_time[u], up_time[v] + times[u])

    # Critical path length
    critical_path = float(np.max(up_time)) if n > 0 else 0.0

    # desc_work[i] = total work in the downstream subtree of task i
    desc_work = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        desc_work[u] = times[u]
        for v in succs[u]:
            desc_work[u] += desc_work[v]

    # ===== Critical Path Density Heuristic =====
    # slack[i] = cycle_time - (times[i] + (down_time[i] - times[i]))
    #          = cycle_time - down_time[i]
    # This represents the scheduling flexibility: how much room remains
    # in a station after placing task i and its critical downstream path.
    
    eps = 1e-8
    # Downstream path excluding the task itself
    down_excl = down_time - times  # longest path after task i
    # Slack: how much extra time is available in a cycle beyond what i + its critical downstream needs
    slack = cycle_time - down_time
    # Ensure slack is at least a small positive value to avoid division by zero/negative
    slack = np.maximum(slack, eps)
    
    # Primary signal: Critical Path Density = time / slack
    # Tasks with high time and low slack (tight scheduling) get highest priority
    cpd = times / slack
    
    # Secondary signal: Critical path proximity (smooth)
    # path_length[i] = up_time[i] + down_time[i] - times[i] = longest path through i
    path_length = up_time + down_time - times
    dist_from_critical = np.maximum(critical_path - path_length, 0)
    # Exponential decay: tasks closer to critical path get higher score
    alpha_critical = 0.3
    criticality_score = np.exp(-alpha_critical * dist_from_critical)
    
    # Tertiary signal: Successor count urgency
    num_succs = np.array([len(s) for s in succs], dtype=float)
    succ_urgency = num_succs / (num_succs + 3.0)  # smooth saturation in [0,1)
    
    # Quaternary signal: Downstream work ratio
    total_work = float(np.sum(times))
    if total_work > 0:
        downstream_ratio = desc_work / total_work
    else:
        downstream_ratio = np.zeros(n, dtype=float)
    
    # Normalize components for combination
    max_cpd = np.max(cpd) if n > 0 else 1.0
    if max_cpd > 0:
        cpd_norm = cpd / max_cpd
    else:
        cpd_norm = np.zeros(n, dtype=float)
    
    # Combine: CPD is dominant, with critical path proximity and structural signals
    # Use a smooth multiplicative interaction to capture joint effects
    priority_scores = (
        cpd_norm * (1.0 + 0.5 * criticality_score) +
        0.15 * succ_urgency * cpd_norm +
        0.10 * downstream_ratio * cpd_norm
    )
    
    # Ensure all priorities are positive and well-scaled
    priority_scores = np.maximum(priority_scores, eps)
    
    return priority_scores


def priority(instance):
    """Score each assembly task; higher priority tasks are placed first."""
    start = time.time()
    try:
        scores = _compute_priorities(instance)
    except Exception:
        # Fallback to LPT
        try:
            scores = np.asarray(instance["times"], dtype=float)
        except Exception:
            scores = np.zeros(1)

    elapsed = time.time() - start
    if elapsed > _DEADLINE:
        # Shouldn't happen for 30 tasks, but guard anyway
        pass

    return scores
# EVOLVE-BLOCK-END
