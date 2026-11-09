"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 140.0  # safety margin before 150s kill


def _compute_priorities(instance):
    """Compute a priority score per task using bottleneck-density heuristic."""
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

    # Longest path from each node to any sink (downstream depth in time)
    # Use topological order (Kahn's algorithm)
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

    # down_time[i] = max time from i to any sink (longest path downstream including i)
    down_time = np.zeros(n, dtype=float)
    # Process in reverse topological order
    for u in reversed(topo_order):
        down_time[u] = times[u]
        for v in succs[u]:
            if down_time[v] > 0 or len(succs[u]) > 0:
                down_time[u] = max(down_time[u], times[u] + down_time[v])

    # up_time[i] = max time from any source to i (longest path upstream including i)
    up_time = np.zeros(n, dtype=float)
    for u in topo_order:
        up_time[u] = times[u]
        if preds is not None and preds[u] is not None:
            for v in preds[u]:
                up_time[u] = max(up_time[u], up_time[v] + times[u])

    # Critical path length
    critical_path = float(np.max(up_time)) if n > 0 else 0.0

    # Minimum number of remaining stations needed for downstream subtree of task i
    # Approximate: ceil(down_time[i] / cycle_time)
    # But more accurately, consider the total downstream work / cycle_time
    # Let's compute total downstream work (sum of times of all descendants + self)
    desc_work = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        desc_work[u] = times[u]
        for v in succs[u]:
            desc_work[u] += desc_work[v]

    # Minimum stations needed for downstream = max(ceil(down_time[i]/CT), ceil(desc_work[i]/CT))
    # Actually the binding constraint is the critical path through downstream
    min_stations_down = np.ceil(down_time / cycle_time)
    # Also consider work-based lower bound
    work_stations = np.ceil(desc_work / cycle_time)
    min_stations = np.maximum(min_stations_down, work_stations)
    min_stations = np.maximum(min_stations, 1)

    # Bottleneck density: time / min_stations
    bottleneck_density = times / min_stations

    # Combine with critical path position: tasks on the critical path get a boost
    # Criticality: up_time[i] + down_time[i] - times[i] vs critical_path
    # A task is on critical path if up_time[i] + down_time[i] - times[i] == critical_path
    on_critical = (up_time + down_time - times) >= (critical_path - 1e-9)
    critical_boost = on_critical.astype(float) * 0.5

    # Also weight by number of successors (bottleneck tasks with many successors are more important)
    num_succs = np.array([len(s) for s in succs], dtype=float)
    succ_weight = num_succs / (num_succs + 1.0)  # in [0,1)

    # Final priority: bottleneck_density + criticality boost + successor urgency
    priority_scores = bottleneck_density + 0.5 * critical_boost + 0.3 * succ_weight

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
