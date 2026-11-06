"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque


def _compute_graph_metrics(times, predecessors):
    """Compute earliest start times, transitive successor counts, and topological order."""
    n = len(times)
    # Build adjacency: pred -> successors
    succ = [[] for _ in range(n)]
    pred_count = [0] * n
    for i in range(n):
        preds = predecessors[i]
        pred_count[i] = len(preds)
        for p in preds:
            succ[p].append(i)

    # Topological order via Kahn's algorithm
    in_degree = pred_count[:]
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo = []
    while queue:
        u = queue.popleft()
        topo.append(u)
        for v in succ[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    # Longest path from sources (earliest start time)
    earliest = [0.0] * n
    for u in topo:
        val = earliest[u] + times[u]
        for v in succ[u]:
            if val > earliest[v]:
                earliest[v] = val

    # Transitive successor count via reverse topological DP
    trans_succ_count = [0] * n
    for u in reversed(topo):
        total = len(succ[u])
        for v in succ[u]:
            total += trans_succ_count[v]
        trans_succ_count[u] = total

    return np.asarray(earliest, dtype=float), np.asarray(trans_succ_count, dtype=float), topo, succ


def priority(instance):
    """
    Score each assembly task for the priority rule.
    
    Strategy: Duration-dominant priority with modest boosts for:
      1. Transitive successor count (bottleneck/fan-out) - unlocks parallelism
      2. Positional criticality (low earliest start) - timeline start-phase constraint
    
    Rationale: Pure LPT (duration) is the proven baseline (score 1.0).
    We keep duration as the dominant term and add small, normalized boosts
    for tasks that are structural bottlenecks (many transitive successors)
    and tasks that constrain the start phase of the timeline.
    """
    t0 = time.time()
    deadline = t0 + 135.0  # safety margin well under 150s

    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)

    if n == 0:
        return np.array([], dtype=float)

    # Compute earliest start times and transitive successor counts
    earliest, trans_succ_count, topo, succ = _compute_graph_metrics(times, predecessors)

    # Check deadline
    if time.time() > deadline:
        return times.copy()

    # Normalize components to [0, 1]
    # Transitive successor count: higher is more critical (bottleneck)
    ts_min, ts_max = trans_succ_count.min(), trans_succ_count.max()
    if ts_max > ts_min:
        succ_norm = (trans_succ_count - ts_min) / (ts_max - ts_min)
    else:
        succ_norm = np.zeros(n)

    # Earliest start: lower is more critical (start phase constraint)
    e_min, e_max = earliest.min(), earliest.max()
    if e_max > e_min:
        pos_crit = (e_max - earliest) / (e_max - e_min)  # 1 = earliest possible
    else:
        pos_crit = np.zeros(n)

    # Normalize duration to [0, 1] for consistent scaling of boost terms
    t_min, t_max = times.min(), times.max()
    if t_max > t_min:
        dur_norm = (times - t_min) / (t_max - t_min)
    else:
        dur_norm = np.zeros(n)

    # Duration-dominant score with modest boosts:
    # - Primary: raw duration (proven LPT baseline)
    # - Secondary: small boost for high-fanout bottleneck tasks (alpha_succ)
    # - Tertiary: small boost for start-phase-constraining tasks (alpha_pos)
    #
    # The boosts are scaled by the duration range so they act as tie-breakers
    # and modest re-orderings rather than overriding the LPT ordering.
    alpha_succ = 0.3  # modest boost for bottleneck tasks
    alpha_pos = 0.1   # small boost for start-phase criticality

    # Scale boosts relative to duration range to keep them as refinements
    dur_range = (t_max - t_min) if t_max > t_min else 1.0

    score = times + alpha_succ * succ_norm * dur_range + alpha_pos * pos_crit * dur_range

    # Check deadline
    if time.time() > deadline:
        return times.copy()

    return score
# EVOLVE-BLOCK-END
