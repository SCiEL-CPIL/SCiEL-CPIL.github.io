"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _build_graph(n, predecessors):
    """Build successor list and in-degree from predecessors (list of lists)."""
    succ = [[] for _ in range(n)]
    in_deg = np.zeros(n, dtype=int)
    for i, preds in enumerate(predecessors):
        if i >= len(predecessors):
            continue
        for p in preds:
            if 0 <= p < n and p != i:
                succ[p].append(i)
                in_deg[i] += 1
    return succ, in_deg


def _topo_order(n, succ, in_deg):
    """Kahn topological sort, returns list (start -> end)."""
    indeg = in_deg.copy()
    queue = [i for i in range(n) if indeg[i] == 0]
    order = []
    while queue:
        u = queue.pop(0)
        order.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
    # Fallback for any unprocessed (shouldn't happen for a valid DAG)
    if len(order) < n:
        present = set(order)
        for i in range(n):
            if i not in present:
                order.append(i)
    return order


def _downstream_weight(n, succ, order):
    """Total number of tasks in each task's successor subtree (including itself)."""
    weight = np.ones(n, dtype=float)
    for i in reversed(order):
        for s in succ[i]:
            weight[i] += weight[s]
    return weight


def _rank(n, times, succ, order):
    """Longest-path Rank: times[i] + max(rank of successors)."""
    rank = np.zeros(n, dtype=float)
    for i in reversed(order):
        if succ[i]:
            rank[i] = times[i] + max(rank[j] for j in succ[i])
        else:
            rank[i] = times[i]
    return rank


def _norm(x):
    x = np.asarray(x, dtype=float)
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-12:
        return np.ones_like(x)
    return (x - mn) / (mx - mn)


def _greedy_stations(n, times, succ, in_deg, priority_scores, cycle_time):
    """Simulate greedy station filling. Returns (num_stations, assigned, station_loads)."""
    assigned = np.full(n, -1, dtype=int)
    station_loads = []
    station_idx = 0
    station_load = 0.0
    eligible = [i for i in range(n) if in_deg[i] == 0]
    remaining_indeg = in_deg.copy()
    placed = 0
    while placed < n:
        if not eligible:
            station_loads.append(station_load)
            station_idx += 1
            station_load = 0.0
            continue
        best = None
        best_p = -np.inf
        for i in eligible:
            if times[i] <= cycle_time - station_load:
                if priority_scores[i] > best_p:
                    best_p = priority_scores[i]
                    best = i
        if best is None:
            station_loads.append(station_load)
            station_idx += 1
            station_load = 0.0
            continue
        assigned[best] = station_idx
        station_load += times[best]
        placed += 1
        eligible.remove(best)
        for s in succ[best]:
            remaining_indeg[s] -= 1
            if remaining_indeg[s] == 0:
                eligible.append(s)
    station_loads.append(station_load)
    return station_idx + 1, assigned, station_loads


def priority(instance):
    t0 = time.time()
    deadline = t0 + 145.0  # safety margin before 150s kill

    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = float(instance.get("cycle_time", 40.0))
    n = len(times)

    succ, in_deg = _build_graph(n, predecessors)
    order = _topo_order(n, succ, in_deg)

    # Core signals
    rank = _rank(n, times, succ, order)
    down_w = _downstream_weight(n, succ, order)
    imm_succ = np.array([len(succ[i]) for i in range(n)], dtype=float)
    imm_succ_safe = np.maximum(imm_succ, 1.0)

    # Temporal congestion: ratio of processing time to available slack,
    # adjusted by competition (number of immediate successors sharing the slack).
    # A task that consumes a large fraction of the cycle's slack AND unlocks
    # many parallel tasks is a high-congestion (tight-fitting, high-branching) task.
    eps = 1e-6
    slack = np.maximum(cycle_time - times, eps)
    congestion = times / slack * imm_succ_safe

    # Downstream weight (log to avoid scale issues)
    down_score = np.log1p(down_w)

    # Predecessor-count mild penalty: tasks with many preds are harder to place early
    pred_count = in_deg.astype(float)
    pred_penalty = 1.0 / (1.0 + 0.3 * pred_count)

    # Normalize each component to [0,1]
    s_rank = _norm(rank)
    s_down = _norm(down_score)
    s_cong = _norm(congestion)
    s_time = _norm(times)
    s_pred = _norm(pred_penalty)

    # Base composite: Rank (critical path) dominates, supported by downstream
    # weight, temporal congestion, and a small LPT tie-breaker.
    base = (1.0 * s_rank
            + 0.7 * s_down
            + 0.5 * s_cong
            + 0.2 * s_time
            + 0.3 * s_pred)

    # Local refinement: simulate greedy with base scores, then boost tasks that
    # are "tight" in their station (little remaining capacity) and carry high
    # downstream weight. This reduces idle-time fragmentation.
    if time.time() < deadline - 5.0:
        try:
            stations, assigned, station_loads = _greedy_stations(
                n, times, succ, in_deg, base, cycle_time)
            # Per-station minimum remaining capacity (tightness proxy)
            min_rem = {}
            for i in range(n):
                st = assigned[i]
                rem = cycle_time - times[i]
                if st not in min_rem or rem < min_rem[st]:
                    min_rem[st] = rem
            tight = np.zeros(n, dtype=float)
            for i in range(n):
                st = assigned[i]
                tight[i] = 1.0 / (min_rem[st] + eps)
            tight = _norm(tight)
            # Blend: keep base, add a boost for tight tasks with high downstream weight
            base = base + 0.15 * tight * s_down
        except Exception:
            pass

    base = np.nan_to_num(base, nan=0.0, posinf=0.0, neginf=0.0)
    return base
# EVOLVE-BLOCK-END
