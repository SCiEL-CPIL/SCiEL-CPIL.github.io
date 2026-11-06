"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE_MARGIN = 5.0  # seconds of safety margin before the 150s wall-clock kill
_START_TIME = time.perf_counter()


def _elapsed():
    return time.perf_counter() - _START_TIME


def _check_deadline():
    """Return True if we are too close to the deadline and should stop early."""
    return _elapsed() > (150.0 - _DEADLINE_MARGIN)


def _topo_order(n, preds):
    """Return a topological order (list of task indices) given predecessors as list of lists."""
    in_deg = np.zeros(n, dtype=int)
    succs = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            succs[p].append(i)
            in_deg[i] += 1
    # Kahn's algorithm
    from collections import deque
    q = deque([i for i in range(n) if in_deg[i] == 0])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in succs[u]:
            in_deg[v] -= 1
            if in_deg[v] == 0:
                q.append(v)
    # If not all nodes were reached (shouldn't happen for valid DAG), append remaining
    if len(order) < n:
        seen = set(order)
        for i in range(n):
            if i not in seen:
                order.append(i)
    return order


def _longest_path_from_sources(n, preds, times, topo):
    """Compute longest path from any source to each node (earliest start time)."""
    dist = np.zeros(n, dtype=float)
    for u in topo:
        for v in _succs_cache.get(u, []) if False else []:
            pass
    # Recompute successors from preds
    succs = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            succs[p].append(i)
    for u in topo:
        du = dist[u] + times[u]
        for v in succs[u]:
            if du > dist[v]:
                dist[v] = du
    return dist


def _longest_path_to_sinks(n, preds, times, topo):
    """Compute longest path from each node to any sink (latest finish constraint)."""
    succs = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            succs[p].append(i)
    dist = np.zeros(n, dtype=float)
    for u in reversed(topo):
        best = 0.0
        for v in succs[u]:
            if dist[v] > best:
                best = dist[v]
        dist[u] = times[u] + best
    return dist


def _count_transitive_successors(n, preds):
    """Count number of transitive successors for each task."""
    succs = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            succs[p].append(i)
    # For each node, do BFS/DFS to count reachable nodes
    counts = np.zeros(n, dtype=int)
    for i in range(n):
        visited = [False] * n
        stack = [i]
        while stack:
            u = stack.pop()
            for v in succs[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        counts[i] = sum(visited)
    return counts


def _count_transitive_predecessors(n, preds):
    """Count number of transitive predecessors for each task."""
    pred_list = [list(preds[i]) for i in range(n)]
    counts = np.zeros(n, dtype=int)
    for i in range(n):
        visited = [False] * n
        stack = [i]
        while stack:
            u = stack.pop()
            for p in pred_list[u]:
                if not visited[p]:
                    visited[p] = True
                    stack.append(p)
        counts[i] = sum(visited)
    return counts


def _weighted_downstream_time(n, preds, times):
    """For each task, compute the sum of times of all transitive successors (downstream work it unlocks)."""
    succs = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            succs[p].append(i)
    result = np.zeros(n, dtype=float)
    for i in range(n):
        visited = [False] * n
        stack = list(succs[i])
        total = 0.0
        while stack:
            u = stack.pop()
            if not visited[u]:
                visited[u] = True
                total += times[u]
                stack.extend(succs[u])
        result[i] = total
    return result


def priority(instance):
    global _START_TIME
    _START_TIME = time.perf_counter()

    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    cycle_time = float(instance["cycle_time"])
    n = len(times)

    # Handle edge cases
    if n == 0:
        return np.array([], dtype=float)
    if _check_deadline():
        return times.copy()

    # Build topological order
    topo = _topo_order(n, preds)

    # Compute positional metrics
    # 1. Earliest start (longest path from sources)
    est = _longest_path_from_sources(n, preds, times, topo)

    # 2. Latest finish (longest path to sinks)
    lft = _longest_path_to_sinks(n, preds, times, topo)

    # 3. Transitive successor count
    succ_counts = _count_transitive_successors(n, preds)

    # 4. Transitive predecessor count
    pred_counts = _count_transitive_predecessors(n, preds)

    # 5. Weighted downstream time (total work unlocked by this task)
    downstream_time = _weighted_downstream_time(n, preds, times)

    if _check_deadline():
        return times.copy()

    # Build a composite priority score.
    # Goal: place tasks that constrain the timeline early (high EST relative to cycle),
    # have many successors (bottlenecks/fan-out), and are large.
    # We want HIGH priority for tasks we want to place first.

    # Normalize each component to [0, 1]
    def normalize(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    norm_est = normalize(est)
    norm_lft = normalize(lft)
    norm_succ = normalize(succ_counts.astype(float))
    norm_pred = normalize(pred_counts.astype(float))
    norm_time = normalize(times)
    norm_downstream = normalize(downstream_time)

    # Positional criticality: ratio of EST to cycle time, capped
    est_ratio = est / cycle_time if cycle_time > 0 else est
    norm_est_ratio = normalize(est_ratio)

    # Composite: weight combination
    # High downstream time -> unlocks large subgraph, place early
    # High successor count -> unlocks fan-out
    # High time -> big task
    # High EST ratio -> constrains start
    w_downstream = 0.30
    w_succ = 0.20
    w_time = 0.20
    w_est_ratio = 0.15
    w_lft = 0.10

    score = (w_downstream * norm_downstream +
             w_succ * norm_succ +
             w_time * norm_time +
             w_est_ratio * norm_est_ratio +
             w_lft * (1.0 - norm_lft))  # Lower LFT = more urgent

    # Ensure no NaNs or infs
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    if _check_deadline():
        return score

    return score
# EVOLVE-BLOCK-END
