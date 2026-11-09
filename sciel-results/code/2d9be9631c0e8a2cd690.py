"""Direct approach for Assembly Line Balancing using explicit structure-derived priorities."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import defaultdict
import heapq

_START_TIME = time.time()
_DEADLINE = _START_TIME + 130.0  # safety margin before 150s kill


def _build_graph(instance):
    """Build adjacency and in-degree from predecessors."""
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    n = len(times)
    adj = defaultdict(list)
    in_deg = np.zeros(n, dtype=int)
    for j in range(n):
        if j in preds:
            for i in preds[j]:
                adj[i].append(j)
                in_deg[j] += 1
    return adj, in_deg, n


def _topo_order(n, in_deg, adj):
    """Kahn's algorithm for topological order."""
    in_deg = in_deg.copy()
    queue = [i for i in range(n) if in_deg[i] == 0]
    heapq.heapify(queue)
    topo = []
    while queue:
        u = heapq.heappop(queue)
        topo.append(u)
        for v in adj[u]:
            in_deg[v] -= 1
            if in_deg[v] == 0:
                heapq.heappush(queue, v)
    return topo


def _forward_longest(instance):
    """Longest path from start to each node."""
    adj, in_deg, n = _build_graph(instance)
    times = np.asarray(instance["times"], dtype=float)
    topo = _topo_order(n, in_deg, adj)
    fwd = np.zeros(n, dtype=float)
    for u in topo:
        fwd[u] = times[u]
        for v in adj[u]:
            if fwd[u] + times[v] > fwd[v]:
                fwd[v] = fwd[u] + times[v]
    return fwd


def _backward_longest(instance):
    """Longest path from each node to end."""
    adj, in_deg, n = _build_graph(instance)
    times = np.asarray(instance["times"], dtype=float)
    topo = _topo_order(n, in_deg, adj)
    bwd = np.zeros(n, dtype=float)
    for u in reversed(topo):
        max_succ = 0.0
        for v in adj[u]:
            if bwd[v] > max_succ:
                max_succ = bwd[v]
        bwd[u] = times[u] + max_succ
    return bwd


def _successor_counts(instance):
    """Count all (direct and indirect) successors for each task."""
    adj, in_deg, n = _build_graph(instance)
    counts = np.zeros(n, dtype=int)
    for i in range(n):
        visited = set()
        stack = list(adj[i])
        while stack:
            v = stack.pop()
            if v not in visited:
                visited.add(v)
                stack.extend(adj[v])
        counts[i] = len(visited)
    return counts


def _predecessor_counts(instance):
    """Count all (direct and indirect) predecessors for each task."""
    preds = instance["predecessors"]
    n = len(instance["times"])
    counts = np.zeros(n, dtype=int)
    radj = defaultdict(list)
    for j in range(n):
        if j in preds:
            for i in preds[j]:
                radj[j].append(i)
    for i in range(n):
        visited = set()
        stack = list(radj[i])
        while stack:
            v = stack.pop()
            if v not in visited:
                visited.add(v)
                stack.extend(radj[v])
        counts[i] = len(visited)
    return counts


def _assign_stations(instance, priority_scores, cycle_time, local_search=True):
    """
    Greedy assignment: for each station, repeatedly pick the highest-priority
    eligible task that fits. Returns number of stations used.
    """
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    n = len(times)
    
    pred_counts = np.zeros(n, dtype=int)
    for i in range(n):
        if i in preds:
            pred_counts[i] = len(preds[i])
    
    succ_map = defaultdict(list)
    for j in range(n):
        if j in preds:
            for i in preds[j]:
                succ_map[i].append(j)
    
    assigned = np.zeros(n, dtype=bool)
    rem_preds = pred_counts.copy()
    
    station_time = 0.0
    num_stations = 1
    
    station_of_task = np.full(n, -1, dtype=int)
    station_tasks = [[]]
    
    while assigned.sum() < n:
        eligible = [i for i in range(n) if not assigned[i] and rem_preds[i] == 0]
        
        if not eligible:
            break
        
        remaining_capacity = cycle_time - station_time
        fit = [i for i in eligible if times[i] <= remaining_capacity + 1e-9]
        
        if fit:
            best = max(fit, key=lambda i: priority_scores[i])
        else:
            station_time = 0.0
            num_stations += 1
            station_tasks.append([])
            fit = eligible
            best = max(fit, key=lambda i: priority_scores[i])
        
        assigned[best] = True
        station_time += times[best]
        station_of_task[best] = num_stations - 1
        station_tasks[num_stations - 1].append(best)
        
        for s in succ_map[best]:
            rem_preds[s] -= 1
    
    if local_search:
        improved = True
        max_iter = 80
        iter_count = 0
        while improved and iter_count < max_iter and time.time() < _DEADLINE - 5:
            improved = False
            iter_count += 1
            for s in range(num_stations - 1, 0, -1):
                if time.time() > _DEADLINE - 5:
                    break
                tasks_in_s = list(station_tasks[s])
                for t in tasks_in_s:
                    for s2 in range(s - 1, -1, -1):
                        can_move = True
                        for p in preds.get(t, []):
                            if station_of_task[p] >= s2:
                                can_move = False
                                break
                        if not can_move:
                            continue
                        cap_s2 = cycle_time - sum(times[x] for x in station_tasks[s2])
                        if times[t] <= cap_s2 + 1e-9:
                            station_tasks[s].remove(t)
                            station_tasks[s2].append(t)
                            station_of_task[t] = s2
                            improved = True
                            break
    
    non_empty = [s for s in range(num_stations) if len(station_tasks[s]) > 0]
    return len(non_empty)


def _priority_critical_path(instance):
    fwd = _forward_longest(instance)
    bwd = _backward_longest(instance)
    times = np.asarray(instance["times"], dtype=float)
    return fwd + bwd - times


def _priority_rank(instance):
    return _backward_longest(instance)


def _priority_time(instance):
    return np.asarray(instance["times"], dtype=float)


def _priority_successor_weighted(instance):
    times = np.asarray(instance["times"], dtype=float)
    succ = _successor_counts(instance).astype(float)
    return times * (1.0 + succ)


def _priority_composite(instance):
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    
    def norm(arr):
        arr = np.asarray(arr, dtype=float)
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-9:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    crit = norm(_priority_critical_path(instance))
    time_norm = norm(times)
    succ = norm(_successor_counts(instance).astype(float))
    pred = _predecessor_counts(instance).astype(float)
    pred_inv = norm(1.0 / (1.0 + pred))
    
    return 0.4 * crit + 0.3 * time_norm + 0.2 * succ + 0.1 * pred_inv


def _priority_time_rank_product(instance):
    times = np.asarray(instance["times"], dtype=float)
    rank = _backward_longest(instance)
    return times * rank


def _priority_min_pos(instance):
    times = np.asarray(instance["times"], dtype=float)
    succ = _successor_counts(instance).astype(float)
    return succ / (1.0 + times)


def _priority_fwd_rank_weighted(instance):
    fwd = _forward_longest(instance)
    succ = _successor_counts(instance).astype(float)
    fwd_min, fwd_max = fwd.min(), fwd.max()
    if fwd_max - fwd_min < 1e-9:
        fwd_norm = np.zeros_like(fwd)
    else:
        fwd_norm = (fwd - fwd_min) / (fwd_max - fwd_min)
    return fwd_norm * (1.0 + succ)


def _priority_soft_critical(instance):
    """
    Smooth surrogate: log-sum-exp over successor paths instead of hard max.
    """
    adj, in_deg, n = _build_graph(instance)
    times = np.asarray(instance["times"], dtype=float)
    topo = _topo_order(n, in_deg, adj)
    
    temp = 2.0
    
    bwd = np.zeros(n, dtype=float)
    for u in reversed(topo):
        if adj[u]:
            succ_vals = np.array([bwd[v] - times[v] for v in adj[u]], dtype=float)
            max_val = succ_vals.max()
            if max_val > -1e15:
                lse = max_val + np.log(np.sum(np.exp((succ_vals - max_val) / temp))) * temp
            else:
                lse = 0.0
            bwd[u] = times[u] + lse
        else:
            bwd[u] = times[u]
    
    return bwd


def _priority_rank_plus_time(instance):
    rank = _backward_longest(instance)
    times = np.asarray(instance["times"], dtype=float)
    return rank + 0.5 * times


def _priority_critical_time_ratio(instance):
    crit = _priority_critical_path(instance)
    times = np.asarray(instance["times"], dtype=float)
    return crit / (1.0 + times)


def _priority_fwd_bwd_product(instance):
    """
    New: product of forward rank and backward rank, normalized.
    Tasks that are both deep in the past and long in the future get highest priority.
    """
    fwd = _forward_longest(instance)
    bwd = _backward_longest(instance)
    times = np.asarray(instance["times"], dtype=float)
    
    def norm(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-9:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    return norm(fwd) * norm(bwd)


def _priority_bottleneck_weighted(instance):
    """
    New: weight by number of direct successors (bottleneck indicator).
    Tasks with many direct successors are bottlenecks; prioritize them.
    """
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    n = len(times)
    
    # Count direct successors
    direct_succ = np.zeros(n, dtype=int)
    for j in range(n):
        if j in preds:
            for i in preds[j]:
                direct_succ[i] += 1
    
    rank = _backward_longest(instance)
    return rank * (1.0 + direct_succ.astype(float))


def _priority_urgency_ratio(instance):
    """
    New: urgency = (backward rank) / (1 + time). 
    Favors tasks on long remaining paths that are individually small (can fit anywhere).
    """
    rank = _backward_longest(instance)
    times = np.asarray(instance["times"], dtype=float)
    return rank / (1.0 + times)


def _priority_depth_aware(instance):
    """
    New: combine depth (predecessor count) with critical path.
    Tasks with many predecessors are 'deep' and should be placed early to unblock.
    """
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    n = len(times)
    
    direct_pred = np.zeros(n, dtype=int)
    for i in range(n):
        if i in preds:
            direct_pred[i] = len(preds[i])
    
    crit = _priority_critical_path(instance)
    
    def norm(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-9:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    return norm(crit) * (1.0 + norm(direct_pred.astype(float)))


def priority(instance):
    """
    Score each assembly task. Stations are filled with the highest-priority 
    eligible task that fits the cycle time.
    
    Uses a direct approach: generate several structure-derived priority vectors,
    evaluate each with the greedy assignment, and return the best.
    """
    times = np.asarray(instance["times"], dtype=float)
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    candidates = [
        ("critical_path", _priority_critical_path(instance)),
        ("rank", _priority_rank(instance)),
        ("time", _priority_time(instance)),
        ("succ_weighted", _priority_successor_weighted(instance)),
        ("composite", _priority_composite(instance)),
        ("time_rank", _priority_time_rank_product(instance)),
        ("min_pos", _priority_min_pos(instance)),
        ("fwd_rank_weighted", _priority_fwd_rank_weighted(instance)),
        ("soft_critical", _priority_soft_critical(instance)),
        ("rank_plus_time", _priority_rank_plus_time(instance)),
        ("critical_time_ratio", _priority_critical_time_ratio(instance)),
        ("fwd_bwd_product", _priority_fwd_bwd_product(instance)),
        ("bottleneck_weighted", _priority_bottleneck_weighted(instance)),
        ("urgency_ratio", _priority_urgency_ratio(instance)),
        ("depth_aware", _priority_depth_aware(instance)),
    ]
    
    best_priority = np.asarray(times, dtype=float)
    best_num_stations = float('inf')
    
    # First pass: quick evaluation without local search to find a good candidate
    for name, p in candidates:
        if time.time() > _DEADLINE - 30:
            break
        try:
            ns = _assign_stations(instance, p, cycle_time, local_search=False)
            if ns < best_num_stations:
                best_num_stations = ns
                best_priority = p
        except Exception:
            continue
    
    # Second pass: local search on the top candidates to polish
    if time.time() < _DEADLINE - 15:
        # Re-evaluate top candidates with local search
        top_candidates = []
        for name, p in candidates:
            if time.time() > _DEADLINE - 15:
                break
            try:
                ns = _assign_stations(instance, p, cycle_time, local_search=False)
                top_candidates.append((ns, name, p))
            except Exception:
                continue
        
        top_candidates.sort(key=lambda x: x[0])
        
        for ns, name, p in top_candidates[:5]:
            if time.time() > _DEADLINE - 5:
                break
            try:
                ns_ls = _assign_stations(instance, p, cycle_time, local_search=True)
                if ns_ls < best_num_stations:
                    best_num_stations = ns_ls
                    best_priority = p
            except Exception:
                continue
    
    return best_priority
# EVOLVE-BLOCK-END
