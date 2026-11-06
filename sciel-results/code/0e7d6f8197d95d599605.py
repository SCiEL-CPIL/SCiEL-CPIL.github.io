"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict, deque

def _greedy_balance(times, predecessors, cycle_time, priority_scores):
    """
    Greedy station filling: repeatedly pick the highest-priority eligible task that fits.
    Returns number of stations.
    """
    n = len(times)
    if n == 0:
        return 0
    
    successors = defaultdict(list)
    unassigned_preds = np.array([len(predecessors[i]) for i in range(n)], dtype=int)
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)
    
    eligible = set(i for i in range(n) if unassigned_preds[i] == 0)
    
    if not eligible and n > 0:
        return 10**9
    
    station_count = 1
    current_load = 0
    
    while eligible:
        # Sort eligible by priority descending, with tie-breaking by shorter processing time
        candidates = sorted(eligible, key=lambda i: (-priority_scores[i], times[i]))
        placed_in_this_pass = False
        
        for task in candidates:
            if times[task] <= cycle_time - current_load:
                current_load += times[task]
                eligible.discard(task)
                placed_in_this_pass = True
                
                for succ in successors[task]:
                    unassigned_preds[succ] -= 1
                    if unassigned_preds[succ] == 0:
                        eligible.add(succ)
                break
        
        if not placed_in_this_pass:
            station_count += 1
            current_load = 0
            # Check for impossible case
            if any(times[t] > cycle_time for t in eligible):
                return 10**9
    
    return station_count


def _compute_topo_order(n, predecessors, successors):
    """Compute topological order using Kahn's algorithm."""
    in_degree = np.array([len(predecessors[i]) for i in range(n)], dtype=int)
    q = deque()
    for i in range(n):
        if in_degree[i] == 0:
            q.append(i)
    
    topo_order = []
    in_deg_copy = in_degree.copy()
    while q:
        u = q.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_deg_copy[v] -= 1
            if in_deg_copy[v] == 0:
                q.append(v)
    
    return topo_order


def _compute_critical_path(n, times, predecessors, successors):
    """Compute longest path from each task to any sink (downstream)."""
    topo_order = _compute_topo_order(n, predecessors, successors)
    cp = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        cp[u] = times[u]
        for v in successors[u]:
            cp[u] = max(cp[u], times[u] + cp[v])
    return cp


def _compute_upstream_path(n, times, predecessors, successors):
    """Compute longest path from any source to each task (upstream)."""
    topo_order = _compute_topo_order(n, predecessors, successors)
    up = np.zeros(n, dtype=float)
    for u in topo_order:
        for v in successors[u]:
            up[v] = max(up[v], up[u] + times[v])
    return up


def _count_transitive(start, adj):
    """Count number of transitive successors/predecessors."""
    visited = set()
    stack = [start]
    while stack:
        node = stack.pop()
        for s in adj[node]:
            if s not in visited:
                visited.add(s)
                stack.append(s)
    return len(visited)


def _compute_rank_by_successors(n, predecessors, successors):
    """Rank tasks by number of transitive successors (more successors = higher rank)."""
    ranks = np.zeros(n, dtype=float)
    for i in range(n):
        ranks[i] = _count_transitive(i, successors)
    return ranks


def _compute_rank_by_predecessors(n, predecessors, successors):
    """Rank tasks by number of transitive predecessors (fewer = higher rank)."""
    pred_map = defaultdict(list)
    for i in range(n):
        for p in predecessors[i]:
            pred_map[i].append(p)
    ranks = np.zeros(n, dtype=float)
    for i in range(n):
        ranks[i] = _count_transitive(i, pred_map)
    return ranks


def _compute_position_in_topo(n, predecessors, successors):
    """Get normalized position in topological order."""
    topo_order = _compute_topo_order(n, predecessors, successors)
    pos = np.zeros(n, dtype=float)
    for idx, task in enumerate(topo_order):
        pos[task] = idx / max(1, n - 1)
    return pos


def _compute_weighted_successors(n, times, predecessors, successors):
    """Sum of times of all transitive successors."""
    weighted = np.zeros(n, dtype=float)
    for i in range(n):
        visited = set()
        stack = [i]
        total = 0.0
        while stack:
            node = stack.pop()
            for s in successors[node]:
                if s not in visited:
                    visited.add(s)
                    total += times[s]
                    stack.append(s)
        weighted[i] = total
    return weighted


def _compute_weighted_predecessors(n, times, predecessors, successors):
    """Sum of times of all transitive predecessors."""
    pred_map = defaultdict(list)
    for i in range(n):
        for p in predecessors[i]:
            pred_map[i].append(p)
    weighted = np.zeros(n, dtype=float)
    for i in range(n):
        visited = set()
        stack = [i]
        total = 0.0
        while stack:
            node = stack.pop()
            for p in pred_map[node]:
                if p not in visited:
                    visited.add(p)
                    total += times[p]
                    stack.append(p)
        weighted[i] = total
    return weighted


def _compute_bottleneck_score(n, times, predecessors, successors, cycle_time):
    """
    Score tasks that are bottlenecks: tasks whose removal would increase the
    minimum number of stations. Approximated by tasks with high critical path
    and many successors.
    """
    cp = _compute_critical_path(n, times, predecessors, successors)
    num_succ = np.array([len(successors[i]) for i in range(n)], dtype=float)
    num_trans_succ = np.array([_count_transitive(i, successors) for i in range(n)], dtype=float)
    
    # Normalize
    cp_norm = cp / (cp.max() + 1e-9)
    succ_norm = num_trans_succ / (num_trans_succ.max() + 1e-9)
    
    return cp_norm * 0.6 + succ_norm * 0.4


def _generate_candidates(n, times, predecessors, successors, cycle_time):
    """Generate multiple candidate priority vectors."""
    candidates = []
    
    # Basic features
    cp = _compute_critical_path(n, times, predecessors, successors)
    up = _compute_upstream_path(n, predecessors, successors)
    
    num_direct_succ = np.array([len(successors[i]) for i in range(n)], dtype=float)
    num_direct_pred = np.array([len(predecessors[i]) for i in range(n)], dtype=float)
    
    num_trans_succ = np.array([_count_transitive(i, successors) for i in range(n)], dtype=float)
    pred_map = defaultdict(list)
    for i in range(n):
        for p in predecessors[i]:
            pred_map[i].append(p)
    num_trans_pred = np.array([_count_transitive(i, pred_map) for i in range(n)], dtype=float)
    
    weighted_succ = _compute_weighted_successors(n, times, predecessors, successors)
    weighted_pred = _compute_weighted_predecessors(n, times, predecessors, successors)
    
    bottleneck = _compute_bottleneck_score(n, times, predecessors, successors, cycle_time)
    topo_pos = _compute_position_in_topo(n, predecessors, successors)
    
    total_time = times.sum()
    max_time = times.max() if times.max() > 0 else 1.0
    
    # 1. Critical path
    candidates.append(cp.copy())
    
    # 2. Upstream path
    candidates.append(up.copy())
    
    # 3. Time
    candidates.append(times.copy())
    
    # 4. Critical path + time
    candidates.append(cp + times)
    
    # 5. Critical path * time
    candidates.append(cp * times)
    
    # 6. Weighted successors (sum of downstream times)
    candidates.append(weighted_succ.copy())
    
    # 7. Critical path + weighted successors
    candidates.append(cp + weighted_succ)
    
    # 8. Time * (1 + num_trans_succ)
    candidates.append(times * (1.0 + num_trans_succ))
    
    # 9. Critical path * (1 + num_trans_succ)
    candidates.append(cp * (1.0 + num_trans_succ))
    
    # 10. Upstream + downstream
    candidates.append(up + cp)
    
    # 11. Time * upstream + critical path
    candidates.append(times * up + cp)
    
    # 12. Time * (1 + up/max_up) + cp
    max_up = up.max() if up.max() > 0 else 1.0
    candidates.append(times * (1.0 + up / max_up) + cp)
    
    # 13. Bottleneck score
    candidates.append(bottleneck.copy())
    
    # 14. Time * bottleneck
    candidates.append(times * bottleneck)
    
    # 15. Critical path^2
    candidates.append(cp ** 2)
    
    # 16. Time * cp / max_cp
    max_cp = cp.max() if cp.max() > 0 else 1.0
    candidates.append(times * (cp / max_cp))
    
    # 17. (time/max_time) * cp + num_trans_succ
    candidates.append((times / max_time) * cp + num_trans_succ)
    
    # 18. Time * (1 + num_trans_succ/n) + cp * (1 - topo_pos)
    candidates.append(times * (1.0 + num_trans_succ / n) + cp * (1.0 - topo_pos))
    
    # 19. Weighted predecessors (tasks with less upstream work get priority)
    candidates.append(-weighted_pred.copy())
    
    # 20. Time * (1 + weighted_succ / max_ws)
    max_ws = weighted_succ.max() if weighted_succ.max() > 0 else 1.0
    candidates.append(times * (1.0 + weighted_succ / max_ws))
    
    # 21. Critical path * (1 + weighted_succ / max_ws)
    candidates.append(cp * (1.0 + weighted_succ / max_ws))
    
    # 22. Time * num_direct_succ + cp
    candidates.append(times * num_direct_succ + cp)
    
    # 23. (time/max_time)^2 * cp
    candidates.append((times / max_time) ** 2 * cp)
    
    # 24. Time * (1 + up/max_up) * (1 + cp/max_cp)
    candidates.append(times * (1.0 + up / max_up) * (1.0 + cp / max_cp))
    
    # 25. Log-weighted: log(1+time) * log(1+cp)
    candidates.append(np.log1p(times) * np.log1p(cp))
    
    # 26. Time * (1 + num_trans_succ / max_ts) * (1 + cp / max_cp)
    max_ts = num_trans_succ.max() if num_trans_succ.max() > 0 else 1.0
    candidates.append(times * (1.0 + num_trans_succ / max_ts) * (1.0 + cp / max_cp))
    
    # 27. Upstream * downstream combined
    candidates.append(up * cp)
    
    # 28. Time * (1 + (up+cp)/max_total)
    max_total = (up + cp).max() if (up + cp).max() > 0 else 1.0
    candidates.append(times * (1.0 + (up + cp) / max_total))
    
    # 29. Priority based on slack: tasks with less slack get higher priority
    #     Slack = cycle_time - time (simplified)
    slack = cycle_time - times
    candidates.append(-slack.copy())  # Less slack = higher priority
    
    # 30. Time / (1 + num_trans_pred) - tasks with fewer predecessors
    candidates.append(times / (1.0 + num_trans_pred))
    
    return candidates


def _local_search_improve(times, predecessors, cycle_time, initial_priority, deadline):
    """
    Local search: try swapping pairs of priority scores to improve the greedy result.
    """
    n = len(times)
    best_pri = initial_priority.copy()
    best_score = _greedy_balance(times, predecessors, cycle_time, best_pri)
    
    if best_score <= 1:
        return best_pri, best_score
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_score = best_score
        
        # Try all pairs
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                # Swap
                best_pri[i], best_pri[j] = best_pri[j], best_pri[i]
                new_score = _greedy_balance(times, predecessors, cycle_time, best_pri)
                if new_score < current_score:
                    current_score = new_score
                    best_score = new_score
                    improved = True
                    if current_score <= 1:
                        return best_pri, best_score
                else:
                    # Swap back
                    best_pri[i], best_pri[j] = best_pri[j], best_pri[i]
    
    return best_pri, best_score


def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    
    n = len(times)
    if n == 0:
        return np.array([])
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Build successors
    successors = defaultdict(list)
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)
    
    # Check for infeasible tasks
    if np.any(times > cycle_time):
        # Return a valid priority; the greedy will handle it
        return times.copy()
    
    # Generate candidates
    candidates = _generate_candidates(n, times, predecessors, successors, cycle_time)
    
    # Evaluate candidates with greedy
    best_score = float('inf')
    best_priority = candidates[0]
    
    for pri in candidates:
        if time.time() > deadline:
            break
        n_stations = _greedy_balance(times, predecessors, cycle_time, pri)
        if n_stations < best_score:
            best_score = n_stations
            best_priority = pri
        if best_score <= 1:
            break
    
    # Theoretical lower bound
    total_time = times.sum()
    lb_stations = max(1, int(np.ceil(total_time / cycle_time)))
    
    if best_score <= lb_stations:
        return best_priority
    
    # Local search refinement
    if time.time() < deadline - 10.0:
        best_priority, best_score = _local_search_improve(
            times, predecessors, cycle_time, best_priority, deadline - 5.0
        )
    
    return best_priority
# EVOLVE-BLOCK-END
