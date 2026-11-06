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
    
    # Build successor map
    successors = defaultdict(list)
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)
    
    # Track unassigned predecessor count
    unassigned_preds = np.array([len(predecessors[i]) for i in range(n)], dtype=int)
    
    eligible = set(i for i in range(n) if unassigned_preds[i] == 0)
    
    if not eligible and n > 0:
        return 10**9
    
    station_count = 1
    current_load = 0
    
    while eligible:
        # Sort eligible by priority descending
        candidates = sorted(eligible, key=lambda i: -priority_scores[i])
        placed_in_this_pass = False
        
        for task in candidates:
            if times[task] <= cycle_time - current_load:
                # Place task
                current_load += times[task]
                eligible.discard(task)
                placed_in_this_pass = True
                
                for succ in successors[task]:
                    unassigned_preds[succ] -= 1
                    if unassigned_preds[succ] == 0:
                        eligible.add(succ)
                break
        
        if not placed_in_this_pass:
            # Overflow to next station
            station_count += 1
            current_load = 0
            # Check for impossible case
            if any(times[t] > cycle_time for t in eligible):
                return 10**9
    
    return station_count


def _compute_critical_path(times, predecessors, successors):
    """Compute longest path from each task to any sink."""
    n = len(times)
    in_degree = np.array([len(predecessors[i]) for i in range(n)], dtype=int)
    
    topo_order = []
    in_deg_copy = in_degree.copy()
    q = deque()
    for i in range(n):
        if in_deg_copy[i] == 0:
            q.append(i)
    
    while q:
        u = q.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_deg_copy[v] -= 1
            if in_deg_copy[v] == 0:
                q.append(v)
    
    cp = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        cp[u] = times[u]
        for v in successors[u]:
            cp[u] = max(cp[u], times[u] + cp[v])
    
    return cp


def _compute_downstream_work(times, predecessors, successors):
    """Compute sum of times of all transitive successors (excluding self)."""
    n = len(times)
    in_degree = np.array([len(predecessors[i]) for i in range(n)], dtype=int)
    
    topo_order = []
    in_deg_copy = in_degree.copy()
    q = deque()
    for i in range(n):
        if in_deg_copy[i] == 0:
            q.append(i)
    
    while q:
        u = q.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_deg_copy[v] -= 1
            if in_deg_copy[v] == 0:
                q.append(v)
    
    dw = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        total = 0.0
        for v in successors[u]:
            total += times[v] + dw[v]
        dw[u] = total
    
    return dw


def _count_transitive(start, adj_map):
    """Count transitive successors/predecessors."""
    visited = set()
    stack = [start]
    while stack:
        node = stack.pop()
        for s in adj_map[node]:
            if s not in visited:
                visited.add(s)
                stack.append(s)
    return len(visited)


def _local_search(times, predecessors, cycle_time, initial_priority, max_iters=3000, deadline=None):
    """
    Local search: try swapping priorities, small perturbations, and reordering.
    Returns best priority found.
    """
    n = len(times)
    best_priority = initial_priority.copy().astype(float)
    best_score = _greedy_balance(times, predecessors, cycle_time, best_priority)
    
    rng = np.random.RandomState(42)
    no_improve = 0
    max_no_improve = 200
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        if no_improve > max_no_improve:
            break
        
        move_type = rng.random()
        new_priority = best_priority.copy()
        
        if move_type < 0.4:
            # Random swap
            i, j = rng.choice(n, 2, replace=False)
            new_priority[i], new_priority[j] = best_priority[j], best_priority[i]
        elif move_type < 0.7:
            # Small perturbation: add noise to a random task's priority
            i = rng.randint(n)
            new_priority[i] += rng.normal(0, 0.1)
        elif move_type < 0.85:
            # Swap adjacent in priority order
            order = np.argsort(-best_priority)
            k = rng.randint(n - 1)
            i, j = order[k], order[k + 1]
            new_priority[i], new_priority[j] = best_priority[j], best_priority[i]
        else:
            # Reverse a small segment in priority order
            order = np.argsort(-best_priority)
            k = rng.randint(n - 2)
            seg = order[k:k+3]
            for a, b in zip(seg, seg[::-1]):
                new_priority[a], new_priority[b] = best_priority[b], best_priority[a]
        
        score = _greedy_balance(times, predecessors, cycle_time, new_priority)
        if score < best_score:
            best_priority = new_priority
            best_score = score
            no_improve = 0
        elif score == best_score:
            no_improve += 1
            if rng.random() < 0.08:
                best_priority = new_priority
        else:
            no_improve += 1
    
    return best_priority


def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    
    n = len(times)
    if n == 0:
        return np.array([])
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Build successor map
    successors = defaultdict(list)
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)
    
    # Precompute useful quantities
    # Critical path (longest path from task to any sink)
    cp = _compute_critical_path(times, predecessors, successors)
    
    # Downstream work (sum of times of all transitive successors)
    dw = _compute_downstream_work(times, predecessors, successors)
    
    # Number of direct successors
    num_direct_succ = np.array([len(successors[i]) for i in range(n)], dtype=float)
    
    # Number of direct predecessors
    num_direct_pred = np.array([len(predecessors[i]) for i in range(n)], dtype=float)
    
    # Transitive successor counts
    trans_succ = np.array([_count_transitive(i, successors) for i in range(n)], dtype=float)
    
    # Bottleneck slack: (cycle_time - time_i) / (num_direct_succ + 1)
    # Smaller slack = task consumes more capacity relative to branching unlocked
    bottleneck_slack = (cycle_time - times) / (num_direct_succ + 1)
    
    # Candidate priority vectors
    candidates = []
    
    # 1. LPT
    candidates.append(times.copy())
    # 2. Critical path
    candidates.append(cp.copy())
    # 3. Time + critical path
    candidates.append(times + cp)
    # 4. Downstream work
    candidates.append(dw.copy())
    # 5. Critical path + downstream work
    candidates.append(cp + dw)
    # 6. Time + downstream work
    candidates.append(times + dw)
    # 7. Most direct successors
    candidates.append(num_direct_succ.copy())
    # 8. Time + successors
    candidates.append(times + num_direct_succ)
    # 9. Critical path + successors
    candidates.append(cp + num_direct_succ)
    # 10. Time - predecessors
    candidates.append(times - num_direct_pred)
    # 11. (time * successors) / (preds + 1)
    candidates.append(times * num_direct_succ / (num_direct_pred + 1))
    # 12. cp * num_direct_succ
    candidates.append(cp * num_direct_succ)
    # 13. Transitive successors
    candidates.append(trans_succ.copy())
    # 14. Time + transitive successors
    candidates.append(times + trans_succ)
    # 15. Critical path + transitive successors
    candidates.append(cp + trans_succ)
    # 16. Fewest transitive predecessors (start early)
    trans_pred = np.array([_count_transitive(i, {i: preds for i, preds in enumerate(predecessors)}) for i in range(n)], dtype=float)
    candidates.append(-trans_pred.copy())
    # 17. Time * transitive successors
    candidates.append(times * trans_succ)
    # 18. Combined: time + 0.5*cp + 0.5*dw
    candidates.append(times + 0.5 * cp + 0.5 * dw)
    # 19. cp * (1 + trans_succ)
    candidates.append(cp * (1 + trans_succ))
    # 20. Bottleneck slack (inverted: smaller slack = higher priority)
    candidates.append(-bottleneck_slack)
    # 21. cp - bottleneck_slack (favor critical path but penalize low slack)
    candidates.append(cp - 0.5 * bottleneck_slack)
    # 22. times - 0.3 * bottleneck_slack
    candidates.append(times - 0.3 * bottleneck_slack)
    # 23. dw + cp - bottleneck_slack
    candidates.append(dw + cp - 0.5 * bottleneck_slack)
    # 24. Normalized combination
    def normalize(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    cp_norm = normalize(cp)
    dw_norm = normalize(dw)
    slack_norm = normalize(bottleneck_slack)
    # 25. 0.4*cp + 0.35*dw + 0.25*(1 - slack)
    candidates.append(0.4 * cp_norm + 0.35 * dw_norm + 0.25 * (1 - slack_norm))
    # 26. 0.5*cp + 0.3*dw + 0.2*(1 - slack)
    candidates.append(0.5 * cp_norm + 0.3 * dw_norm + 0.2 * (1 - slack_norm))
    # 27. 0.3*cp + 0.4*dw + 0.3*(1 - slack)
    candidates.append(0.3 * cp_norm + 0.4 * dw_norm + 0.3 * (1 - slack_norm))
    
    # Evaluate each candidate and find the best
    best_score = float('inf')
    best_priority = candidates[0]
    scored_candidates = []
    
    for pri in candidates:
        if time.time() > deadline:
            break
        n_stations = _greedy_balance(times, predecessors, cycle_time, pri)
        scored_candidates.append((n_stations, pri))
        if n_stations < best_score:
            best_score = n_stations
            best_priority = pri.copy()
        if best_score == 1:
            break
    
    # Local search from top candidates (not just the best) to diversify
    # Sort by score, take top 3 distinct ones
    scored_candidates.sort(key=lambda x: x[0])
    seen_scores = []
    top_starts = []
    for sc, pri in scored_candidates:
        if sc not in seen_scores:
            top_starts.append(pri)
            seen_scores.append(sc)
        if len(top_starts) >= 3:
            break
    
    # Run local search from each top start, keep the best result
    best_ls_score = best_score
    best_ls_priority = best_priority
    
    for start_pri in top_starts:
        if time.time() > deadline:
            break
        ls_result = _local_search(
            times, predecessors, cycle_time, 
            start_pri.copy(), 
            max_iters=2500,
            deadline=deadline
        )
        ls_score = _greedy_balance(times, predecessors, cycle_time, ls_result)
        if ls_score < best_ls_score:
            best_ls_score = ls_score
            best_ls_priority = ls_result
        if best_ls_score == 1:
            break
    
    return best_ls_priority
# EVOLVE-BLOCK-END
