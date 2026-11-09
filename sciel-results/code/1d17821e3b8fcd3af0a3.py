"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import defaultdict
import heapq

_DEADLINE = time.time() + 145.0  # safety margin before 150s kill


def _build_graph(instance):
    """Build adjacency, in-degree, and successor map."""
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    n = len(times)
    
    adj = defaultdict(list)       # task -> successors
    in_deg = np.zeros(n, dtype=int)
    
    for j in range(n):
        if j in preds:
            for i in preds[j]:
                adj[i].append(j)
                in_deg[j] += 1
    
    return adj, in_deg


def _topo_sort(n, in_deg, adj):
    """Kahn's algorithm for topological sort."""
    in_deg_copy = in_deg.copy()
    queue = [i for i in range(n) if in_deg_copy[i] == 0]
    heapq.heapify(queue)
    topo = []
    while queue:
        u = heapq.heappop(queue)
        topo.append(u)
        for v in adj[u]:
            in_deg_copy[v] -= 1
            if in_deg_copy[v] == 0:
                heapq.heappush(queue, v)
    return topo


def _longest_path_forward(times, topo, adj):
    """Longest path from start to each node."""
    n = len(times)
    fwd = np.zeros(n, dtype=float)
    for u in topo:
        fwd[u] = times[u]
        for v in adj[u]:
            if fwd[u] + times[v] > fwd[v]:
                fwd[v] = fwd[u] + times[v]
    return fwd


def _longest_path_backward(times, topo, adj):
    """Longest path from each node to end."""
    n = len(times)
    bwd = np.zeros(n, dtype=float)
    for u in reversed(topo):
        best = 0.0
        for v in adj[u]:
            if bwd[v] > best:
                best = bwd[v]
        bwd[u] = times[u] + best
    return bwd


def _count_successors(n, adj):
    """Count direct and indirect successors for each node."""
    succ_count = np.zeros(n, dtype=int)
    for i in range(n):
        visited = set()
        stack = list(adj[i])
        while stack:
            v = stack.pop()
            if v not in visited:
                visited.add(v)
                stack.extend(adj[v])
        succ_count[i] = len(visited)
    return succ_count


def _count_predecessors(instance):
    """Count direct predecessors for each task."""
    preds = instance["predecessors"]
    n = len(instance["times"])
    pred_counts = np.zeros(n, dtype=int)
    for i in range(n):
        if i in preds:
            pred_counts[i] = len(preds[i])
    return pred_counts


def _normalize(arr):
    """Normalize array to [0, 1]."""
    arr = np.asarray(arr, dtype=float)
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-9:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def _assign_stations(instance, priority_scores, cycle_time):
    """
    Greedy assignment: fill stations with highest-priority eligible task that fits.
    Returns number of stations used.
    """
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    n = len(times)
    
    adj, in_deg = _build_graph(instance)
    pred_counts = in_deg.copy()
    
    assigned = np.zeros(n, dtype=bool)
    rem_preds = pred_counts.copy()
    station_time = 0.0
    num_stations = 1
    
    # Precompute eligible set using a heap keyed by -priority for fast max extraction
    # But since eligibility changes dynamically, we'll use a simple scan but optimize
    # by maintaining a list of eligible tasks
    
    while assigned.sum() < n:
        # Find eligible tasks
        eligible = [i for i in range(n) if not assigned[i] and rem_preds[i] == 0]
        
        if not eligible:
            break
        
        # Among eligible, find those that fit in current station
        remaining_capacity = cycle_time - station_time
        fit = [i for i in eligible if times[i] <= remaining_capacity + 1e-9]
        
        if fit:
            # Pick highest priority among those that fit
            best = max(fit, key=lambda i: priority_scores[i])
        else:
            # Close current station, start new one
            station_time = 0.0
            num_stations += 1
            remaining_capacity = cycle_time
            fit = [i for i in eligible if times[i] <= remaining_capacity + 1e-9]
            if not fit:
                # Should not happen if times[i] <= cycle_time
                best = max(eligible, key=lambda i: priority_scores[i])
            else:
                best = max(fit, key=lambda i: priority_scores[i])
        
        assigned[best] = True
        station_time += times[best]
        
        # Update successor counts
        for s in adj[best]:
            rem_preds[s] -= 1
    
    return num_stations


def _generate_candidates(instance):
    """Generate multiple candidate priority vectors."""
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    adj, in_deg = _build_graph(instance)
    topo = _topo_sort(n, in_deg, adj)
    
    candidates = []
    
    # 1. Longest processing time
    candidates.append(np.asarray(times, dtype=float))
    
    # 2. Longest path to end (rank)
    bwd = _longest_path_backward(times, topo, adj)
    candidates.append(bwd)
    
    # 3. Critical path (forward + backward)
    fwd = _longest_path_forward(times, topo, adj)
    crit = fwd + bwd - times
    candidates.append(crit)
    
    # 4. Modified rank (successor count)
    succ_count = _count_successors(n, adj)
    candidates.append(succ_count.astype(float))
    
    # 5. Composite weighted
    p_time = _normalize(times)
    p_rank = _normalize(bwd)
    p_crit = _normalize(crit)
    p_succ = _normalize(succ_count.astype(float))
    composite = 0.35 * p_time + 0.3 * p_rank + 0.2 * p_crit + 0.15 * p_succ
    candidates.append(composite)
    
    # 6. Time * successor count
    candidates.append(times * (1.0 + succ_count.astype(float)))
    
    # 7. Backward path weighted by time
    candidates.append(bwd * (1.0 + 0.1 * times))
    
    # 8. Inverse predecessor count weighted
    pred_counts = _count_predecessors(instance)
    denom = 1.0 + pred_counts.astype(float)
    candidates.append(times / denom)
    
    # 9. Forward path
    candidates.append(fwd)
    
    # 10. Time * critical path
    candidates.append(times * crit)
    
    # 11. RPV with bottom level and successor count (from cell findings)
    # Bottom level = bwd, combined with successor count
    bottom_level = bwd.copy()
    succ_weighted = succ_count.astype(float)
    # Combine: prioritize tasks that have long downstream paths AND unlock many successors
    rpv_combined = 0.5 * _normalize(bottom_level) + 0.3 * _normalize(succ_weighted) + 0.2 * _normalize(times)
    candidates.append(rpv_combined * (bottom_level.max() + 1))  # scale to absolute values
    
    # 12. Pure bottom level normalized
    candidates.append(bottom_level)
    
    # 13. Successor count * time
    candidates.append(succ_count.astype(float) * times)
    
    # 14. Critical path * successor count
    candidates.append(crit * (1.0 + 0.5 * succ_count.astype(float)))
    
    # 15. Weighted: emphasize tasks with high downstream impact
    downstream_impact = bwd * succ_count.astype(float)
    candidates.append(downstream_impact)
    
    return candidates


def _local_search(instance, best_priority, best_num_stations, cycle_time, time_budget=10.0):
    """
    Local search: perturb the best priority vector and try to improve.
    Uses simulated annealing-like approach with decreasing temperature.
    """
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    
    rng = np.random.RandomState(12345)
    current = best_priority.copy()
    current_stations = best_num_stations
    
    # Determine scale for perturbations
    p_range = current.max() - current.min() + 1e-9
    
    start = time.time()
    max_time = time_budget
    
    temperature = 0.1 * p_range
    cooling = 0.95
    
    while time.time() - start < max_time:
        # Generate a perturbation
        if rng.random() < 0.3:
            # Swap two random positions
            i, j = rng.choice(n, size=2, replace=False)
            candidate = current.copy()
            candidate[i], candidate[j] = current[j], current[i]
        elif rng.random() < 0.6:
            # Add small noise
            noise = rng.normal(0, temperature, size=n)
            candidate = current + noise
        else:
            # Scale one element
            i = rng.randint(n)
            candidate = current.copy()
            candidate[i] *= rng.uniform(0.5, 1.5)
        
        try:
            ns = _assign_stations(instance, candidate, cycle_time)
        except Exception:
            continue
        
        if ns < current_stations:
            current = candidate
            current_stations = ns
            # Reset temperature on improvement
            temperature = 0.1 * p_range
        elif rng.random() < np.exp(-(ns - current_stations) / (temperature + 1e-9)):
            current = candidate
            current_stations = ns
        
        temperature *= cooling
    
    return current, current_stations


def priority(instance):
    """
    Score each assembly task. Stations are filled with the highest-priority 
    eligible task that fits the cycle time.
    
    Uses a model-and-optimize approach: generate parameterized priority vectors
    and select the one that minimizes station count.
    """
    start_time = time.time()
    times = np.asarray(instance["times"], dtype=float)
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Generate candidate priority vectors
    candidates = _generate_candidates(instance)
    
    # Evaluate each candidate and track the best
    best_priority = candidates[0]
    best_num_stations = _assign_stations(instance, candidates[0], cycle_time)
    
    for idx in range(1, len(candidates)):
        if time.time() > _DEADLINE - 30:
            break
        try:
            ns = _assign_stations(instance, candidates[idx], cycle_time)
            if ns < best_num_stations:
                best_num_stations = ns
                best_priority = candidates[idx]
        except Exception:
            continue
    
    # If we have time, try local optimization
    if time.time() < _DEADLINE - 35:
        time_budget = min(20.0, _DEADLINE - 35 - time.time())
        if time_budget > 2.0:
            best_priority, best_num_stations = _local_search(
                instance, best_priority, best_num_stations, cycle_time, time_budget
            )
    
    return best_priority
# EVOLVE-BLOCK-END
