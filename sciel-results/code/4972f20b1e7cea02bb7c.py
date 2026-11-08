"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import defaultdict
import heapq

_DEADLINE = time.time() + 130.0  # safety margin before 150s kill


def _build_graph(instance):
    """Build successor map, predecessor counts, and topological order."""
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    n = len(times)
    
    succ = defaultdict(list)
    pred_counts = np.zeros(n, dtype=int)
    in_deg = np.zeros(n, dtype=int)
    
    for j in range(n):
        if j in preds:
            for i in preds[j]:
                succ[i].append(j)
                in_deg[j] += 1
            pred_counts[j] = len(preds[j])
    
    # Topological sort using Kahn's algorithm with heap for determinism
    queue = [i for i in range(n) if in_deg[i] == 0]
    heapq.heapify(queue)
    topo = []
    in_deg_copy = in_deg.copy()
    while queue:
        u = heapq.heappop(queue)
        topo.append(u)
        for v in succ[u]:
            in_deg_copy[v] -= 1
            if in_deg_copy[v] == 0:
                heapq.heappush(queue, v)
    
    return succ, pred_counts, topo


def _compute_longest_paths(instance, topo, succ):
    """Compute forward and backward longest paths."""
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    
    # Forward: longest path from start to u
    fwd = np.zeros(n, dtype=float)
    for u in topo:
        fwd[u] = times[u]
        for v in succ[u]:
            if fwd[u] + times[v] > fwd[v]:
                fwd[v] = fwd[u] + times[v]
    
    # Backward: longest path from u to end
    bwd = np.zeros(n, dtype=float)
    for u in reversed(topo):
        max_succ = 0.0
        for v in succ[u]:
            max_succ = max(max_succ, bwd[v])
        bwd[u] = times[u] + max_succ
    
    return fwd, bwd


def _compute_successor_counts(instance, succ):
    """Count total number of successors (direct + indirect) for each task."""
    n = len(instance["times"])
    succ_count = np.zeros(n, dtype=int)
    
    for i in range(n):
        visited = set()
        stack = list(succ[i])
        while stack:
            v = stack.pop()
            if v not in visited:
                visited.add(v)
                stack.extend(succ[v])
        succ_count[i] = len(visited)
    
    return succ_count


def _assign_stations(instance, priority_scores, cycle_time, succ, pred_counts):
    """
    Greedy assignment: fill stations with highest-priority eligible task that fits.
    Returns number of stations used.
    """
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    
    assigned = np.zeros(n, dtype=bool)
    rem_preds = pred_counts.copy()
    
    station_time = 0.0
    num_stations = 1
    
    while assigned.sum() < n:
        # Find eligible tasks
        eligible = [i for i in range(n) if not assigned[i] and rem_preds[i] == 0]
        
        if not eligible:
            break
        
        # Among eligible, find those that fit
        fit = [i for i in eligible if times[i] <= cycle_time - station_time + 1e-9]
        
        if fit:
            # Pick highest priority
            best = max(fit, key=lambda i: priority_scores[i])
        else:
            # Close current station, open new one
            station_time = 0.0
            num_stations += 1
            best = max(eligible, key=lambda i: priority_scores[i])
        
        assigned[best] = True
        station_time += times[best]
        for s in succ[best]:
            rem_preds[s] -= 1
    
    return num_stations


def _generate_candidates(instance, succ, pred_counts, topo, fwd, bwd, succ_counts):
    """Generate multiple candidate priority vectors."""
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    
    def normalize(arr):
        arr = np.asarray(arr, dtype=float)
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-9:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    candidates = {}
    
    # 1. Longest Processing Time (LPT)
    candidates["LPT"] = times.copy()
    
    # 2. Rank (longest path to end)
    candidates["Rank"] = bwd.copy()
    
    # 3. Critical path (forward + backward - time)
    candidates["Crit"] = fwd + bwd - times
    
    # 4. Modified rank (successor count)
    candidates["ModRank"] = succ_counts.astype(float)
    
    # 5. Composite: weighted combination
    p_time = normalize(times)
    p_rank = normalize(bwd)
    p_crit = normalize(fwd + bwd - times)
    p_succ = normalize(succ_counts.astype(float))
    candidates["Comp1"] = 0.25 * p_time + 0.25 * p_rank + 0.25 * p_crit + 0.25 * p_succ
    candidates["Comp2"] = 0.3 * p_time + 0.3 * p_rank + 0.2 * p_crit + 0.2 * p_succ
    candidates["Comp3"] = 0.4 * p_time + 0.2 * p_rank + 0.2 * p_crit + 0.2 * p_succ
    
    # 6. Time * rank
    candidates["TimeRank"] = times * bwd
    
    # 7. Time / (1 + pred_count)
    denom = 1.0 + pred_counts.astype(float)
    candidates["TimeDivPred"] = times / denom
    
    # 8. Reverse rank
    candidates["RevRank"] = -succ_counts.astype(float)
    
    # 9. Forward path only
    candidates["Fwd"] = fwd.copy()
    
    # 10. Time + successor count
    candidates["TimeSucc"] = times + 0.1 * succ_counts.astype(float)
    
    # 11. Bounded rank: bwd normalized and scaled
    candidates["NormBwd"] = normalize(bwd) * times.max()
    
    # 12. Hybrid: 0.5*times + 0.5*bwd
    candidates["Hybrid1"] = 0.5 * times + 0.5 * bwd
    
    # 13. Hybrid: 0.7*times + 0.3*bwd
    candidates["Hybrid2"] = 0.7 * times + 0.3 * bwd
    
    # 14. Hybrid: 0.3*times + 0.7*bwd
    candidates["Hybrid3"] = 0.3 * times + 0.7 * bwd
    
    return candidates


def _local_search(instance, initial_priority, cycle_time, succ, pred_counts, max_iter=50):
    """
    Simple local search: try swapping pairs of priority values.
    Returns best priority found.
    """
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    
    best_priority = initial_priority.copy()
    best_stations = _assign_stations(instance, best_priority, cycle_time, succ, pred_counts)
    
    for iteration in range(max_iter):
        if time.time() > _DEADLINE - 10:
            break
        
        improved = False
        # Try a few random swaps
        for _ in range(5):
            if time.time() > _DEADLINE - 10:
                break
            i, j = np.random.randint(0, n, 2)
            if i == j:
                continue
            
            trial = best_priority.copy()
            trial[i], trial[j] = trial[j], trial[i]
            
            ns = _assign_stations(instance, trial, cycle_time, succ, pred_counts)
            if ns < best_stations:
                best_priority = trial
                best_stations = ns
                improved = True
                break
        
        if not improved:
            break
    
    return best_priority, best_stations


def priority(instance):
    """
    Score each assembly task. Stations are filled with the highest-priority 
    eligible task that fits the cycle time.
    
    Uses multiple strategies and local search to find a good priority vector.
    """
    times = np.asarray(instance["times"], dtype=float)
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Build graph structures
    succ, pred_counts, topo = _build_graph(instance)
    fwd, bwd = _compute_longest_paths(instance, topo, succ)
    succ_counts = _compute_successor_counts(instance, succ)
    
    # Generate candidate priority vectors
    candidates = _generate_candidates(instance, succ, pred_counts, topo, fwd, bwd, succ_counts)
    
    # Evaluate each candidate
    best_priority = None
    best_num_stations = float('inf')
    
    for name, p in candidates.items():
        if time.time() > _DEADLINE - 30:
            break
        try:
            ns = _assign_stations(instance, p, cycle_time, succ, pred_counts)
            if ns < best_num_stations:
                best_num_stations = ns
                best_priority = p.copy()
        except Exception:
            continue
    
    if best_priority is None:
        best_priority = times.copy()
        best_num_stations = _assign_stations(instance, best_priority, cycle_time, succ, pred_counts)
    
    # Try local search from the best candidate
    if time.time() < _DEADLINE - 20:
        try:
            improved_priority, improved_stations = _local_search(
                instance, best_priority, cycle_time, succ, pred_counts, max_iter=100
            )
            if improved_stations < best_num_stations:
                best_priority = improved_priority
                best_num_stations = improved_stations
        except Exception:
            pass
    
    # If time remains, try local search from a few other good candidates
    if time.time() < _DEADLINE - 10:
        for name, p in list(candidates.items())[:5]:
            if time.time() > _DEADLINE - 5:
                break
            try:
                ns = _assign_stations(instance, p, cycle_time, succ, pred_counts)
                if ns <= best_num_stations:
                    improved_priority, improved_stations = _local_search(
                        instance, p, cycle_time, succ, pred_counts, max_iter=30
                    )
                    if improved_stations < best_num_stations:
                        best_priority = improved_priority
                        best_num_stations = improved_stations
            except Exception:
                continue
    
    return best_priority
# EVOLVE-BLOCK-END
