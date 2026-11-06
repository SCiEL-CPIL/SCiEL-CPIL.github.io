"""Assembly Line Balancing - Model and Optimize approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
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
    
    if n > 0 and not eligible:
        return 10**9
    
    station_count = 1
    current_load = 0
    unplaced_count = n
    
    while eligible:
        # Sort eligible by priority descending
        candidates = sorted(eligible, key=lambda i: -priority_scores[i])
        placed = False
        
        for task in candidates:
            if times[task] <= cycle_time - current_load:
                # Place task
                current_load += times[task]
                eligible.discard(task)
                unplaced_count -= 1
                placed = True
                
                for succ in successors[task]:
                    unassigned_preds[succ] -= 1
                    if unassigned_preds[succ] == 0:
                        eligible.add(succ)
                break
        
        if not placed:
            # Start new station
            station_count += 1
            current_load = 0
            if any(times[t] > cycle_time for t in eligible):
                return 10**9
    
    return station_count


def _compute_critical_path(times, predecessors, successors, n):
    """Compute longest path from each task to any sink."""
    in_degree = np.array([len(predecessors[i]) for i in range(n)], dtype=int)
    in_deg_copy = in_degree.copy()
    q = deque()
    for i in range(n):
        if in_deg_copy[i] == 0:
            q.append(i)
    
    topo_order = []
    while q:
        u = q.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_deg_copy[v] -= 1
            if in_deg_copy[v] == 0:
                q.append(v)
    
    if len(topo_order) != n:
        return None  # Cycle detected
    
    cp = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        cp[u] = times[u]
        for v in successors[u]:
            cp[u] = max(cp[u], times[u] + cp[v])
    
    return cp


def _count_transitive_successors(start, succ_map):
    visited = set()
    stack = [start]
    while stack:
        node = stack.pop()
        for s in succ_map[node]:
            if s not in visited:
                visited.add(s)
                stack.append(s)
    return len(visited)


def _count_transitive_preds(start, pred_map):
    visited = set()
    stack = [start]
    while stack:
        node = stack.pop()
        for p in pred_map[node]:
            if p not in visited:
                visited.add(p)
                stack.append(s)
    return len(visited)


def _local_search(priority, times, predecessors, cycle_time, n, max_iters=5000, deadline=None):
    """
    Local search: try swapping priority values of two tasks, keep if it improves station count.
    """
    current_score = _greedy_balance(times, predecessors, cycle_time, priority)
    best_priority = priority.copy()
    best_score = current_score
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Pick two random tasks to swap
        i, j = random.sample(range(n), 2)
        
        # Swap priorities
        priority[i], priority[j] = priority[j], priority[i]
        
        new_score = _greedy_balance(times, predecessors, cycle_time, priority)
        
        if new_score < best_score:
            best_score = new_score
            best_priority = priority.copy()
        else:
            # Revert
            priority[i], priority[j] = priority[j], priority[i]
        
        if best_score == 1:
            break
    
    return best_priority, best_score


def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    
    n = len(times)
    if n == 0:
        return np.array([])
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Build successor map
    successors = defaultdict(list)
    pred_map = defaultdict(list)
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)
            pred_map[i].append(pred)
    
    # Compute features
    cp = _compute_critical_path(times, predecessors, successors, n)
    if cp is None:
        cp = np.zeros(n, dtype=float)
    
    num_succ = np.array([_count_transitive_successors(i, successors) for i in range(n)], dtype=float)
    num_pred = np.array([_count_transitive_preds(i, pred_map) for i in range(n)], dtype=float)
    
    # Direct successors count
    direct_succ = np.array([len(successors[i]) for i in range(n)], dtype=float)
    direct_pred = np.array([len(predecessors[i]) for i in range(n)], dtype=float)
    
    # Generate candidate initial priorities
    candidates = []
    
    # Critical path based
    candidates.append(cp.copy())
    candidates.append(cp + 0.1 * times)
    candidates.append(cp + 0.5 * num_succ)
    candidates.append(cp + 0.3 * direct_succ)
    
    # Time based
    candidates.append(times.copy())
    candidates.append(times + 0.2 * cp)
    
    # Successor based
    candidates.append(num_succ.copy())
    candidates.append(direct_succ.copy())
    candidates.append(cp + num_succ)
    
    # Combined
    candidates.append(0.5 * cp + 0.3 * times + 0.2 * num_succ)
    candidates.append(0.6 * cp + 0.4 * direct_succ)
    candidates.append(cp + times - 0.1 * num_pred)
    
    # Evaluate initial candidates
    best_priority = None
    best_score = float('inf')
    
    for pri in candidates:
        if time.time() > deadline - 30:
            break
        score = _greedy_balance(times, predecessors, cycle_time, pri)
        if score < best_score:
            best_score = score
            best_priority = pri.copy()
        if best_score == 1:
            break
    
    if best_priority is None:
        best_priority = cp.copy()
        best_score = _greedy_balance(times, predecessors, cycle_time, best_priority)
    
    # Local search from best initial
    if time.time() < deadline - 10:
        best_priority, best_score = _local_search(
            best_priority, times, predecessors, cycle_time, n,
            max_iters=20000, deadline=deadline - 5
        )
    
    # If we have more time, try local search from a few other starting points
    remaining_time = deadline - 5 - time.time()
    if remaining_time > 20 and best_score > 1:
        rng = np.random.RandomState(42)
        for start_idx in range(3):
            if time.time() > deadline - 5:
                break
            # Start from a different candidate
            pri = candidates[start_idx + 1].copy() if start_idx + 1 < len(candidates) else cp.copy()
            # Add some noise
            pri = pri + rng.normal(0, 0.1, n)
            
            pri, score = _local_search(
                pri, times, predecessors, cycle_time, n,
                max_iters=10000, deadline=deadline - 5 - max(0, (2 - start_idx) * 15)
            )
            
            if score < best_score:
                best_score = score
                best_priority = pri
    
    return best_priority
# EVOLVE-BLOCK-END
