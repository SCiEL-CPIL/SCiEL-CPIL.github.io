"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict, deque


def _topological_order(n, preds):
    """Return a topological ordering of tasks 0..n-1 given predecessors list."""
    in_degree = [0] * n
    succs = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            in_degree[i] += 1
            succs[p].append(i)
    q = deque([i for i in range(n) if in_degree[i] == 0])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in succs[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                q.append(v)
    # If there's a cycle (shouldn't happen), append remaining
    if len(order) < n:
        remaining = set(range(n)) - set(order)
        order.extend(sorted(remaining))
    return order


def _compute_longest_path_to_end(n, times, preds, succs):
    """Compute longest path from each node to any sink (end)."""
    order = _topological_order(n, preds)
    lp = [0.0] * n
    # Process in reverse topological order
    for u in reversed(order):
        max_succ = 0.0
        for v in succs[u]:
            if lp[v] > max_succ:
                max_succ = lp[v]
        lp[u] = times[u] + max_succ
    return lp


def _compute_longest_path_from_start(n, times, preds, succs):
    """Compute longest path from any source (start) to each node."""
    order = _topological_order(n, preds)
    lp = [0.0] * n
    for u in order:
        max_pred = 0.0
        for p in preds[u]:
            if lp[p] > max_pred:
                max_pred = lp[p]
        lp[u] = times[u] + max_pred
    return lp


def _count_successors(n, preds, succs):
    """Count number of direct and indirect successors for each node."""
    # Direct successors
    direct_count = [len(succs[i]) for i in range(n)]
    
    # Indirect successors via BFS/DFS from each node
    # For 30 nodes this is fine
    indirect_count = [0] * n
    for start in range(n):
        visited = set()
        stack = [start]
        while stack:
            u = stack.pop()
            for v in succs[u]:
                if v not in visited:
                    visited.add(v)
                    stack.append(v)
        indirect_count[start] = len(visited)
    
    return indirect_count


def _simulate_station_count(n, times, preds, succs, cycle_time, priority_scores):
    """
    Simulate the greedy station filling algorithm.
    Returns the number of stations used.
    """
    # Build successor set for quick lookup
    succ_set = [set(succs[i]) for i in range(n)]
    
    # Track which predecessors are done
    pred_count = [len(preds[i]) for i in range(n)]
    preds_remaining = [set(preds[i]) for i in range(n)]
    
    # Track remaining time in current station
    station_time = 0.0
    stations = 1
    assigned = [False] * n
    unassigned = set(range(n))
    
    while unassigned:
        # Find eligible tasks: all predecessors assigned, and fits in current station
        eligible = []
        for i in unassigned:
            if all(assigned[p] for p in preds[i]):
                if station_time + times[i] <= cycle_time + 1e-9:
                    eligible.append(i)
        
        if not eligible:
            # Start new station
            stations += 1
            station_time = 0.0
            # Re-find eligible
            for i in unassigned:
                if all(assigned[p] for p in preds[i]):
                    if station_time + times[i] <= cycle_time + 1e-9:
                        eligible.append(i)
        
        if not eligible:
            # Should not happen if problem is feasible, but break to avoid infinite loop
            break
        
        # Pick highest priority
        best = max(eligible, key=lambda i: priority_scores[i])
        
        assigned[best] = True
        unassigned.remove(best)
        station_time += times[best]
        
        # Update predecessors of successors
        # (not strictly needed since we check all preds each time, but for clarity)
    
    return stations


def priority(instance):
    """
    Compute priority scores for each task.
    Uses a multi-criteria approach:
    1. Longest path to end (critical path) - primary
    2. Number of successors (indirect) - secondary
    3. Topological position (earlier is higher priority) - tertiary
    
    The idea: tasks that are on critical paths and have many successors should
    be scheduled early to leave room for their dependents.
    """
    start_time = time.time()
    
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    
    # Parse predecessors
    preds_raw = instance.get("predecessors", [])
    preds = [list(p) if p else [] for p in preds_raw]
    
    # Build successor list
    succs = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            succs[p].append(i)
    
    cycle_time = instance.get("cycle_time", 40.0)
    
    # Compute critical path metrics
    lp_to_end = _compute_longest_path_to_end(n, times, preds, succs)
    lp_from_start = _compute_longest_path_from_start(n, times, preds, succs)
    
    # Count indirect successors
    succ_counts = _count_successors(n, preds, succs)
    
    # Topological position (0 = earliest)
    topo_order = _topological_order(n, preds)
    topo_pos = [0] * n
    for pos, node in enumerate(topo_order):
        topo_pos[node] = pos
    
    # Normalize each metric to [0, 1]
    def normalize(arr):
        arr = np.array(arr, dtype=float)
        mn = np.min(arr)
        mx = np.max(arr)
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    lp_end_norm = normalize(lp_to_end)
    lp_start_norm = normalize(lp_from_start)
    succ_norm = normalize(succ_counts)
    topo_norm = normalize(topo_pos)  # Earlier = smaller value, so we want to invert for priority
    
    # Invert topo so that earlier tasks get higher priority
    topo_inv = 1.0 - topo_norm
    
    # Combine: weighted sum
    # Primary: longest path to end (critical path)
    # Secondary: number of successors
    # Tertiary: topological position (earlier is better)
    w1 = 0.5   # critical path
    w2 = 0.3   # successors
    w3 = 0.2   # topological position
    
    scores = w1 * lp_end_norm + w2 * succ_norm + w3 * topo_inv
    
    # Ensure scores are positive and distinct enough
    scores = scores + 1e-6  # small offset to avoid zeros
    
    # Convert to numpy array
    return np.asarray(scores, dtype=float)
# EVOLVE-BLOCK-END
