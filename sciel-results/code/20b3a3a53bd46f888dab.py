"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq
from collections import deque

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # safety margin before 150s wall-clock


def _elapsed_ok():
    return time.time() < _DEADLINE


def _build_successor_adjacency(predecessors):
    """Build adjacency list: predecessors[i] = list of direct predecessors of task i.
    We return successors: successors[j] = list of tasks that depend on j."""
    n = len(predecessors)
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
    return successors


def _topological_order(times, predecessors):
    """Return a topological ordering of tasks."""
    n = len(times)
    in_degree = [0] * n
    adj = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            adj[p].append(i)
            in_degree[i] += 1
    
    q = deque()
    for i in range(n):
        if in_degree[i] == 0:
            q.append(i)
    
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in adj[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                q.append(v)
    
    return order


def _compute_rank_by_longest_path_downstream(times, predecessors):
    """For each task, compute the longest path from that task to any successor."""
    n = len(times)
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
            in_degree[i] += 1
    
    q = deque()
    for i in range(n):
        if in_degree[i] == 0:
            q.append(i)
    
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                q.append(v)
    
    longest_down = [0.0] * n
    for u in reversed(topo):
        max_path = 0.0
        for v in successors[u]:
            cand = times[v] + longest_down[v]
            if cand > max_path:
                max_path = cand
        longest_down[u] = max_path
    
    return longest_down


def _compute_rank_by_number_of_successors(times, predecessors):
    """Count the number of descendants for each task."""
    n = len(times)
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
            in_degree[i] += 1
    
    q = deque()
    for i in range(n):
        if in_degree[i] == 0:
            q.append(i)
    
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                q.append(v)
    
    count_desc = [0] * n
    for u in reversed(topo):
        total = 0
        for v in successors[u]:
            total += 1 + count_desc[v]
        count_desc[u] = total
    
    return np.array(count_desc, dtype=float)


def _compute_upstream_longest_path(times, predecessors):
    """For each task, compute the longest path from any source to that task (earliest start)."""
    n = len(times)
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
            in_degree[i] += 1
    
    q = deque()
    for i in range(n):
        if in_degree[i] == 0:
            q.append(i)
    
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                q.append(v)
    
    longest_up = [0.0] * n
    for u in topo:
        for v in successors[u]:
            cand = times[u] + longest_up[u]
            if cand > longest_up[v]:
                longest_up[v] = cand
    
    return np.array(longest_up, dtype=float)


def _assign_stations_greedy(times, predecessors, cycle_time, priority_scores):
    """Assign tasks to stations using greedy rule: highest priority eligible task that fits.
    Maintains a ready-set of tasks whose predecessors are all assigned."""
    n = len(times)
    
    succ_set = [[] for _ in range(n)]
    remaining_preds = [0] * n
    for i in range(n):
        for p in predecessors[i]:
            succ_set[p].append(i)
            remaining_preds[i] += 1
    
    ready = []
    for i in range(n):
        if remaining_preds[i] == 0:
            ready.append(i)
    
    assigned = [False] * n
    station_count = 0
    station_load = 0.0
    
    while not all(assigned):
        # Best eligible task that fits current station load
        best_task = -1
        best_score = -1.0
        rem = cycle_time - station_load
        for t in ready:
            if not assigned[t] and times[t] <= rem:
                s = priority_scores[t]
                if s > best_score:
                    best_score = s
                    best_task = t
        
        if best_task < 0:
            # Close current station and start a new one
            station_count += 1
            station_load = 0.0
            rem = cycle_time
            for t in ready:
                if not assigned[t] and times[t] <= rem:
                    s = priority_scores[t]
                    if s > best_score:
                        best_score = s
                        best_task = t
        
        if best_task < 0:
            break
        
        assigned[best_task] = True
        station_load += times[best_task]
        
        for s in succ_set[best_task]:
            remaining_preds[s] -= 1
            if remaining_preds[s] == 0:
                ready.append(s)
    
    station_count += 1  # the last station
    return station_count


def _compute_time_bounded_downstream(times, predecessors, cycle_time, max_depth=2):
    """Sum of immediate-successor times, capped by remaining cycle time.
    Rewards tasks that can still be packed with successors within one station."""
    n = len(times)
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
            in_degree[i] += 1
    
    q = deque()
    for i in range(n):
        if in_degree[i] == 0:
            q.append(i)
    
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                q.append(v)
    
    tb = [0.0] * n
    for u in reversed(topo):
        if not successors[u]:
            tb[u] = 0.0
            continue
        # depth-1: direct successor times
        total = 0.0
        for v in successors[u]:
            total += times[v]
        tb[u] = min(total, cycle_time - times[u])
    
    return np.array(tb, dtype=float)


def _compute_comprehensive_priority(instance):
    """Compute a comprehensive priority score combining multiple heuristics."""
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # 1. Task time (longer is better)
    time_score = times / (max(times) + 1e-9)
    
    # 2. Longest path downstream (critical path depth)
    longest_down = _compute_rank_by_longest_path_downstream(times, predecessors)
    max_ld = max(longest_down) + 1e-9
    down_score = np.array(longest_down, dtype=float) / max_ld
    
    # 3. Number of successors (more dependent tasks = more urgent)
    succ_count = _compute_rank_by_number_of_successors(times, predecessors)
    max_sc = max(succ_count) + 1e-9
    succ_score = succ_count / max_sc
    
    # 4. Time-bounded downstream load (packing potential within a station)
    tb = _compute_time_bounded_downstream(times, predecessors, cycle_time)
    max_tb = max(tb) + 1e-9
    tb_score = tb / max_tb
    
    # 5. Positional criticality: ratio of earliest possible start time to cycle time
    #    Tasks with high upstream longest path are constrained by the timeline start
    longest_up = _compute_upstream_longest_path(times, predecessors)
    max_lu = max(longest_up) + 1e-9
    # Higher upstream path means the task is further from the start, so it's less critical
    # for early placement. We want to prioritize tasks with LOW upstream path (early in timeline)
    # but also consider that tasks with high upstream path may be bottlenecks.
    # Actually, for assembly line balancing, we want to place tasks that have many successors
    # and long downstream paths early. The upstream path indicates when a task can start.
    # Tasks with low upstream path can start early, so they should be prioritized to fill
    # the first stations efficiently.
    up_score = 1.0 - (longest_up / max_lu)  # Invert: low upstream path = high priority
    
    # Weighted combination: time + downstream path + successor count + packing potential + upstream position
    composite = (0.30 * time_score + 0.20 * down_score
                 + 0.15 * succ_score + 0.20 * tb_score + 0.15 * up_score)
    
    return composite


def priority(instance):
    """Score each assembly task. Returns one score per task."""
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    
    # Compute base priority using comprehensive heuristic
    base_priority = _compute_comprehensive_priority(instance)
    
    return base_priority
# EVOLVE-BLOCK-END
