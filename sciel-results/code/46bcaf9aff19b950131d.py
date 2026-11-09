"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq
from collections import deque

_DEADLINE = 120.0
_START_TIME = None


def _serial_schedule(durations, demands, capacities, predecessors, priority_scores):
    """
    Serial schedule generation scheme.
    durations: (n,) array of activity durations
    demands: (n, m) array of resource demands per activity per resource
    capacities: (m,) array of resource capacities
    predecessors: list of lists, predecessors[i] is list of indices
    priority_scores: (n,) array of priority scores (higher = earlier)
    Returns: makespan (total project duration)
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    n = len(durations)
    m = len(capacities)
    
    # Build successor lists and in-degrees
    in_degree = [0] * n
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            in_degree[i] += 1
            successors[p].append(i)
    
    # Track start times
    start_times = [0] * n
    completed = [False] * n
    scheduled_count = 0
    
    # Precompute: for each activity, which resources it uses and their demands
    act_resources = []
    for i in range(n):
        res_list = []
        for r in range(m):
            if demands[i][r] > 0:
                res_list.append((r, demands[i][r]))
        act_resources.append(res_list)
    
    # Resource usage tracking: for each resource, list of (start, end, amount)
    resource_intervals = [[] for _ in range(m)]
    
    def get_usage_at(r, t):
        """Get usage of resource r at time t."""
        usage = 0.0
        for (s, e, a) in resource_intervals[r]:
            if s <= t < e:
                usage += a
        return usage
    
    def can_start(activity, t):
        """Check if activity can start at time t given resource constraints."""
        for r, demand in act_resources[activity]:
            usage = get_usage_at(r, t)
            if usage + demand > capacities[r] + 1e-9:
                return False
        return True
    
    def earliest_start(activity, t_min):
        """Find earliest time >= t_min when activity can start."""
        # Candidate times: t_min and all interval end times > t_min for relevant resources
        candidates = [t_min]
        for r, demand in act_resources[activity]:
            for (s, e, a) in resource_intervals[r]:
                if e > t_min:
                    candidates.append(e)
        
        candidates = sorted(set(candidates))
        
        for ct in candidates:
            if ct >= t_min and can_start(activity, ct):
                return ct
        
        # Fallback: extend beyond all current intervals
        max_end = t_min
        for r, demand in act_resources[activity]:
            for (s, e, a) in resource_intervals[r]:
                if e > max_end:
                    max_end = e
        
        t = max_end
        while not can_start(activity, t):
            next_t = t + 1
            for r, demand in act_resources[activity]:
                for (s, e, a) in resource_intervals[r]:
                    if e > t and e < next_t:
                        next_t = e
            t = next_t
            if time.time() - _START_TIME > _DEADLINE - 3.0:
                return t
        
        return t
    
    # Use a heap for eligible activities: (-priority, activity_index)
    eligible_heap = []
    for i in range(n):
        if in_degree[i] == 0:
            heapq.heappush(eligible_heap, (-priority_scores[i], i))
    
    current_time = 0
    
    while scheduled_count < n:
        # Deadline check
        if time.time() - _START_TIME > _DEADLINE - 2.0:
            scheduled_max = 0
            for i in range(n):
                if completed[i]:
                    scheduled_max = max(scheduled_max, start_times[i] + durations[i])
            remaining_dur = sum(durations[i] for i in range(n) if not completed[i])
            return scheduled_max + remaining_dur
        
        # Find the highest priority eligible activity
        best_activity = None
        while eligible_heap:
            neg_p, a = heapq.heappop(eligible_heap)
            if not completed[a]:
                best_activity = a
                break
        
        if best_activity is None:
            break
        
        # Find earliest start time
        st = earliest_start(best_activity, current_time)
        
        # Schedule it
        start_times[best_activity] = st
        et = st + durations[best_activity]
        
        # Add resource usage intervals
        for r, demand in act_resources[best_activity]:
            resource_intervals[r].append((st, et, demand))
        
        completed[best_activity] = True
        scheduled_count += 1
        
        if st > current_time:
            current_time = st
        
        # Mark successors as potentially eligible
        for succ in successors[best_activity]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                heapq.heappush(eligible_heap, (-priority_scores[succ], succ))
    
    # Calculate makespan
    makespan = 0
    for i in range(n):
        if completed[i]:
            makespan = max(makespan, start_times[i] + durations[i])
    
    return makespan


def priority(instance):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demand"] if "demand" in instance else instance.get("demands", np.zeros((len(durations), 0))), dtype=float)
    capacities = np.asarray(instance["capacity"] if "capacity" in instance else instance.get("capacities", np.array([float('inf')])), dtype=float)
    predecessors = instance["predecessors"] if "predecessors" in instance else instance.get("predecessor_lists", [[] for _ in range(len(durations))])
    
    n = len(durations)
    m = len(capacities)
    
    # Ensure demands has shape (n, m)
    if demands.ndim == 1:
        demands = demands.reshape(n, 1)
        capacities = capacities.reshape(1)
        m = 1
    
    if len(capacities) == 0:
        m = 0
        capacities = np.array([float('inf')])
        demands = np.zeros((n, 1))
    
    # Ensure m matches
    if demands.shape[1] < m:
        pad = np.zeros((n, m - demands.shape[1]))
        demands = np.hstack([demands, pad])
    elif demands.shape[1] > m:
        demands = demands[:, :m]
    
    # Compute successors
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
    
    # Compute in-degree
    in_degree = [0] * n
    for i in range(n):
        for p in predecessors[i]:
            in_degree[i] += 1
    
    # Topological order
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for s in successors[node]:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)
    
    # If there's a cycle, handle gracefully
    if len(topo_order) < n:
        in_topo = set(topo_order)
        for i in range(n):
            if i not in in_topo:
                topo_order.append(i)
    
    # Compute head (longest path from start to i, excluding i's duration)
    head = np.zeros(n)
    for node in topo_order:
        best = 0
        for p in predecessors[node]:
            path = head[p] + durations[p]
            if path > best:
                best = path
        head[node] = best
    
    # Compute tail (longest path from i to end, excluding i's duration)
    tail = np.zeros(n)
    for node in reversed(topo_order):
        best = 0
        for s in successors[node]:
            path = durations[s] + tail[s]
            if path > best:
                best = path
        tail[node] = best
    
    # Critical path length
    cp_length = max(head[i] + durations[i] + tail[i] for i in range(n))
    
    # Total float (slack)
    total_float = cp_length - head - durations - tail
    total_float = np.maximum(total_float, 0)
    
    # Total resource demand per activity
    total_demand = demands.sum(axis=1)
    
    # Demand-to-capacity ratio (max across resources)
    demand_ratio_max = np.zeros(n)
    for r in range(m):
        if capacities[r] > 0:
            ratio = demands[:, r] / capacities[r]
            demand_ratio_max = np.maximum(demand_ratio_max, ratio)
    
    # Sum of demand-to-capacity ratios (total resource pressure)
    demand_ratio_sum = np.zeros(n)
    for r in range(m):
        if capacities[r] > 0:
            demand_ratio_sum += demands[:, r] / capacities[r]
    
    # Successor count
    succ_count = np.array([len(s) for s in successors], dtype=float)
    
    # Normalize each feature
    def normalize(x):
        x = np.asarray(x, dtype=float)
        rng = x.max() - x.min()
        if rng > 1e-12:
            return (x - x.min()) / rng
        return np.zeros_like(x)
    
    # For total_float, we want LOWER float to have HIGHER priority
    norm_float_inv = 1.0 - normalize(total_float)
    
    norm_tail = normalize(tail)
    norm_duration = normalize(durations)
    norm_total_demand = normalize(total_demand)
    norm_demand_ratio_max = normalize(demand_ratio_max)
    norm_demand_ratio_sum = normalize(demand_ratio_sum)
    norm_succ_count = normalize(succ_count)
    
    # Composite score with strong emphasis on CPM tail and float
    # Based on findings: tail (0.30) and float (0.45) over duration/resource
    w_float_inv = 0.45
    w_tail = 0.30
    w_duration = 0.08
    w_total_demand = 0.05
    w_demand_ratio_max = 0.05
    w_demand_ratio_sum = 0.05
    w_succ_count = 0.02
    
    scores = (w_float_inv * norm_float_inv + 
              w_tail * norm_tail + 
              w_duration * norm_duration + 
              w_total_demand * norm_total_demand +
              w_demand_ratio_max * norm_demand_ratio_max +
              w_demand_ratio_sum * norm_demand_ratio_sum +
              w_succ_count * norm_succ_count)
    
    return scores
# EVOLVE-BLOCK-END
