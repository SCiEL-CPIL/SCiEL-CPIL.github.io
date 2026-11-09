"""Priority scorer for Resource-Constrained Project Scheduling (serial SGS)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import deque

_DEADLINE = 150.0          # wall-clock kill limit
_SAFETY = 8.0              # return at least this many seconds before the limit


def _to_int_list(x):
    """Normalize a predecessor entry to a list of ints."""
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x]
    if isinstance(x, np.ndarray):
        return [int(v) for v in x]
    # single scalar predecessor
    return [int(x)]


def _cpm_ranks(instance):
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    preds = [_to_int_list(instance["predecessors"][i]) for i in range(n)]

    # in-degree for topological order
    indeg = np.zeros(n, dtype=int)
    succs = [[] for _ in range(n)]
    for j in range(n):
        for i in preds[j]:
            if 0 <= i < n and i != j:
                succs[i].append(j)
                indeg[j] += 1

    # forward pass (earliest start / earliest finish)
    order = []
    q = deque([i for i in range(n) if indeg[i] == 0])
    indeg_work = indeg.copy()
    while q:
        u = q.popleft()
        order.append(u)
        for v in succs[u]:
            indeg_work[v] -= 1
            if indeg_work[v] == 0:
                q.append(v)

    # fallback if cycle: append remaining
    if len(order) < n:
        seen = set(order)
        for i in range(n):
            if i not in seen:
                order.append(i)

    es = np.zeros(n, dtype=float)
    for u in order:
        for i in preds[u]:
            if 0 <= i < n:
                es[u] = max(es[u], es[i] + durations[i])
    ef = es + durations

    # project end from forward pass
    project_end = float(np.max(ef)) if n else 0.0

    # backward pass (latest start / latest finish)
    ls = np.full(n, project_end, dtype=float)
    for u in reversed(order):
        for v in succs[u]:
            if 0 <= v < n:
                ls[u] = min(ls[u], ls[v] - durations[u])
        # also respect project end
        ls[u] = min(ls[u], project_end - durations[u])
    lf = ls + durations

    # total float = latest start - earliest start
    total_float = ls - es
    # tail / bottom-level rank: longest path from node to any sink
    tail = np.zeros(n, dtype=float)
    for u in reversed(order):
        best = 0.0
        for v in succs[u]:
            if 0 <= v < n:
                best = max(best, durations[v] + tail[v])
        tail[u] = best

    # head: longest path from start to node (excluding node duration)
    head = np.zeros(n, dtype=float)
    for u in order:
        for i in preds[u]:
            if 0 <= i < n:
                head[u] = max(head[u], head[i] + durations[i])

    return total_float, tail, ef, project_end, head, succs, preds


def _greedy_simulate(durations, demands, capacities, preds, succs, base_scores, t0):
    """
    Simulate a greedy serial schedule using base_scores to rank eligible activities.
    Returns a 'pressure' score per activity based on how tightly they fit
    into the resource envelope during the simulation.
    """
    n = len(durations)
    if n == 0:
        return np.zeros(0)

    # Normalize demands to (n, m)
    if demands is None:
        demands = np.zeros((n, 1))
    demands = np.asarray(demands, dtype=float)
    if demands.ndim == 1:
        demands = demands.reshape(n, 1)
    
    if capacities is None:
        capacities = np.array([float('inf')])
    capacities = np.asarray(capacities, dtype=float)
    
    m = len(capacities)
    if demands.shape[1] < m:
        pad = np.zeros((n, m - demands.shape[1]))
        demands = np.hstack([demands, pad])
    elif demands.shape[1] > m:
        demands = demands[:, :m]

    # Initialize
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for p in preds[i]:
            if 0 <= p < n and p != i:
                in_degree[i] += 1

    eligible = set(i for i in range(n) if in_degree[i] == 0)
    scheduled = np.zeros(n, dtype=bool)
    start_times = np.zeros(n, dtype=float)
    
    # Resource usage intervals: list of (start, end, amount) per resource
    res_intervals = [[] for _ in range(m)]
    
    current_time = 0.0
    scheduled_count = 0
    
    # Track "pressure" or "slack" contribution for each activity
    # Idea: if an activity is scheduled exactly when resources are tight, it might be "good" 
    # but we want to penalize activities that *cause* bottlenecks or are delayed by them.
    # Let's measure: for each activity, how much "slack" was available when it started?
    # Or better: measure the "congestion" it faces.
    # Actually, the strategy says: "dynamically penalizing activities that would cause immediate resource bottlenecks".
    # In a static priority function, we can't know the future. But we can simulate a *reference* schedule
    # and see which activities are "critical" in terms of resource contention.
    
    # Let's compute a score based on:
    # 1. How late the activity is scheduled compared to its earliest possible start (delay due to resources)
    # 2. How much resource capacity it consumes relative to available slack at its start time.
    
    pressure = np.zeros(n, dtype=float)
    
    def get_usage(r, t):
        total = 0.0
        for s, e, a in res_intervals[r]:
            if s <= t < e:
                total += a
        return total
    
    def can_start(a, t):
        for r in range(m):
            d = demands[a, r]
            if d > 0:
                if get_usage(r, t) + d > capacities[r] + 1e-9:
                    return False
        return True
    
    def earliest_start(a, t_min):
        candidates = {t_min}
        for r in range(m):
            for s, e, _ in res_intervals[r]:
                if s > t_min:
                    candidates.add(s)
                if e > t_min:
                    candidates.add(e)
        
        cands = sorted(candidates)
        for ct in cands:
            if ct >= t_min and can_start(a, ct):
                return ct
        
        # Fallback search
        t = t_min
        max_t = max(cands) + 1000 if cands else t_min + 1000
        while t < max_t:
            if can_start(a, t):
                return t
            next_bp = t + 1
            for r in range(m):
                for s, e, _ in res_intervals[r]:
                    if e > t and e < next_bp:
                        next_bp = e
            t = next_bp
        return t

    while scheduled_count < n:
        if time.time() - t0 > _DEADLINE - _SAFETY:
            break
        
        if not eligible:
            break
            
        # Find best eligible activity by base score
        best_a = None
        best_score = -float('inf')
        for a in eligible:
            if base_scores[a] > best_score:
                best_score = base_scores[a]
                best_a = a
        
        if best_a is None:
            break
            
        st = earliest_start(best_a, current_time)
        et = st + durations[best_a]
        
        # Calculate resource slack at start time
        min_slack_ratio = float('inf')
        for r in range(m):
            d = demands[best_a, r]
            if d > 0:
                usage = get_usage(r, st)
                slack = capacities[r] - usage
                if slack <= 0:
                    min_slack_ratio = 0.0
                    break
                ratio = d / slack
                if ratio < min_slack_ratio:
                    min_slack_ratio = ratio
        
        # If min_slack_ratio is high, it means the activity consumes a large portion of available slack.
        # This indicates it is likely causing a bottleneck or is highly constrained.
        # We want to *penalize* this in the final priority? Or reward?
        # The strategy says: "penalizing activities that would cause immediate resource bottlenecks".
        # So high pressure -> low priority.
        # However, if an activity is *delayed* significantly, it's also bad.
        
        delay = st - current_time
        pressure[best_a] = min_slack_ratio + 0.1 * delay / max(1.0, durations[best_a])
        
        # Update state
        start_times[best_a] = st
        for r in range(m):
            d = demands[best_a, r]
            if d > 0:
                res_intervals[r].append((st, et, d))
        
        scheduled[best_a] = True
        eligible.discard(best_a)
        scheduled_count += 1
        
        if st > current_time:
            current_time = st
        
        for succ in succs[best_a]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                eligible.add(succ)
        
        # Deadline check inside loop
        if time.time() - t0 > _DEADLINE - _SAFETY:
            break

    return pressure


def priority(instance):
    t0 = time.time()
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    if n == 0:
        return np.asarray([], dtype=float)

    # Resource demand per activity (renewable resources)
    demands = instance.get("demands", None)
    if demands is None:
        demands = instance.get("demand", None)
        
    capacities = instance.get("capacities", None)
    if capacities is None:
        capacities = instance.get("capacity", None)

    if demands is not None:
        demands_arr = np.asarray(demands, dtype=float)
        if demands_arr.ndim == 1:
            demand_intensity = demands_arr
        else:
            demand_intensity = np.sum(demands_arr, axis=1)
    else:
        demand_intensity = np.ones(n, dtype=float)
        demands_arr = None

    try:
        total_float, tail, ef, project_end, head, succs, preds = _cpm_ranks(instance)
    except Exception:
        # safe fallback
        total_float = np.zeros(n)
        tail = durations.copy()
        ef = durations.copy()
        project_end = float(np.max(durations)) if n else 0.0
        head = np.zeros(n)
        preds = [_to_int_list(instance["predecessors"][i]) for i in range(n)]
        succs = [[] for _ in range(n)]
        for j in range(n):
            for i in preds[j]:
                if 0 <= i < n and i != j:
                    succs[i].append(j)

    # Guard: if we are too close to the deadline, return a simple robust score
    if time.time() - t0 > _DEADLINE - _SAFETY:
        return durations.copy()

    # Normalize each component to [0, 1]
    def norm_inv(x):
        x = np.asarray(x, dtype=float)
        lo, hi = float(np.min(x)), float(np.max(x))
        if hi - lo < 1e-12:
            return np.zeros_like(x)
        return (hi - x) / (hi - lo)

    def norm(x):
        x = np.asarray(x, dtype=float)
        lo, hi = float(np.min(x)), float(np.max(x))
        if hi - lo < 1e-12:
            return np.zeros_like(x)
        return (x - lo) / (hi - lo)

    score_float = norm_inv(total_float)   # low float -> high priority
    score_tail = norm(tail)               # long tail -> high priority
    score_dur = norm(durations)           # long duration -> high priority
    score_demand = norm(demand_intensity) # heavy resource use -> high priority
    score_head = norm(head)               # long head -> high priority (often critical)

    # Base weights
    w_float = 0.40
    w_tail = 0.20
    w_dur = 0.15
    w_demand = 0.10
    w_head = 0.15

    base_scores = (w_float * score_float
                   + w_tail * score_tail
                   + w_dur * score_dur
                   + w_demand * score_demand
                   + w_head * score_head)

    # Attempt greedy simulation to get dynamic pressure scores
    # This is the new "inspiration" component
    pressure_scores = np.zeros(n)
    try:
        if time.time() - t0 < _DEADLINE - _SAFETY - 5.0: # Need at least 5s for sim
            pressure_scores = _greedy_simulate(
                durations, demands_arr, capacities, preds, succs, base_scores, t0
            )
            # Normalize pressure: high pressure (tight fit) should lower priority
            # So we invert it: low pressure -> high priority
            # But wait, if an activity is *delayed* by resources, it's also bad.
            # The pressure metric combines tightness and delay.
            # High pressure = bad. So we want to subtract it or invert it.
            pressure_norm = norm_inv(pressure_scores) # High pressure -> Low score
            # Blend base scores with pressure adjustment
            # If simulation failed or was cut short, pressure_scores might be partial.
            # Check if most activities got a score
            if np.sum(pressure_scores > 0) > n * 0.5:
                base_scores = 0.7 * base_scores + 0.3 * pressure_norm
    except Exception:
        pass

    # Final guard
    if time.time() - t0 > _DEADLINE - _SAFETY:
        return durations.copy()

    return np.asarray(base_scores, dtype=float)
# EVOLVE-BLOCK-END
