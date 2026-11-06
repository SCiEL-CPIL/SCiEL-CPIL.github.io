import numpy as np
import time
from collections import deque

_T0 = time.time()
_DEADLINE = 150.0
_MARGIN = 2.0

def _elapsed():
    return time.time() - _T0

def _topo_order(n, preds):
    """Return topological order and successor list."""
    indeg = [0] * n
    succ = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            indeg[i] += 1
            succ[p].append(i)
    q = deque([i for i in range(n) if indeg[i] == 0])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if len(order) < n:  # cycle fallback
        order = list(range(n))
    return order, succ

def priority(instance):
    """
    Score each activity for a serial schedule-generation scheme.
    Higher score = higher priority (started earlier).
    
    Heuristic: combine
      - tail (longest path from activity to end, i.e., critical-path length)
      - head (longest path from start to activity)
      - resource contention (how scarce and contested the resources are)
      - downstream unblocking potential
    """
    if _elapsed() > _DEADLINE - _MARGIN:
        return np.asarray(instance["durations"], dtype=float)

    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    preds = instance["predecessors"]
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)

    if demands.ndim == 1:
        if n > 0:
            demands = demands.reshape(n, -1)
        else:
            demands = demands.reshape(0, 0)
    R = demands.shape[1] if demands.size else 0

    order, succ = _topo_order(n, preds)

    # Tail: longest duration path from i to end (inclusive of i's duration)
    tail = np.zeros(n, dtype=float)
    for i in reversed(order):
        best = 0.0
        for s in succ[i]:
            if tail[s] > best:
                best = tail[s]
        tail[i] = durations[i] + best

    # Head: longest duration path from start to i (inclusive)
    head = np.zeros(n, dtype=float)
    for i in order:
        best = 0.0
        for p in preds[i]:
            if head[p] > best:
                best = head[p]
        head[i] = durations[i] + best

    # Total float: critical path length - head - tail + duration
    # Activities with low total float are on or near the critical path
    cp_len = max(tail) if n > 0 else 0.0
    total_float = cp_len - head - tail + durations

    # Successor count: how many direct successors (unblocking potential)
    succ_count = np.zeros(n, dtype=float)
    for i in range(n):
        succ_count[i] = len(succ[i])

    # Resource contention: for each activity, measure how contested its resources are
    if R > 0 and capacities.size:
        cap = np.where(capacities <= 0, 1e-9, capacities)
        
        # For each resource, compute scarcity weight and contention
        # scarcity_r = 1/capacity_r (higher when capacity is low)
        # contention for activity i on resource r = demand_i_r * scarcity_r * (num_users_r / num_activities)
        
        resource_score = np.zeros(n, dtype=float)
        for r in range(R):
            cap_r = cap[r]
            if cap_r <= 0:
                continue
            scarcity = 1.0 / cap_r
            # How many activities use this resource
            users = demands[:, r]
            num_users = np.sum(users > 0)
            if num_users == 0:
                continue
            # Contention: proportional to demand, scarcity, and fraction of activities using this resource
            # Higher contention means this activity is competing for a scarce resource
            contention = users * scarcity * (num_users / n) if n > 0 else users * scarcity
            resource_score += contention
        
        # Normalize resource score
        r_max = resource_score.max() if resource_score.size else 1.0
        if r_max > 0:
            resource_norm = resource_score / r_max
        else:
            resource_norm = resource_score
    else:
        resource_norm = np.zeros(n, dtype=float)

    # Normalize tail to [0,1]
    t_max = tail.max() if tail.size else 1.0
    if t_max > 0:
        tail_norm = tail / t_max
    else:
        tail_norm = tail

    # Normalize head to [0,1]
    h_max = head.max() if head.size else 1.0
    if h_max > 0:
        head_norm = head / h_max
    else:
        head_norm = head

    # Normalize total float (invert: lower float = higher priority)
    tf_max = total_float.max() if total_float.size else 1.0
    if tf_max > 0:
        float_norm = 1.0 - total_float / tf_max
    else:
        float_norm = np.ones(n, dtype=float)

    # Normalize successor count
    sc_max = succ_count.max() if succ_count.size else 1.0
    if sc_max > 0:
        succ_norm = succ_count / sc_max
    else:
        succ_norm = succ_count

    # Combine: critical path (tail) dominates, then resource contention, then float, then unblocking
    # Tail is the strongest signal for makespan reduction
    # Resource contention helps avoid resource bottlenecks
    # Low total float indicates critical path proximity
    # Successor count helps unblock downstream work
    prio = 1.0 * tail_norm + 0.35 * resource_norm + 0.25 * float_norm + 0.10 * succ_norm + 0.05 * head_norm

    if _elapsed() > _DEADLINE - _MARGIN:
        return np.asarray(durations, dtype=float)

    return np.asarray(prio, dtype=float)
