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

    # Resource contention: model how much each activity's demands actually constrain
    # the schedule. For each resource, compute the "blocking potential" — the extent to
    # which starting this activity would delay others due to resource scarcity.
    # Use a weighted sum where weight reflects both scarcity and the number of
    # downstream activities that also need that resource (bottleneck centrality).
    if R > 0 and capacities.size:
        cap = np.where(capacities <= 0, 1e-9, capacities)
        
        # For each resource r, compute:
        #   scarcity_r = 1 / capacity_r
        #   downstream_pressure_r[i] = sum of durations of all successors of i (transitive)
        #     that also demand resource r, weighted by their own demand on r
        # This captures: if i uses a scarce resource that many downstream tasks also need,
        # delaying i cascades badly.
        
        resource_score = np.zeros(n, dtype=float)
        
        # Precompute transitive successor sets efficiently via BFS/DFS from each node
        # For 30 activities, we can afford O(n^2 * R)
        # trans_succ[i] = set of all transitive successors of i
        trans_succ = [set() for _ in range(n)]
        for i in reversed(order):
            for s in succ[i]:
                trans_succ[i].add(s)
                trans_succ[i].update(trans_succ[s])
        
        for r in range(R):
            cap_r = cap[r]
            if cap_r <= 0:
                continue
            scarcity = 1.0 / cap_r
            users = demands[:, r]
            num_users = np.sum(users > 0)
            if num_users == 0:
                continue
            
            # For each activity i, compute downstream pressure on resource r
            # = sum over transitive successors j of i: (demand[j][r] * tail[j]) / capacity_r
            # This measures: if we delay i, how much critical-path work on resource r is affected
            for i in range(n):
                if users[i] <= 0:
                    continue
                pressure = 0.0
                for j in trans_succ[i]:
                    if users[j] > 0:
                        pressure += users[j] * tail[j]
                # Normalize by total demand on this resource among all users
                total_demand_r = np.sum(users)
                if total_demand_r > 0:
                    pressure /= total_demand_r
                resource_score[i] += users[i] * scarcity * pressure
        
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

    # Combine: use a multiplicative interaction between critical-path proximity and
    # resource bottleneck centrality. Activities that are both on the critical path
    # AND gate scarce resources for downstream work get the highest priority.
    # This models the serial SGS decision: starting an activity early is most valuable
    # if it's on the critical path AND its resource use would otherwise block others.
    
    # Critical path weight: activities with low total float (near critical path)
    # get high weight. Use exponential decay for smoothness.
    if tf_max > 0:
        # float_norm is already 1 - total_float/tf_max, so 1 = on critical path
        cp_weight = float_norm
    else:
        cp_weight = np.ones(n, dtype=float)
    
    # Resource bottleneck centrality: how much this activity's resource use gates
    # downstream critical work. Higher = more central bottleneck.
    bottleneck = resource_norm
    
    # Interaction: critical path activities that are also resource bottlenecks
    # get boosted. Non-critical activities get less boost from bottleneck status.
    interaction = cp_weight * bottleneck
    
    # Final priority: dominated by tail (makespan reduction), with a strong
    # interaction term for resource-constrained critical path activities
    prio = 0.6 * tail_norm + 0.4 * interaction + 0.1 * float_norm
    
    # Ensure monotonicity: higher tail should generally mean higher priority
    # Add a small head component to break ties and handle start-level activities
    prio += 0.05 * head_norm

    if _elapsed() > _DEADLINE - _MARGIN:
        return np.asarray(durations, dtype=float)

    return np.asarray(prio, dtype=float)
