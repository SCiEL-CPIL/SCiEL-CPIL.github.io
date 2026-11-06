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
    
    Heuristic: 
    1. Total Float (Slack): Activities with less slack (closer to 0) are more critical.
    2. Global Resource Pressure: Activities consuming globally scarce resources are prioritized.
    3. Duration: Longer activities are prioritized (LPT tie-breaker).
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
    
    # 1. Calculate Earliest Start (ES) and Earliest Finish (EF)
    es = np.zeros(n, dtype=float)
    ef = np.zeros(n, dtype=float)
    for i in order:
        for p in preds[i]:
            if ef[p] > es[i]:
                es[i] = ef[p]
        ef[i] = es[i] + durations[i]
        
    project_duration = np.max(ef) if n > 0 else 0.0

    # 2. Calculate Latest Finish (LF) and Latest Start (LS)
    # LF[i] = min(LS[s]) for all successors s. If no successors, LF = project_duration.
    lf = np.full(n, project_duration, dtype=float)
    for i in reversed(order):
        for s in succ[i]:
            if lf[s] < lf[i]:
                lf[i] = lf[s]
    
    ls = lf - durations
    # Total Float = LS - ES (or LF - EF)
    float_slack = ls - es
    
    # Normalize Float: Inverse float, so higher score = less slack (more critical)
    # If float is 0, it's critical. We want high priority for low float.
    # Map [min_float, max_float] to [1, 0] or similar? 
    # Actually, standard priority is: High Tail, Low Float.
    # Let's use: (max_float - float_slack) / max_float if max_float > 0 else 0
    # Or simply normalize inverse float.
    # If all floats are 0 (all critical), this term is 0 for all, which is fine.
    
    f_max = float_slack.max() if n > 0 else 1.0
    if f_max > 0:
        # Activities with 0 float get score 1.0. Activities with max float get 0.0.
        float_norm = 1.0 - (float_slack / f_max)
    else:
        float_norm = np.ones(n, dtype=float) # All critical, all equal

    # 3. Global Resource Pressure
    # For each resource, compute TotalDemand / Capacity.
    # This represents how "tight" the resource is globally.
    # Then for each activity, sum (demand_i_r * scarcity_r).
    
    if R > 0 and capacities.size > 0:
        cap = np.where(capacities <= 0, 1e-9, capacities)
        total_demand_per_resource = np.sum(demands, axis=0)
        # Scarcity factor: How much total demand exceeds or uses up capacity
        # If TotalDemand / Cap > 1, it's definitely bottlenecked.
        # We want to weight activities that use these bottlenecked resources higher.
        scarcity = total_demand_per_resource / cap
        
        # For each activity, sum of (demand_i_r * scarcity_r)
        # This gives higher weight to activities using scarce resources.
        resource_pressure = demands @ scarcity
        
        rp_max = resource_pressure.max() if n > 0 else 1.0
        if rp_max > 0:
            resource_norm = resource_pressure / rp_max
        else:
            resource_norm = np.zeros(n, dtype=float)
    else:
        resource_norm = np.zeros(n, dtype=float)

    # 4. Duration Normalization (LPT component)
    d_max = durations.max() if n > 0 else 1.0
    if d_max > 0:
        dur_norm = durations / d_max
    else:
        dur_norm = np.zeros(n, dtype=float)

    # Combine Heuristics
    # Weighting:
    # Criticality (Float) is the strongest predictor in RCPSP.
    # Resource Pressure is the second strongest for resource-constrained problems.
    # Duration is a weak tie-breaker.
    
    # Adjust weights based on the nature of the instance?
    # For now, fixed weights that favor critical path and scarcity.
    w_float = 1.0
    w_resource = 0.6
    w_duration = 0.1
    
    prio = w_float * float_norm + w_resource * resource_norm + w_duration * dur_norm

    if _elapsed() > _DEADLINE - _MARGIN:
        return np.asarray(instance["durations"], dtype=float)

    return np.asarray(prio, dtype=float)
