import numpy as np
from collections import defaultdict, deque
import time

_START_TIME = time.time()
_TIMEOUT = 115.0  # Safety margin before 120s limit


def priority(instance):
    t0 = time.time()
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    if n == 0:
        return np.array([], dtype=float)
    
    # Get predecessors with defensive key handling
    preds = None
    for key in ["predecessors", "preds", "predecessors_list", "pre", "pred", "precedence"]:
        if key in instance:
            preds = instance[key]
            break
    
    if preds is None:
        preds = [[] for _ in range(n)]
    
    # Normalize preds to list of lists of ints
    if isinstance(preds, np.ndarray):
        preds = preds.tolist()
    
    # Build successor list
    succs = [[] for _ in range(n)]
    for j in range(n):
        if j < len(preds):
            p_list = preds[j]
            if p_list is None:
                continue
            for i in p_list:
                if i is not None:
                    i = int(i)
                    if 0 <= i < n and i != j:
                        if j not in succs[i]:
                            succs[i].append(j)
    
    # Topological sort
    in_degree = np.zeros(n, dtype=int)
    for j in range(n):
        for i in succs[j]:
            in_degree[i] += 1
    
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    in_deg_copy = in_degree.copy()
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in succs[u]:
            in_deg_copy[v] -= 1
            if in_deg_copy[v] == 0:
                queue.append(v)
    
    # If not all nodes in topo_order, there's a cycle; add remaining
    if len(topo_order) < n:
        remaining = [i for i in range(n) if i not in set(topo_order)]
        topo_order.extend(remaining)
    
    # Forward pass: ES
    ES = np.zeros(n, dtype=float)
    for u in topo_order:
        for v in succs[u]:
            if ES[u] + durations[u] > ES[v]:
                ES[v] = ES[u] + durations[u]
    
    project_duration = np.max(ES + durations) if n > 0 else 0
    
    # Backward pass: LS
    LS = np.full(n, project_duration, dtype=float)
    for u in reversed(topo_order):
        for v in succs[u]:
            if LS[v] - durations[u] < LS[u]:
                LS[u] = LS[v] - durations[u]
    
    total_float = LS - ES
    
    # Resource scarcity
    demands = None
    capacities = None
    for key in ["demands", "resource_demands", "demand", "resource_demand"]:
        if key in instance:
            demands = instance[key]
            break
    for key in ["capacities", "resource_capacities", "capacity", "resource_capacity"]:
        if key in instance:
            capacities = instance[key]
            break
    
    scarcity = np.zeros(n, dtype=float)
    num_resources = 0
    if demands is not None and capacities is not None:
        demands = np.asarray(demands, dtype=float)
        capacities = np.asarray(capacities, dtype=float)
        if demands.ndim == 1:
            demands = demands.reshape(-1, 1)
        if capacities.ndim == 0:
            capacities = capacities.reshape(1)
        if demands.shape[0] == n and capacities.shape[0] > 0:
            num_resources = capacities.shape[0]
            safe_cap = np.where(capacities > 0, capacities, 1.0)
            scarcity = np.sum(demands / safe_cap, axis=1)
    
    # Successor count
    succ_count = np.array([len(s) for s in succs], dtype=float)
    
    # Resource contention: for each activity, sum over resources of
    # (demand on that resource) * (number of successors that also demand that resource)
    # This identifies activities whose completion frees up resources for many successors
    contention = np.zeros(n, dtype=float)
    if demands is not None and capacities is not None and num_resources > 0:
        for j in range(n):
            for v in succs[j]:
                # Sum over resources where both j and v have non-zero demand
                shared = np.minimum(demands[j], demands[v])
                contention[j] += np.sum(shared)
    
    # Normalize
    def normalize(x):
        if len(x) == 0:
            return x
        mn, mx = float(np.min(x)), float(np.max(x))
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)
    
    tf_norm = normalize(-total_float)
    sc_norm = normalize(scarcity)
    sc_count_norm = normalize(succ_count)
    cont_norm = normalize(contention)
    
    # Weights: emphasize critical path, then resource contention, then scarcity, then successor count
    w_tf = 0.55
    w_cont = 0.20
    w_sc = 0.15
    w_succ = 0.10
    
    priority_scores = w_tf * tf_norm + w_cont * cont_norm + w_sc * sc_norm + w_succ * sc_count_norm
    
    # Check time
    if time.time() - t0 > _TIMEOUT - 2.0:
        # Return what we have
        return priority_scores
    
    return priority_scores
