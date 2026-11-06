import numpy as np
from collections import deque
import time

def priority(instance):
    t0 = time.time()
    DEADLINE = 115.0  # safety margin before 120s limit

    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    if n == 0:
        return np.array([], dtype=float)

    # Extract predecessors
    preds = None
    for key in ["predecessors", "preds", "predecessors_list", "pre", "pred", "precedence"]:
        if key in instance:
            preds = instance[key]
            break
    if preds is None:
        preds = [[] for _ in range(n)]
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

    # Extract resource data
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

    demands_arr = None
    capacities_arr = None
    if demands is not None and capacities is not None:
        demands_arr = np.asarray(demands, dtype=float)
        capacities_arr = np.asarray(capacities, dtype=float)
        if demands_arr.ndim == 1:
            demands_arr = demands_arr.reshape(-1, 1)
        if capacities_arr.ndim == 0:
            capacities_arr = capacities_arr.reshape(1)
        if demands_arr.shape[0] != n:
            demands_arr = None
            capacities_arr = None
        elif capacities_arr.ndim == 1 and capacities_arr.size != demands_arr.shape[1]:
            if capacities_arr.size == 1:
                capacities_arr = np.tile(capacities_arr, (demands_arr.shape[1],))
            elif demands_arr.shape[1] == 1:
                demands_arr = np.tile(demands_arr, (1, capacities_arr.size))
            else:
                demands_arr = None
                capacities_arr = None
        if demands_arr is not None:
            capacities_arr = np.where(capacities_arr > 0, capacities_arr, 1.0)

    # Forward pass for ES
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
    if len(topo_order) < n:
        remaining = [i for i in range(n) if i not in set(topo_order)]
        topo_order.extend(remaining)

    ES = np.zeros(n, dtype=float)
    for u in topo_order:
        for v in succs[u]:
            if ES[u] + durations[u] > ES[v]:
                ES[v] = ES[u] + durations[u]

    project_duration = float(np.max(ES + durations)) if n > 0 else 0.0

    # Backward pass for LS
    LS = np.full(n, project_duration, dtype=float)
    for u in reversed(topo_order):
        for v in succs[u]:
            if LS[v] - durations[u] < LS[u]:
                LS[u] = LS[v] - durations[u]
    total_float = LS - ES

    # Compute resource contention: for each activity, sum over successors of shared resource demand
    contention = np.zeros(n, dtype=float)
    if demands_arr is not None and capacities_arr is not None:
        for j in range(n):
            for v in succs[j]:
                shared = np.minimum(demands_arr[j], demands_arr[v])
                contention[j] += np.sum(shared)

    # Successor count
    succ_count = np.array([len(succs[i]) for i in range(n)], dtype=float)

    # Resource scarcity
    scarcity = np.zeros(n, dtype=float)
    if demands_arr is not None and capacities_arr is not None:
        scarcity = np.sum(demands_arr / capacities_arr, axis=1)

    # Duration
    d_score = durations.copy()

    # Normalize helper
    def normalize(x):
        if len(x) == 0:
            return x
        mn, mx = float(np.min(x)), float(np.max(x))
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)

    # Normalize all components
    tf_norm = normalize(-total_float)
    cont_norm = normalize(contention)
    sc_norm = normalize(scarcity)
    succ_norm = normalize(succ_count)
    d_norm = normalize(d_score)

    # Weights: emphasize critical path, then resource contention, then scarcity, then successor count
    w_tf = 0.45
    w_cont = 0.25
    w_sc = 0.15
    w_succ = 0.10
    w_d = 0.05

    priority_scores = (w_tf * tf_norm + w_cont * cont_norm + w_sc * sc_norm +
                       w_succ * succ_norm + w_d * d_norm)

    # Add a small tiebreaker: lower index for stability
    priority_scores += np.arange(n) * 1e-10

    if time.time() - t0 > DEADLINE:
        return priority_scores

    return priority_scores
