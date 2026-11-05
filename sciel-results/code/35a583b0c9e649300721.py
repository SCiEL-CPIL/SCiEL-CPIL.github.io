"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(instance):
    """
    Returns a priority score per activity for the serial SGS.
    Higher priority = scheduled earlier.
    
    Strategy: Combine total float (urgency), resource scarcity pressure,
    and successor count into a composite score.
    """
    start_time = time.time()
    DEADLINE = 110.0  # safety margin before 120s limit

    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)

    # Parse predecessors
    # Try different possible key names
    preds_key = None
    for k in ["predecessors", "preds", "predecessor", "pred"]:
        if k in instance:
            preds_key = k
            break

    if preds_key is None:
        # Fallback: longest duration
        return durations.copy()

    predecessors = instance[preds_key]

    # Build adjacency and in-degree
    # predecessors[i] = list of predecessor indices for activity i
    adj = [[] for _ in range(n)]  # adj[i] = successors of i
    in_degree = np.zeros(n, dtype=int)

    for i in range(n):
        preds = predecessors[i] if isinstance(predecessors, list) else predecessors[i]
        if preds is None:
            preds = []
        for p in preds:
            adj[p].append(i)
            in_degree[i] += 1

    # Forward pass (earliest start)
    es = np.zeros(n, dtype=float)
    # Topological order
    from collections import deque
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo = []
    in_deg_copy = in_degree.copy()
    while queue:
        u = queue.popleft()
        topo.append(u)
        for v in adj[u]:
            in_deg_copy[v] -= 1
            if in_deg_copy[v] == 0:
                queue.append(v)
    # If not all nodes in topo (cycle), use all
    if len(topo) < n:
        topo = list(range(n))

    for u in topo:
        for v in adj[u]:
            es[v] = max(es[v], es[u] + durations[u])

    # Project makespan (unconstrained)
    project_len = float(np.max(es + durations)) if n > 0 else 0.0

    # Backward pass (latest start)
    ls = np.full(n, project_len, dtype=float)
    # Reverse topological
    for u in reversed(topo):
        for v in adj[u]:
            ls[u] = min(ls[u], ls[v] - durations[u])

    # Total float
    tf = ls - es  # total float per activity
    # Normalize float to [0,1] range (lower float = more urgent)
    tf_min = float(np.min(tf)) if n > 0 else 0.0
    tf_max = float(np.max(tf)) if n > 0 else 1.0
    tf_range = tf_max - tf_min if tf_max > tf_min else 1.0
    norm_float = (tf - tf_min) / tf_range  # 0 = zero float (most urgent), 1 = max float

    # Resource scarcity: for each resource, compute total demand vs capacity
    # demands[i] = list of resource demands for activity i
    # capacities = list of resource capacities
    demands_key = None
    for k in ["demands", "resource_demands", "resources", "demand"]:
        if k in instance:
            demands_key = k
            break
    caps_key = None
    for k in ["capacities", "resource_capacities", "capacity", "caps"]:
        if k in instance:
            caps_key = k
            break

    resource_pressure = np.zeros(n, dtype=float)

    if demands_key is not None and caps_key is not None:
        demands = instance[demands_key]
        capacities = instance[caps_key]
        num_resources = len(capacities)

        # For each resource, compute average utilization pressure
        # An activity's pressure on resource r = demand[i][r] / capacity[r]
        # Weighted by duration (longer activities tie up resources longer)
        for r in range(num_resources):
            cap_r = float(capacities[r])
            if cap_r <= 0:
                continue
            for i in range(n):
                if isinstance(demands, list):
                    d = demands[i]
                    if d is None or r >= len(d):
                        continue
                    demand_ir = float(d[r])
                else:
                    # numpy array
                    demand_ir = float(demands[i][r])
                # Pressure = (demand/capacity) * duration
                # This penalizes activities that use a lot of a scarce resource for a long time
                resource_pressure[i] += (demand_ir / cap_r) * durations[i]

        # Normalize resource pressure
        rp_max = float(np.max(resource_pressure)) if n > 0 else 1.0
        if rp_max > 0:
            resource_pressure = resource_pressure / rp_max
    else:
        # No resource info, use duration as proxy
        d_max = float(np.max(durations)) if n > 0 else 1.0
        if d_max > 0:
            resource_pressure = durations / d_max

    # Successor count (how many activities depend on this one)
    succ_count = np.array([len(adj[i]) for i in range(n)], dtype=float)
    sc_max = float(np.max(succ_count)) if n > 0 else 1.0
    if sc_max > 0:
        succ_count = succ_count / sc_max

    # Composite priority:
    # We want HIGH priority for: low float (urgent), high resource pressure, many successors
    # priority = w1 * (1 - norm_float) + w2 * resource_pressure + w3 * succ_count
    # Using weights that emphasize urgency and resource scarcity
    w1 = 0.50  # total float (urgency)
    w2 = 0.35  # resource scarcity
    w3 = 0.15  # successor count (critical path)

    priority_scores = w1 * (1.0 - norm_float) + w2 * resource_pressure + w3 * succ_count

    # Add a small duration component to break ties (longer activities first among equals)
    d_max = float(np.max(durations)) if n > 0 else 1.0
    if d_max > 0:
        priority_scores += 0.05 * (durations / d_max)

    return priority_scores
# EVOLVE-BLOCK-END
