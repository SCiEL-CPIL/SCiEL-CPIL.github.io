"""Direct approach for resource-constrained project scheduling priority."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

_DEADLINE = 150.0
_START_TIME = None


def _soft_max(values, beta=10.0):
    """Smooth max via log-sum-exp."""
    if len(values) == 0:
        return 0.0
    v = np.asarray(values, dtype=float)
    v_max = np.max(v)
    if v_max < -1e10:
        return 0.0
    shifted = (v - v_max) * beta
    # Clip to avoid overflow
    shifted = np.clip(shifted, -50.0, 50.0)
    exp_shifted = np.exp(shifted)
    log_sum = np.log(np.sum(exp_shifted)) / beta
    return v_max + log_sum


def priority(instance):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

    # Deadline guard
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE - 2.0:
        n = len(instance.get("durations", []))
        return np.zeros(n)

    # Extract instance data
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)

    if "demand" in instance:
        demands = np.asarray(instance["demand"], dtype=float)
    elif "demands" in instance:
        demands = np.asarray(instance["demands"], dtype=float)
    else:
        demands = np.zeros((n, 0))

    if "capacity" in instance:
        capacities = np.asarray(instance["capacity"], dtype=float)
    elif "capacities" in instance:
        capacities = np.asarray(instance["capacities"], dtype=float)
    else:
        capacities = np.array([float('inf')])

    if "predecessors" in instance:
        predecessors = instance["predecessors"]
    elif "predecessor_lists" in instance:
        predecessors = instance["predecessor_lists"]
    else:
        predecessors = [[] for _ in range(n)]

    # Normalize shapes
    if demands.ndim == 1:
        demands = demands.reshape(n, 1)
        capacities = capacities.reshape(1)
    m = len(capacities)
    if demands.shape[1] < m:
        pad = np.zeros((n, m - demands.shape[1]))
        demands = np.hstack([demands, pad])
    elif demands.shape[1] > m:
        demands = demands[:, :m]
        capacities = capacities[:m]
        m = len(capacities)

    # Build graph
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        for p in predecessors[i]:
            if p >= 0 and p < n:
                in_degree[i] += 1
                successors[p].append(i)

    # Topological order
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    temp_in_degree = in_degree[:]
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for s in successors[node]:
            temp_in_degree[s] -= 1
            if temp_in_degree[s] == 0:
                queue.append(s)

    if len(topo_order) < n:
        in_topo = set(topo_order)
        for i in range(n):
            if i not in in_topo:
                topo_order.append(i)

    # Compute head (earliest start) using smooth max surrogate
    head = np.zeros(n)
    for node in topo_order:
        pred_paths = []
        for p in predecessors[node]:
            if p >= 0 and p < n:
                pred_paths.append(head[p] + durations[p])
        if pred_paths:
            head[node] = _soft_max(pred_paths, beta=15.0)
        else:
            head[node] = 0.0

    # Compute tail (latest finish distance) using smooth max surrogate
    tail = np.zeros(n)
    for node in reversed(topo_order):
        succ_paths = []
        for s in successors[node]:
            succ_paths.append(durations[s] + tail[s])
        if succ_paths:
            tail[node] = _soft_max(succ_paths, beta=15.0)
        else:
            tail[node] = 0.0

    # Critical path length (smooth)
    all_cp = head + durations + tail
    cp_length = _soft_max(all_cp.tolist(), beta=15.0) if n > 0 else 0.0

    # Total float (slack) - using smooth cp
    total_float = cp_length - head - durations - tail
    total_float = np.maximum(total_float, 0.0)

    # Resource intensity features
    max_demand_ratio = np.zeros(n)
    for r in range(m):
        if capacities[r] > 0 and np.isfinite(capacities[r]):
            ratio = demands[:, r] / capacities[r]
            max_demand_ratio = np.maximum(max_demand_ratio, ratio)

    # Weighted demand (bottleneck awareness)
    weighted_demand = np.zeros(n)
    for r in range(m):
        if capacities[r] > 0 and np.isfinite(capacities[r]):
            weighted_demand += demands[:, r] / capacities[r] * durations

    dur = durations
    succ_count = np.array([len(s) for s in successors], dtype=float)
    pred_count = np.array([len(p) for p in predecessors], dtype=float)

    # Deadline guard after expensive computation
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE - 1.0:
        # Fallback: use only head/tail based priority
        def normalize(x):
            x = np.asarray(x, dtype=float)
            x_max = x.max()
            x_min = x.min()
            rng = x_max - x_min
            if rng > 1e-12:
                return (x - x_min) / rng
            return np.zeros_like(x)
        scores = 0.6 * (1.0 - normalize(total_float)) + 0.4 * normalize(tail)
        return scores

    # Normalize features
    def normalize(x):
        x = np.asarray(x, dtype=float)
        x_max = x.max()
        x_min = x.min()
        rng = x_max - x_min
        if rng > 1e-12:
            return (x - x_min) / rng
        return np.zeros_like(x)

    # Priority scoring with stronger emphasis on smooth critical-path features
    norm_float_inv = 1.0 - normalize(total_float)
    norm_weighted_demand = normalize(weighted_demand)
    norm_duration = normalize(dur)
    norm_tail = normalize(tail)
    norm_succ = normalize(succ_count)
    norm_max_ratio = normalize(max_demand_ratio)
    norm_head = normalize(head)

    # Composite priority score - emphasize smooth float, tail, and head
    scores = (0.40 * norm_float_inv +
              0.20 * norm_tail +
              0.15 * norm_weighted_demand +
              0.10 * norm_duration +
              0.08 * norm_head +
              0.05 * norm_succ +
              0.02 * norm_max_ratio)

    return scores
# EVOLVE-BLOCK-END
