# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE_SECONDS = 135.0
_START_TIME = time.time()


def _remaining_budget():
    return _DEADLINE_SECONDS - (time.time() - _START_TIME)


def priority(instance):
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demand"], dtype=float) if "demand" in instance else np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    predecessors = instance["predecessors"]

    n = len(durations)
    if n == 0:
        return np.array([])

    # Build successors and in-degree
    successors = [[] for _ in range(n)]
    in_degree = np.zeros(n, dtype=int)
    for j in range(n):
        preds = predecessors[j]
        if preds:
            in_degree[j] = len(preds)
            for p in preds:
                successors[p].append(j)

    # Topological sort (Kahn)
    queue = [j for j in range(n) if in_degree[j] == 0]
    top_order = []
    head = 0
    while head < len(queue):
        u = queue[head]; head += 1
        top_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    # If not all nodes reached (shouldn't happen), append the rest
    if len(top_order) < n:
        seen = set(top_order)
        for j in range(n):
            if j not in seen:
                top_order.append(j)

    rev_top_order = list(reversed(top_order))

    # 1) Smooth Critical Path Length from each node to end
    # Using Log-Sum-Exp to capture near-critical paths
    # cp[i] = d[i] + soft_max_{j in succ(i)} (d[j] + cp[j])
    TEMP = 3.0  # Temperature for smoothing. Lower = closer to hard max.
    cp = np.zeros(n, dtype=float)
    
    for i in rev_top_order:
        succ_vals = []
        for j in successors[i]:
            succ_vals.append(durations[j] + cp[j])
        
        if succ_vals:
            arr = np.array(succ_vals, dtype=float)
            # Numerically stable log-sum-exp
            m = np.max(arr)
            # soft_max = TEMP * log(sum(exp((arr - m) / TEMP))) + m
            # This is equivalent to TEMP * logsumexp(arr/TEMP)
            # Let's use the stable form: m + TEMP * log(sum(exp((arr-m)/TEMP)))
            smooth_max = m + TEMP * np.log(np.sum(np.exp((arr - m) / TEMP)))
        else:
            smooth_max = 0.0
            
        cp[i] = durations[i] + smooth_max

    # 2) Transitive Dependency Count (Cone Size)
    # Number of nodes that transitively depend on i
    cone_count = np.zeros(n, dtype=int)
    cone_set = [None] * n
    for i in rev_top_order:
        s = set()
        for j in successors[i]:
            s.add(j)
            if cone_set[j] is not None:
                s |= cone_set[j]
        cone_set[i] = s
        cone_count[i] = len(s)

    max_cone = int(np.max(cone_count)) if n > 0 else 1
    if max_cone == 0:
        max_cone = 1
    norm_cone = cone_count.astype(float) / max_cone

    # 3) Resource Pressure
    # We combine a duration-weighted sum of demand/capacity ratios (total resource-time pressure)
    # and the maximum single-resource ratio (bottleneck pressure).
    n_res = len(capacities)
    max_ratio = np.zeros(n, dtype=float)
    dw_ratio = np.zeros(n, dtype=float)
    for j in range(n):
        mx = 0.0
        s = 0.0
        d = durations[j]
        for k in range(n_res):
            if capacities[k] > 0:
                r = demands[j][k] / capacities[k]
                if r > mx:
                    mx = r
                s += d * r
        max_ratio[j] = mx
        dw_ratio[j] = s

    max_mr = float(np.max(max_ratio)) if n > 0 else 0.0
    if max_mr == 0:
        max_mr = 1.0
    norm_mr = max_ratio / max_mr

    max_dw = float(np.max(dw_ratio)) if n > 0 else 0.0
    if max_dw == 0:
        max_dw = 1.0
    norm_dw = dw_ratio / max_dw

    # Normalize Smooth Critical Path
    max_cp = float(np.max(cp)) if n > 0 else 0.0
    if max_cp == 0:
        max_cp = 1.0
    norm_cp = cp / max_cp

    # Combine Scores
    # Weights tuned based on Inspiration 1 (higher cone, lower res) and Insp 2 (smooth cp)
    w_cp = 1.0
    w_cone = 0.35
    w_res = 0.15
    
    # Resource component: blend weighted sum and max bottleneck
    res_component = 0.7 * norm_dw + 0.3 * norm_mr
    
    score = w_cp * norm_cp + w_cone * norm_cone + w_res * res_component

    return score
# EVOLVE-BLOCK-END
