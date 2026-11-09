# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from collections import deque

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

    # 1) Critical path length from each node to end (longest remaining chain)
    cp = np.zeros(n, dtype=float)
    for i in rev_top_order:
        max_succ = 0.0
        for j in successors[i]:
            val = durations[j] + cp[j]
            if val > max_succ:
                max_succ = val
        cp[i] = durations[i] + max_succ

    # 2) Smoothed Critical Path Length (Soft-Max)
    T = 1.0
    scpl = np.zeros(n, dtype=float)
    for i in rev_top_order:
        if not successors[i]:
            scpl[i] = durations[i]
        else:
            succ_vals = [durations[j] + scpl[j] for j in successors[i]]
            m = max(succ_vals)
            # log-sum-exp for numerical stability
            sum_exp = sum(math.exp((x - m) / T) for x in succ_vals)
            scpl[i] = durations[i] + m + T * math.log(sum_exp)

    # 3) Earliest Start times and Slack
    es = np.zeros(n, dtype=float)
    for i in top_order:
        if predecessors[i]:
            es[i] = max(es[p] + durations[p] for p in predecessors[i])
    
    project_length = max(es[i] + durations[i] for i in range(n))
    slack = np.zeros(n, dtype=float)
    for i in range(n):
        slack[i] = (es[i] + durations[i] + cp[i]) - project_length

    # 4) Resource Pressure
    n_res = len(capacities)
    max_ratio = np.zeros(n, dtype=float)
    sum_ratio = np.zeros(n, dtype=float)
    for j in range(n):
        mx = 0.0
        s = 0.0
        for k in range(n_res):
            if capacities[k] > 0:
                r = demands[j][k] / capacities[k]
                if r > mx:
                    mx = r
                s += r
        max_ratio[j] = mx
        sum_ratio[j] = s

    # 5) Resource Weighted Tail (Chain of resource-heavy activities)
    #    res_tail[i] = res_max[i] + max(res_tail[s] for s in successors[i])
    res_tail = np.zeros(n, dtype=float)
    for i in rev_top_order:
        res_tail[i] = max_ratio[i]
        for j in successors[i]:
            val = res_tail[j]
            if val > res_tail[i]:
                res_tail[i] = val
        # Note: The definition above is slightly different from the inspiration's "sum along path".
        # Inspiration used: res_tail[u] = res_max[u] + max(res_tail[v] for v in successors[u])
        # Let's correct to match the inspiration's logic which sums the bottleneck along the chain.
    res_tail = np.zeros(n, dtype=float)
    for i in rev_top_order:
        res_tail[i] = max_ratio[i]
        for j in successors[i]:
            val = max_ratio[i] + res_tail[j]
            if val > res_tail[i]:
                res_tail[i] = val

    # 6) Transitive dependency count (Cone count)
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

    # Normalizations
    def norm_min_max(arr):
        mn, mx = float(arr.min()), float(arr.max())
        if mx - mn < 1e-10:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    def norm_max(arr):
        mx = float(np.max(arr))
        if mx < 1e-10:
            return np.zeros_like(arr)
        return arr / mx

    norm_cp = norm_max(cp)
    norm_scpl = norm_max(scpl)
    norm_slack = norm_min_max(slack) # Lower slack is better (more critical)
    norm_res_max = norm_max(max_ratio)
    norm_res_sum = norm_max(sum_ratio)
    norm_res_tail = norm_max(res_tail)

    # Combine features
    # Weights tuned based on the inspiration's high-scoring structure but simplified
    w_cp = 1.0
    w_scpl = 0.5
    w_slack = 0.3
    w_res_max = 0.4
    w_res_sum = 0.2
    w_res_tail = 0.5
    w_cone = 0.1

    score = (
        w_cp * norm_cp +
        w_scpl * norm_scpl +
        w_slack * (1.0 - norm_slack) + # Invert slack: low slack -> high priority
        w_res_max * norm_res_max +
        w_res_sum * norm_res_sum +
        w_res_tail * norm_res_tail +
        w_cone * norm_cone
    )

    return score
# EVOLVE-BLOCK-END
