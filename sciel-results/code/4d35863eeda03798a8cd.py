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

    # 1) Smooth critical path: log-sum-exp over successor chains
    #    This captures near-critical paths, not just the single longest.
    #    cp[i] = durations[i] + soft_max_j(durations[j] + cp[j])
    #    where soft_max uses log-sum-exp with temperature parameter.
    TEMP = 1.5  # lower = more peaked (closer to hard max), higher = smoother
    cp = np.zeros(n, dtype=float)
    for i in rev_top_order:
        succ_vals = []
        for j in successors[i]:
            succ_vals.append(durations[j] + cp[j])
        if succ_vals:
            arr = np.array(succ_vals, dtype=float)
            m = np.max(arr)
            # Smooth max: TEMP * log(sum(exp(arr/TEMP)))
            smooth_max = TEMP * (m / TEMP + np.log(np.sum(np.exp((arr - m) / TEMP))))
        else:
            smooth_max = 0.0
        cp[i] = durations[i] + smooth_max

    # 2) Transitive dependency count: number of nodes that transitively depend on i
    #    (i.e., size of the subproject that is blocked until i finishes).
    #    Computed by reverse topo: dep[i] = 1 + sum(dep[s] for s in succ(i))
    #    but that double counts shared descendants. For a cheap, monotone proxy
    #    of "unlocks a lot of work", use the count of direct+indirect successors
    #    via a union count is expensive; instead use longest-path *width* proxy:
    #    number of immediate successors is weak, so use the count of nodes in
    #    the forward cone via a boolean-union in reverse topo (n=30 is tiny).
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

    # 2b) Downward slack: how much "float" activity i has relative to the
    #     longest path from its start to project end. Activities with less
    #     slack (i.e., closer to critical) should be prioritized.
    #     We compute the longest path from i to any sink (which is cp[i] above,
    #     but without the smooth max—use hard max for slack).
    hard_cp = np.zeros(n, dtype=float)
    for i in rev_top_order:
        mx = 0.0
        for j in successors[i]:
            v = durations[j] + hard_cp[j]
            if v > mx:
                mx = v
        hard_cp[i] = durations[i] + mx

    # Total project makespan lower bound (longest path from any source to sink)
    total_lb = float(np.max(hard_cp)) if n > 0 else 0.0

    # Upward slack: longest path from any source to i (excluding i's own duration)
    # Compute in forward topo order
    up_cp = np.zeros(n, dtype=float)
    for i in top_order:
        mx = 0.0
        for p in predecessors[i]:
            v = durations[p] + up_cp[p]
            if v > mx:
                mx = v
        up_cp[i] = mx  # longest path ending at i's start

    # Slack[i] = total_lb - (up_cp[i] + hard_cp[i])
    # Normalize: less slack = higher priority
    slack = total_lb - (up_cp + hard_cp)
    max_slack = float(np.max(slack)) if n > 0 else 0.0
    if max_slack > 0:
        norm_slack = 1.0 - slack / max_slack  # 1 for zero slack, 0 for max slack
    else:
        norm_slack = np.ones(n, dtype=float)

    # 3) Resource pressure: duration-weighted bottleneck and sum
    #    Weighting by duration emphasizes activities that occupy resources longer.
    n_res = len(capacities)
    max_ratio = np.zeros(n, dtype=float)
    sum_ratio = np.zeros(n, dtype=float)
    dur_weighted_max = np.zeros(n, dtype=float)
    dur_weighted_sum = np.zeros(n, dtype=float)
    for j in range(n):
        mx = 0.0
        s = 0.0
        dw_mx = 0.0
        dw_s = 0.0
        for k in range(n_res):
            if capacities[k] > 0:
                r = demands[j][k] / capacities[k]
                if r > mx:
                    mx = r
                s += r
                # duration-weighted: how much total resource-time this activity consumes relative to capacity
                dw_r = durations[j] * r
                if dw_r > dw_mx:
                    dw_mx = dw_r
                dw_s += dw_r
        max_ratio[j] = mx
        sum_ratio[j] = s
        dur_weighted_max[j] = dw_mx
        dur_weighted_sum[j] = dw_s

    max_mr = float(np.max(max_ratio)) if n > 0 else 0.0
    if max_mr == 0:
        max_mr = 1.0
    norm_mr = max_ratio / max_mr

    max_sr = float(np.max(sum_ratio)) if n > 0 else 0.0
    if max_sr == 0:
        max_sr = 1.0
    norm_sr = sum_ratio / max_sr

    max_dwm = float(np.max(dur_weighted_max)) if n > 0 else 0.0
    if max_dwm == 0:
        max_dwm = 1.0
    norm_dwm = dur_weighted_max / max_dwm

    max_dws = float(np.max(dur_weighted_sum)) if n > 0 else 0.0
    if max_dws == 0:
        max_dws = 1.0
    norm_dws = dur_weighted_sum / max_dws

    # Normalize critical path
    max_cp = float(np.max(cp)) if n > 0 else 0.0
    if max_cp == 0:
        max_cp = 1.0
    norm_cp = cp / max_cp

    # Combine. Primary: smooth critical path.
    # Secondary: transitive unlock (front-load gatekeepers).
    # Tertiary: duration-weighted resource pressure.
    # Quaternary: slack-based priority (less slack = higher priority).
    w_cp = 1.0
    w_cone = 0.25
    w_res = 0.12
    w_slack = 0.30
    # Blend raw ratio and duration-weighted for resource signal
    res_signal = 0.4 * norm_mr + 0.2 * norm_sr + 0.3 * norm_dwm + 0.1 * norm_dws
    score = w_cp * norm_cp + w_cone * norm_cone + w_res * res_signal + w_slack * norm_slack

    return score
# EVOLVE-BLOCK-END
