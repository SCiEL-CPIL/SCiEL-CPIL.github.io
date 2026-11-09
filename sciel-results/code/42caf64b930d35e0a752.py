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

    # 1) Forward Pass: Earliest Start (ES) and Critical Path length (cp)
    # cp[i] is the longest path from i to the end (including i's duration)
    # ES[i] is the earliest start time of i
    es = np.zeros(n, dtype=float)
    cp = np.zeros(n, dtype=float)
    
    # Forward pass for ES
    for i in top_order:
        if predecessors[i]:
            max_ef = 0.0
            for p in predecessors[i]:
                ef_p = es[p] + durations[p]
                if ef_p > max_ef:
                    max_ef = ef_p
            es[i] = max_ef
        # else es[i] remains 0

    # Backward pass for cp (Longest path to end)
    for i in rev_top_order:
        max_succ_cp = 0.0
        for j in successors[i]:
            if cp[j] > max_succ_cp:
                max_succ_cp = cp[j]
        cp[i] = durations[i] + max_succ_cp

    # Theoretical Makespan (CPM)
    makespan_cpm = float(np.max(cp)) if n > 0 else 0.0
    if makespan_cpm == 0:
        makespan_cpm = 1.0

    # 2) Backward Pass: Latest Finish (LF) and Latest Start (LS)
    # LF[i] is the latest finish time of i without delaying the project
    # We assume the project duration is makespan_cpm
    lf = np.full(n, makespan_cpm, dtype=float)
    for i in rev_top_order:
        if successors[i]:
            min_ls = np.inf
            for j in successors[i]:
                ls_j = lf[j] - durations[j]
                if ls_j < min_ls:
                    min_ls = ls_j
            lf[i] = min_ls
        # else lf[i] remains makespan_cpm

    ls = lf - durations

    # 3) Total Slack (TS)
    # TS = LS - ES
    slack = ls - es
    # Ensure non-negative (should be, but floating point might cause tiny negatives)
    slack = np.maximum(slack, 0.0)
    
    max_slack = float(np.max(slack)) if n > 0 else 0.0
    if max_slack == 0:
        max_slack = 1.0
    # Normalize slack: High slack = Low priority. We want High Priority for Low Slack.
    # norm_slack_inv = 1 - (slack / max_slack)
    norm_slack_inv = 1.0 - (slack / max_slack)

    # 4) Soft Critical Path (LSE)
    # LSE provides a smooth approximation of the max operation
    LSE_T = 1.0  # Temperature
    soft_cp = np.zeros(n, dtype=float)
    for i in rev_top_order:
        vals = [durations[j] + soft_cp[j] for j in successors[i]]
        if vals:
            m = max(vals)
            if m > 1e-12:
                # LSE = m + T * log(sum(exp((v-m)/T)))
                # To avoid overflow, we subtract m
                exp_terms = np.exp((np.array(vals) - m) / LSE_T)
                lse = m + LSE_T * np.log(np.sum(exp_terms))
            else:
                lse = 0.0
        else:
            lse = 0.0
        soft_cp[i] = durations[i] + lse

    max_scp = float(np.max(soft_cp)) if n > 0 else 0.0
    if max_scp == 0:
        max_scp = 1.0
    norm_scp = soft_cp / max_scp

    # 5) Transitive Dependency Count (Cone)
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

    # 6) Resource Pressure
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

    max_mr = float(np.max(max_ratio)) if n > 0 else 0.0
    if max_mr == 0:
        max_mr = 1.0
    norm_mr = max_ratio / max_mr

    max_sr = float(np.max(sum_ratio)) if n > 0 else 0.0
    if max_sr == 0:
        max_sr = 1.0
    norm_sr = sum_ratio / max_sr

    # Normalize Hard CP
    max_cp = float(np.max(cp)) if n > 0 else 0.0
    if max_cp == 0:
        max_cp = 1.0
    norm_cp = cp / max_cp

    # Combine Scores
    # Weights:
    # CP is the primary driver.
    # Slack is a strong secondary driver (MINSLACK is a classic rule).
    # Soft CP adds robustness.
    # Cone count helps with structural importance.
    # Resource pressure is a tie-breaker.
    
    w_cp = 1.0
    w_slack = 0.8
    w_scp = 0.3
    w_cone = 0.2
    w_res = 0.1

    score = (w_cp * norm_cp +
             w_slack * norm_slack_inv +
             w_scp * norm_scp +
             w_cone * norm_cone +
             w_res * (0.7 * norm_mr + 0.3 * norm_sr))

    return score
# EVOLVE-BLOCK-END
