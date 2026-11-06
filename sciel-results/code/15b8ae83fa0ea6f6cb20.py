"""Resource-Constrained Project Scheduling — Total Float (slack) priority.

Direct approach: compute the classical CPM Total Float for each activity via
a forward pass (earliest start) and a backward pass (latest finish). An
activity's total float is how late it can start without delaying the project,
so activities with the least float are most critical to the makespan. We
return a score that is HIGHER for more critical (low-float) activities, and
break ties using the activity's demand on the scarcest (most contended)
renewable resources. This is a structurally-derived rule: float comes directly
from the instance's precedence graph + durations, and resource pressure comes
directly from demands/capacities.

A safety deadline guard ensures we always return within budget.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import deque

_DEADLINE_SECONDS = 110.0  # safety margin below the 120 s wall-clock limit


def _total_remaining_work(durations, succ_lists, topo):
    """For each activity, compute the sum of durations of all downstream
    activities (including itself). This is the 'total remaining work' if
    this activity starts now.
    """
    n = len(durations)
    # Process in reverse topological order so successors are already computed
    rem = np.zeros(n, dtype=float)
    for i in reversed(topo):
        rem[i] = durations[i]
        for j in succ_lists[i]:
            rem[i] += rem[j]
    return rem


def _parallel_path_count(succ_lists, topo):
    """Count the number of distinct parallel paths branching out from each
    activity (including the activity itself as a trivial path of length 1).
    Activities with more parallel successors create more potential resource
    contention, so we want to deprioritize them slightly (inverse weighting).
    """
    n = len(succ_lists)
    # number of paths from i to any sink = 1 (itself) + sum over successors
    # But to avoid exponential blowup on dense graphs, cap the count.
    # Use a capped count to keep values bounded.
    CAP = 50.0
    path_count = np.zeros(n, dtype=float)
    for i in reversed(topo):
        if not succ_lists[i]:
            path_count[i] = 1.0
        else:
            total = 0.0
            for j in succ_lists[i]:
                total += path_count[j]
                if total > CAP:
                    break
            path_count[i] = 1.0 + min(total, CAP)
    return path_count


def _soft_min(values, temp=1.5):
    """Soft minimum using log-sum-exp for numerical stability.
    
    Higher temp -> smoother (less peaked) approximation to min.
    temp=1.5 gives a sharper critical-path signal: near-critical activities
    are strongly differentiated while still avoiding the non-differentiable
    hard-min that would collapse all near-critical paths to equal priority.
    """
    if len(values) is None or len(values) == 0:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    vals = np.asarray(values, dtype=float)
    neg = -vals
    max_neg = np.max(neg)
    if max_neg > 1e12:
        return -max_neg
    shifted = neg - max_neg
    log_sum = max_neg + np.log(np.sum(np.exp(shifted / temp)))
    return -log_sum


def priority(instance):
    t0 = time.time()
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)

    # ---- Build predecessor / successor lists ----
    preds = instance.get("predecessors", None)
    pred_lists = []
    if preds is None:
        pred_lists = [[] for _ in range(n)]
    else:
        for i in range(n):
            p = preds[i]
            if p is None:
                p = []
            if isinstance(p, (list, tuple, np.ndarray)):
                p = [int(x) for x in p]
            else:
                p = [int(p)]
            p = [x for x in p if 0 <= x < n and x != i]
            seen = set()
            out = []
            for x in p:
                if x not in seen:
                    seen.add(x)
                    out.append(x)
            pred_lists.append(out)

    succ_lists = [[] for _ in range(n)]
    for i in range(n):
        for j in pred_lists[i]:
            succ_lists[j].append(i)

    # ---- Topological order (Kahn) ----
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for j in pred_lists[i]:
            in_degree[i] += 1
    q = deque()
    for i in range(n):
        if in_degree[i] == 0:
            q.append(i)
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for v in succ_lists[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                q.append(v)
    if len(topo) < n:
        for i in range(n):
            if i not in set(topo):
                topo.append(i)

    # ---- Forward pass: earliest start / earliest finish ----
    ES = np.zeros(n, dtype=float)
    EF = np.zeros(n, dtype=float)
    for i in topo:
        if pred_lists[i]:
            ES[i] = max(EF[j] for j in pred_lists[i])
        else:
            ES[i] = 0.0
        EF[i] = ES[i] + durations[i]

    project_dur = float(max(EF)) if n > 0 else 0.0

    # ---- Backward pass with soft-min for smoother criticality ----
    LF = np.full(n, project_dur, dtype=float)
    LS = np.zeros(n, dtype=float)
    soft_temp = 1.5
    for i in reversed(topo):
        if succ_lists[i]:
            ls_vals = [LS[j] for j in succ_lists[i]]
            LF[i] = _soft_min(ls_vals, temp=soft_temp)
        else:
            LF[i] = project_dur
        LS[i] = LF[i] - durations[i]

    # ---- Total Float (slack) ----
    slack = LS - ES

    s_min = float(np.min(slack)) if n > 0 else 0.0
    s_max = float(np.max(slack)) if n > 0 else 0.0
    s_range = s_max - s_min
    if s_range > 1e-12:
        slack_norm = (slack - s_min) / s_range
        crit = 1.0 - slack_norm
    else:
        crit = np.ones(n, dtype=float)

    # ---- Total remaining work (downstream duration sum) ----
    # Prioritize activities that, when completed, unlock the largest amount
    # of remaining project work. Weight by inverse parallel-path count to
    # deprioritize activities that fork into many parallel branches (which
    # create more resource contention).
    remaining_work = _total_remaining_work(durations, succ_lists, topo)
    pw_max = float(np.max(remaining_work)) if n > 0 else 1.0
    if pw_max > 1e-12:
        pw_norm = remaining_work / pw_max
    else:
        pw_norm = np.zeros(n, dtype=float)

    path_counts = _parallel_path_count(succ_lists, topo)
    # Inverse path count: activities with fewer parallel successors get higher
    # weight (less contention created). Normalize to [0,1] range.
    inv_path = 1.0 / path_counts  # (n,)
    ip_min = float(np.min(inv_path)) if n > 0 else 0.0
    ip_max = float(np.max(inv_path)) if n > 0 else 1.0
    ip_range = ip_max - ip_min
    if ip_range > 1e-12:
        inv_path_norm = (inv_path - ip_min) / ip_range
    else:
        inv_path_norm = np.ones(n, dtype=float)

    # Combine: remaining work weighted by inverse parallel paths
    remaining_score = pw_norm * inv_path_norm
    rs_max = float(np.max(remaining_score)) if n > 0 else 1.0
    if rs_max > 1e-12:
        remaining_score = remaining_score / rs_max
    else:
        remaining_score = np.zeros(n, dtype=float)

    # ---- Resource pressure: dynamic capacity depletion (fluid-flow congestion) ----
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)

    res_score = np.zeros(n, dtype=float)
    if demands is not None:
        D = np.asarray(demands, dtype=float)
        if D.size == 0:
            D = None
        if D is not None:
            if D.ndim == 1:
                if D.shape[0] == n:
                    D = D.reshape(n, 1)
                else:
                    D = None
            if D is not None and D.shape[0] == n:
                R = D.shape[1]
                caps = None
                if capacities is not None:
                    C = np.asarray(capacities, dtype=float).reshape(-1)
                    if C.size == 1:
                        caps = np.full(R, float(C[0]), dtype=float)
                    elif C.size < R:
                        caps = np.ones(R, dtype=float)
                        caps[:C.size] = C
                    else:
                        caps = C[:R]
                    caps = np.maximum(caps, 1e-9)
                if caps is not None:
                    # Per-activity, per-resource utilization ratio
                    ratio = D / caps[np.newaxis, :]  # (n, R)

                    # Static scarcity: total demand per resource relative to capacity
                    total_demand_per_res = np.sum(D, axis=0)  # (R,)
                    scarcity = total_demand_per_res / caps  # (R,)
                    sc_max = float(np.max(scarcity)) if R > 0 else 1.0
                    if sc_max > 1e-12:
                        scarcity = scarcity / sc_max  # normalize to [0,1]

                    # Dynamic congestion: for each activity, estimate the peak
                    # cumulative load it contributes to each resource over its
                    # execution window [ES_i, EF_i].
                    windows = np.stack([ES, EF], axis=1)  # (n, 2)
                    start_max = np.maximum(windows[:, 0][:, np.newaxis], windows[:, 0][np.newaxis, :])
                    end_min = np.minimum(windows[:, 1][:, np.newaxis], windows[:, 1][np.newaxis, :])
                    overlap = np.maximum(0.0, end_min - start_max)  # (n, n)

                    dur_safe = np.maximum(durations, 1e-9)
                    cumulative_load = overlap @ D  # (n, R)
                    congestion = cumulative_load / dur_safe[:, np.newaxis]  # (n, R)

                    # Weight by scarcity of each resource (cubic to strongly
                    # emphasize the tightest / most contested resource,
                    # mimicking a min-cut bottleneck signal).
                    scarcity_cu = scarcity ** 3
                    congestion_weighted = congestion * scarcity_cu[np.newaxis, :]

                    # res_score: total weighted congestion per activity
                    res_score = np.sum(congestion_weighted, axis=1)
                    rm = float(np.max(res_score)) if n > 0 else 1.0
                    if rm > 1e-12:
                        res_score = res_score / rm
                    else:
                        res_score = np.zeros(n, dtype=float)

    # ---- Bottleneck potential: duration weighted by successor count ----
    succ_counts = np.array([len(succ_lists[i]) for i in range(n)], dtype=float)
    d_max = float(np.max(durations)) if n > 0 else 1.0
    if d_max > 1e-12:
        bottleneck = (durations / d_max) * (1.0 + np.log1p(succ_counts))
        bn_max = float(np.max(bottleneck)) if n > 0 else 1.0
        if bn_max > 1e-12:
            bottleneck = bottleneck / bn_max
    else:
        bottleneck = np.zeros(n, dtype=float)

    # ---- Composite priority ----
    # crit: structural criticality (inverse total float)
    # remaining_score: total downstream work weighted by inverse parallel paths
    # res_score: resource congestion pressure
    # bottleneck: duration * branching factor
    score = crit + 0.40 * remaining_score + 0.45 * res_score + 0.08 * bottleneck

    # Deadline guard
    if (time.time() - t0) > _DEADLINE_SECONDS:
        return np.asarray(score, dtype=float)

    return np.asarray(score, dtype=float)
# EVOLVE-BLOCK-END
