"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq

_START_TIME = time.time()
_DEADLINE = 140.0  # Safety margin before 150s kill


def _remaining_budget():
    return _DEADLINE - (time.time() - _START_TIME)


def priority(instance):
    """
    Compute priority scores for each activity in a resource-constrained
    project scheduling problem.

    Strategy: primary driver is the critical path (longest path from each
    activity to any sink). This is multiplicatively boosted by resource
    contention using a smooth soft-max surrogate, and augmented with a
    resource-weighted critical path tail that captures chains of
    resource-heavy activities. Tie-breakers use duration and successor count.
    """
    if time.time() - _START_TIME > _DEADLINE:
        return np.asarray(instance["durations"], dtype=float)

    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    predecessors = instance["predecessors"]

    n = len(durations)
    if n == 0:
        return np.array([])

    # Build successor list and in-degree
    successors = [[] for _ in range(n)]
    in_degree = np.zeros(n, dtype=int)
    for j in range(n):
        if predecessors[j]:
            for p in predecessors[j]:
                successors[p].append(j)
                in_degree[j] += 1

    # Topological sort using heapq for efficiency
    queue = [j for j in range(n) if in_degree[j] == 0]
    heapq.heapify(queue)
    top_order = []
    while queue:
        u = heapq.heappop(queue)
        top_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                heapq.heappush(queue, v)

    # Reverse topological order
    rev_top_order = list(reversed(top_order))

    # Longest path from each activity to any sink (critical path tail)
    longest_path_to_end = np.zeros(n, dtype=float)
    for i in rev_top_order:
        max_succ = 0.0
        for j in successors[i]:
            cand = durations[j] + longest_path_to_end[j]
            if cand > max_succ:
                max_succ = cand
        longest_path_to_end[i] = durations[i] + max_succ

    # Number of (direct) successors - tie-breaker for bottleneck potential
    succ_count = np.array([len(successors[i]) for i in range(n)], dtype=float)

    # Resource pressure using smooth soft-max (log-sum-exp) and sum ratio
    n_resources = len(capacities) if capacities.ndim > 0 else 0
    sum_ratio = np.zeros(n, dtype=float)
    soft_max_ratio = np.zeros(n, dtype=float)

    if n_resources > 0:
        temp = 0.15  # temperature for soft-max smoothness
        for j in range(n):
            s = 0.0
            ratios = []
            for k in range(n_resources):
                if capacities[k] > 0:
                    r = demands[j][k] / capacities[k]
                    s += r
                    if r > 0:
                        ratios.append(r)
            sum_ratio[j] = s
            if ratios:
                r_arr = np.array(ratios)
                m = float(np.max(r_arr))
                if m > 1e-10:
                    # log-sum-exp with temperature: m + temp * log(sum(exp((r-m)/temp)))
                    log_sum_exp = m + temp * np.log(np.sum(np.exp((r_arr - m) / temp)))
                    soft_max_ratio[j] = log_sum_exp
                else:
                    soft_max_ratio[j] = 0.0
            else:
                soft_max_ratio[j] = 0.0

    # Resource-weighted critical path tail:
    # For each activity, compute the longest path where edge weights are
    # boosted by resource contention of the successor. This captures the
    # idea that a chain of resource-heavy activities is more constraining
    # than a chain of light ones.
    resource_tail = np.zeros(n, dtype=float)
    if n_resources > 0:
        for i in rev_top_order:
            max_rt = 0.0
            for j in successors[i]:
                # Weight successor's path by its own resource pressure
                w = 1.0 + 0.25 * (sum_ratio[j] / max(1.0, n_resources))
                cand = durations[j] * w + resource_tail[j]
                if cand > max_rt:
                    max_rt = cand
            resource_tail[i] = durations[i] + max_rt

    def safe_norm(arr):
        m = float(np.max(arr))
        if m <= 1e-10:
            return np.zeros_like(arr, dtype=float)
        return arr / m

    norm_cp = safe_norm(longest_path_to_end)
    norm_sum = safe_norm(sum_ratio)
    norm_sm = safe_norm(soft_max_ratio)
    norm_dur = safe_norm(durations)
    norm_succ = safe_norm(succ_count)
    norm_rt = safe_norm(resource_tail)

    # Multiplicative boost: critical path primary, resource contention secondary.
    # Combine pure critical path with resource-weighted tail to capture
    # both structural and resource-driven scheduling pressure.
    cp_combined = 0.65 * norm_cp + 0.35 * norm_rt
    resource_term = 0.55 * norm_sm + 0.45 * norm_sum
    score = cp_combined * (1.0 + resource_term)

    # Small additive tie-breakers to differentiate equal-critical-path activities
    score = score + 0.04 * norm_dur + 0.02 * norm_succ

    return score
# EVOLVE-BLOCK-END
