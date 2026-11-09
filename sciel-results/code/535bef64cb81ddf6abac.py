# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE_SECONDS = 140.0


def _remaining_budget():
    return _DEADLINE_SECONDS - (time.time() - _START_TIME)


def priority(instance):
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    predecessors = instance["predecessors"]

    n = len(durations)
    if n == 0:
        return np.array([])

    # Build successors and in-degree
    successors = [[] for _ in range(n)]
    in_degree = np.zeros(n, dtype=int)
    for j in range(n):
        if predecessors[j]:
            for p in predecessors[j]:
                successors[p].append(j)
                in_degree[j] += 1

    # Topological sort (Kahn's algorithm)
    queue = [j for j in range(n) if in_degree[j] == 0]
    top_order = []
    while queue:
        u = queue.pop(0)
        top_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

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

    # Number of direct successors
    succ_count = np.array([len(successors[i]) for i in range(n)], dtype=float)

    # Resource pressure: sum ratio and soft-max (log-sum-exp) ratio
    n_resources = len(capacities) if capacities.ndim > 0 else 0
    sum_ratio = np.zeros(n, dtype=float)
    max_ratio = np.zeros(n, dtype=float)
    if n_resources > 0:
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
                # Soft-max via log-sum-exp with temperature for smoothness
                temp = 0.1
                r_arr = np.array(ratios)
                m = float(np.max(r_arr))
                if m > 1e-10:
                    # log-sum-exp: m + temp * log(sum(exp((r - m) / temp)))
                    log_sum_exp = m + temp * np.log(np.sum(np.exp((r_arr - m) / temp)))
                    max_ratio[j] = log_sum_exp
                else:
                    max_ratio[j] = 0.0
            else:
                max_ratio[j] = 0.0

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
                w = 1.0 + 0.3 * sum_ratio[j]
                cand = durations[j] * w + resource_tail[j]
                if cand > max_rt:
                    max_rt = cand
            resource_tail[i] = durations[i] + max_rt

    # Compute "required capacity" proxy: for each activity, the minimum
    # capacity that would be needed so that its resource demand is not
    # a bottleneck relative to the critical path. This is modeled as
    # the ratio of duration to critical path length, scaled by demand.
    # Activities with high demand relative to their critical path share
    # are more constraining.
    max_cp = np.max(longest_path_to_end) if n > 0 else 1.0
    if max_cp <= 1e-10:
        max_cp = 1.0

    # Required capacity proxy: duration / (critical path length) * sum of demands
    # This captures how much resource "pressure" each activity exerts
    # relative to its position in the critical path.
    required_capacity = np.zeros(n, dtype=float)
    if n_resources > 0:
        for j in range(n):
            total_demand = 0.0
            for k in range(n_resources):
                total_demand += demands[j][k]
            # Scale by inverse of critical path share
            cp_share = durations[j] / max_cp
            if cp_share > 1e-10:
                required_capacity[j] = total_demand / cp_share
            else:
                required_capacity[j] = total_demand

    # Normalize components
    def safe_norm(arr):
        m = float(np.max(arr))
        if m <= 1e-10:
            return np.zeros_like(arr, dtype=float)
        return arr / m

    norm_cp = safe_norm(longest_path_to_end)
    norm_sum = safe_norm(sum_ratio)
    norm_max = safe_norm(max_ratio)
    norm_dur = safe_norm(durations)
    norm_succ = safe_norm(succ_count)
    norm_rt = safe_norm(resource_tail)
    norm_req_cap = safe_norm(required_capacity)

    # Combine: primary driver is critical path (tail), boosted by resource contention.
    # Use resource-weighted tail and required capacity as additional signals.
    cp_combined = 0.6 * norm_cp + 0.4 * norm_rt
    resource_term = 0.5 * norm_max + 0.3 * norm_sum + 0.2 * norm_req_cap
    score = cp_combined * (1.0 + resource_term)

    # Small additive tie-breakers to differentiate equal-critical-path activities
    score = score + 0.05 * norm_dur + 0.03 * norm_succ

    return score
# EVOLVE-BLOCK-END
