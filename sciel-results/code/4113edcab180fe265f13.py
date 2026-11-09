"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 140.0  # Safety margin before 150s kill


def priority(instance):
    """
    Compute priority scores for each activity in a resource-constrained
    project scheduling problem.

    Strategy: primary driver is the critical path (longest path from each
    activity to any sink). This is multiplicatively boosted by resource
    contention (sum of demand/capacity ratios plus the bottleneck max ratio),
    so activities that are both on the critical path AND resource-heavy are
    strongly prioritized. Small additive tie-breakers (duration, number of
    successors) break ties among equal-critical-path activities.
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

    # Build successor list
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

    # Number of (direct) successors - tie-breaker for bottleneck potential
    succ_count = np.array([len(successors[i]) for i in range(n)], dtype=float)

    # Resource pressure: sum ratio and max (bottleneck) ratio
    n_resources = len(capacities) if capacities.ndim > 0 else 0
    sum_ratio = np.zeros(n, dtype=float)
    max_ratio = np.zeros(n, dtype=float)
    if n_resources > 0:
        for j in range(n):
            mx = 0.0
            s = 0.0
            for k in range(n_resources):
                if capacities[k] > 0:
                    r = demands[j][k] / capacities[k]
                    if r > mx:
                        mx = r
                    s += r
            sum_ratio[j] = s
            max_ratio[j] = mx

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

    # Multiplicative boost: critical path primary, resource contention secondary.
    # Bottleneck (max ratio) weighted slightly higher than sum ratio because a
    # single saturated resource is the true constraint in serial SGS.
    resource_term = 0.7 * norm_max + 0.5 * norm_sum
    score = norm_cp * (1.0 + resource_term)

    # Small additive tie-breakers to differentiate equal-critical-path activities
    score = score + 0.05 * norm_dur + 0.05 * norm_succ

    return score
# EVOLVE-BLOCK-END
