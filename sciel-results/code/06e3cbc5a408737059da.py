"""Priority function for resource-constrained project scheduling (serial SGS)."""
import numpy as np
import time

# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque


def priority(instance):
    """Return a priority score per activity (higher = schedule earlier).

    Combines:
      - Critical path length (longest path from activity to project end)
      - Dynamic resource contention: for each activity, the max over resources
        of (its demand / capacity) weighted by how many other activities also
        demand that resource (contention pressure)
      - Successor count (activities with many successors are bottlenecks)
    """
    n = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    predecessors = instance["predecessors"]

    # Build successor lists
    successors = [[] for _ in range(n)]
    for j in range(n):
        for i in predecessors[j]:
            successors[i].append(j)

    # Compute critical path length (longest path to end) via reverse topological order
    in_deg = np.zeros(n, dtype=int)
    for j in range(n):
        for i in predecessors[j]:
            in_deg[j] += 1

    # Topological order (Kahn's algorithm)
    queue = deque([i for i in range(n) if in_deg[i] == 0])
    topo = []
    while queue:
        u = queue.popleft()
        topo.append(u)
        for v in successors[u]:
            in_deg[v] -= 1
            if in_deg[v] == 0:
                queue.append(v)

    if len(topo) < n:
        topo = list(range(n))

    rtopo = list(reversed(topo))

    # Longest path from each node to end (including its own duration)
    cp_len = np.zeros(n, dtype=float)
    for u in rtopo:
        cp_len[u] = durations[u]
        for v in successors[u]:
            if cp_len[v] + durations[u] > cp_len[u]:
                cp_len[u] = durations[u] + cp_len[v]

    # Dynamic resource contention score:
    # For each resource r, compute how many activities demand it (non-zero demand).
    # An activity's contention on resource r is proportional to its demand_r / cap_r
    # times the number of other activities also using resource r.
    # This captures: high-demand activities on scarce, heavily-contended resources
    # should be scheduled early to prevent later bottlenecks.

    if demands.ndim == 1:
        # Single resource: reshape to (n, 1)
        demands_2d = demands.reshape(-1, 1)
        capacities_1d = np.asarray([capacities])
    else:
        demands_2d = demands
        capacities_1d = capacities

    R = demands_2d.shape[1]
    cap_safe = np.where(capacities_1d > 0, capacities_1d, 1e-9)

    # For each resource, count how many activities have non-zero demand
    resource_user_count = np.zeros(R, dtype=float)
    for r in range(R):
        resource_user_count[r] = np.sum(demands_2d[:, r] > 0)

    # Contention score per activity:
    # For activity i, contention_i = sum over resources r of:
    #   (demand[i,r] / cap[r]) * (resource_user_count[r] - 1)
    # This is 0 if an activity is the only user of all its resources.
    # It grows with both the relative demand and the number of competitors.
    contention = np.zeros(n, dtype=float)
    for r in range(R):
        rel_demand = demands_2d[:, r] / cap_safe[r]
        # Only count contention from other activities (subtract 1 for self)
        contention += rel_demand * (resource_user_count[r] - 1)

    # Successor count (bottleneck indicator)
    succ_count = np.array([len(successors[i]) for i in range(n)], dtype=float)

    # Normalize each component to [0, 1]
    def norm(x):
        mn, mx = x.min(), x.max()
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)

    cp_norm = norm(cp_len)
    cont_norm = norm(contention)
    succ_norm = norm(succ_count)

    # Weighted combination: critical path most important, then contention, then successors
    score = 0.50 * cp_norm + 0.35 * cont_norm + 0.15 * succ_norm

    return score
# EVOLVE-BLOCK-END
