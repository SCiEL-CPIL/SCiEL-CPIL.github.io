import numpy as np
import time

_START_TIME = None
_DEADLINE = 115.0  # safety margin before 120s kill


def priority(instance):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

    # Deadline guard
    if time.time() - _START_TIME > _DEADLINE - 5.0:
        durations = np.asarray(instance["durations"], dtype=float)
        return durations.copy()

    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    predecessors = instance["predecessors"]

    n = len(durations)
    n_res = len(capacities)

    # Build successor lists
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)

    # Compute CPM Rank (successor depth - longest path from i to end)
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for p in predecessors[i]:
            in_degree[i] += 1

    queue = [i for i in range(n) if in_degree[i] == 0]
    topo_order = []
    temp_in_degree = in_degree.copy()
    while queue:
        u = queue.pop(0)
        topo_order.append(u)
        for v in successors[u]:
            temp_in_degree[v] -= 1
            if temp_in_degree[v] == 0:
                queue.append(v)

    cpm_rank = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        if not successors[u]:
            cpm_rank[u] = durations[u]
        else:
            max_succ = max(cpm_rank[v] for v in successors[u])
            cpm_rank[u] = durations[u] + max_succ

    # --- Dynamic Look-Ahead Simulation ---
    sim_finish = [0.0] * n
    sim_start = [0.0] * n
    scheduled = [False] * n

    for _ in range(n):
        # Find eligible activities (all predecessors scheduled)
        eligible = [i for i in range(n) if not scheduled[i] and all(scheduled[p] for p in predecessors[i])]
        if not eligible:
            break

        # Sort eligible by CPM rank (descending) to pick the most critical one first
        eligible.sort(key=lambda x: cpm_rank[x], reverse=True)

        # Try to schedule the first eligible activity that fits
        # In a strict serial SGS, we pick the highest priority eligible and schedule it.
        # We assume the highest priority one can be scheduled (possibly delayed).
        i = eligible[0]
        
        # Find earliest start time
        earliest_start = 0.0
        for p in predecessors[i]:
            earliest_start = max(earliest_start, sim_finish[p])
            
        start = earliest_start
        
        # Check for resource conflicts and delay if necessary
        # We iterate until we find a slot where resources are sufficient
        # Since N is small, this is fast
        while True:
            conflict = False
            for j in range(n):
                if not scheduled[j]:
                    continue
                # Check if [start, start + durations[i]] overlaps with [sim_start[j], sim_finish[j]]
                if sim_start[j] < start + durations[i] and sim_finish[j] > start:
                    # Overlap exists
                    # Check if resource demand exceeds capacity
                    if np.any(demands[i] + demands[j] > capacities):
                        conflict = True
                        # Delay start to the finish of the conflicting activity
                        if sim_finish[j] > start:
                            start = sim_finish[j]
                        break
            if not conflict:
                break
            # Safety break to avoid infinite loops
            if start > 10000:
                break

        sim_start[i] = start
        sim_finish[i] = start + durations[i]
        scheduled[i] = True

    makespan = max(sim_finish) if sim_finish else 0.0

    # Calculate Slack
    # Slack_i = Makespan - (sim_finish[i] + cpm_rank[i])
    # cpm_rank[i] is the longest path from i to end in the graph
    slack = makespan - (np.array(sim_finish) + cpm_rank)
    slack = np.maximum(slack, 0.0)

    # Priority: Inverse of Slack (higher priority for lower slack)
    inv_slack = 1.0 / (slack + 0.1)

    # Normalize inv_slack
    inv_slack_min, inv_slack_max = np.min(inv_slack), np.max(inv_slack)
    inv_slack_range = inv_slack_max - inv_slack_min if inv_slack_max > inv_slack_min else 1.0
    inv_slack_norm = (inv_slack - inv_slack_min) / inv_slack_range

    # Resource Pressure
    safe_caps = np.where(capacities == 0, 1, capacities)
    ratios = demands / safe_caps
    resource_pressure = np.max(ratios, axis=1)

    # Combine
    # High priority for low slack (critical) and high resource pressure.
    scores = 2.0 * inv_slack_norm * inv_slack_range + 1.0 * resource_pressure * inv_slack_range

    return scores
