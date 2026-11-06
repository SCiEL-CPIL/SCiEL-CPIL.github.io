"""RCPSP priority scoring using MCTS-inspired lookahead over eligible sets."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque, defaultdict
import heapq

def _topo_sort(n, predecessors, successors):
    in_degree = [0] * n
    for i in range(n):
        for j in predecessors[i]:
            in_degree[i] += 1
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo = []
    temp = in_degree[:]
    while queue:
        node = queue.popleft()
        topo.append(node)
        for succ in successors[node]:
            temp[succ] -= 1
            if temp[succ] == 0:
                queue.append(succ)
    return topo

def _simulate(instance, priority_fn, max_steps=10000):
    """
    Serial schedule generation with resource checking.
    Returns makespan.
    """
    n = len(instance["durations"])
    if n == 0:
        return 0.0

    durations = np.asarray(instance["durations"], dtype=float)
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    predecessors = instance.get("predecessors", None)

    if predecessors is None:
        predecessors = [set() for _ in range(n)]
    else:
        predecessors = [set(p) if p is not None else set() for p in predecessors]

    n_resources = 0
    if demands is not None and capacities is not None:
        n_resources = len(capacities)
        cap_vector = np.array(capacities, dtype=float)
        demand_matrix = np.zeros((n, n_resources), dtype=float)
        for i in range(n):
            if isinstance(demands[i], (list, tuple, np.ndarray)):
                for r in range(n_resources):
                    demand_matrix[i, r] = float(demands[i][r])
            else:
                demand_matrix[i, 0] = float(demands[i])
    else:
        cap_vector = None
        demand_matrix = None

    scheduled = [False] * n
    finish_times = [0.0] * n
    start_times = [0.0] * n
    remaining = set(range(n))

    current_time = 0.0
    active_jobs = []  # (finish_time, job_id)

    def _resource_usage_at(time):
        if n_resources == 0:
            return np.zeros(0)
        usage = np.zeros(n_resources)
        for ft, jid in active_jobs:
            if start_times[jid] <= time < ft:
                usage += demand_matrix[jid]
        return usage

    steps = 0
    while remaining and steps < max_steps:
        steps += 1
        # Find eligible jobs
        eligible = []
        for j in remaining:
            all_done = True
            for k in predecessors[j]:
                if not scheduled[k]:
                    all_done = False
                    break
            if all_done:
                eligible.append(j)

        if not eligible:
            if not active_jobs:
                break
            min_finish = min(ft for ft, _ in active_jobs)
            current_time = min_finish
            still_active = []
            for ft, jid in active_jobs:
                if ft == current_time:
                    scheduled[jid] = True
                    finish_times[jid] = current_time
                    remaining.discard(jid)
                else:
                    still_active.append((ft, jid))
            active_jobs = still_active
            continue

        # Try to schedule highest priority eligible job that fits
        eligible_sorted = sorted(eligible, key=lambda j: priority_fn[j], reverse=True)

        placed = False
        for j in eligible_sorted:
            if n_resources == 0:
                placed = True
                break
            current_usage = _resource_usage_at(current_time)
            new_usage = current_usage + demand_matrix[j]
            if np.all(new_usage <= cap_vector + 1e-9):
                placed = True
                break

        if not placed:
            if not active_jobs:
                j = eligible_sorted[0]
                placed = True
            else:
                min_finish = min(ft for ft, _ in active_jobs)
                current_time = min_finish
                still_active = []
                for ft, jid in active_jobs:
                    if ft == current_time:
                        scheduled[jid] = True
                        finish_times[jid] = current_time
                        remaining.discard(jid)
                    else:
                        still_active.append((ft, jid))
                active_jobs = still_active
                continue

        start_times[j] = current_time
        finish_times[j] = current_time + durations[j]
        scheduled[j] = True
        remaining.discard(j)
        active_jobs.append((finish_times[j], j))

    while active_jobs:
        min_finish = min(ft for ft, _ in active_jobs)
        current_time = min_finish
        still_active = []
        for ft, jid in active_jobs:
            if ft == current_time:
                scheduled[jid] = True
                finish_times[jid] = current_time
            else:
                still_active.append((ft, jid))
        active_jobs = still_active

    makespan = max(finish_times) if n > 0 else 0.0
    return makespan

def priority(instance):
    n = len(instance["durations"])
    if n == 0:
        return np.array([], dtype=float)

    durations = np.asarray(instance["durations"], dtype=float)
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    predecessors = instance.get("predecessors", None)

    if predecessors is None:
        predecessors = [set() for _ in range(n)]
    else:
        predecessors = [set(p) if p is not None else set() for p in predecessors]

    successors = [set() for _ in range(n)]
    for i in range(n):
        for j in predecessors[i]:
            successors[j].add(i)

    topo = _topo_sort(n, predecessors, successors)
    if len(topo) < n:
        return durations.copy()

    # Longest path from node to end
    longest_to_end = [0.0] * n
    for node in reversed(topo):
        max_succ = 0.0
        for succ in successors[node]:
            max_succ = max(max_succ, longest_to_end[succ])
        longest_to_end[node] = durations[node] + max_succ

    # Longest path from start to node
    longest_from_start = [0.0] * n
    for node in topo:
        max_pred = 0.0
        for pred in predecessors[node]:
            max_pred = max(max_pred, longest_from_start[pred])
        longest_from_start[node] = max_pred + durations[node]

    # MCTS-inspired: for each activity, simulate a schedule where that activity
    # is strongly boosted in priority, and record the makespan.
    # Score[j] = -makespan_when_boosting_j (higher = better)
    # To keep it fast, we do this for all n activities (n=30, each sim ~O(n^2))

    base_priority = np.array(longest_to_end, dtype=float)
    makespans = np.zeros(n)

    for j in range(n):
        # Create a priority vector that boosts activity j
        boost = 1000.0
        p = base_priority.copy()
        p[j] += boost
        ms = _simulate(instance, p)
        makespans[j] = ms

    # Score: higher is better priority. Lower makespan when boosting j means j is more critical.
    # So score[j] = -makespans[j]
    scores = -makespans

    # Normalize
    if scores.max() > scores.min():
        scores = (scores - scores.min()) / (scores.max() - scores.min())

    return scores
# EVOLVE-BLOCK-END
