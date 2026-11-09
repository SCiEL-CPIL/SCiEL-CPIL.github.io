# EVOLVE-BLOCK-START
import numpy as np
import time

def _simulate_station_count(times, predecessors, cycle_time, priorities):
    """Simulate greedy assignment and return number of stations.
    Uses a sorted active list for efficiency: at each step, scan the
    priority-sorted list from the top and pick the first task that fits.
    """
    n = len(times)
    # Build successor list
    successors = [[] for _ in range(n)]
    pred_count = [0] * n
    for i in range(n):
        preds = predecessors[i]
        pred_count[i] = len(preds)
        for p in preds:
            successors[p].append(i)

    # Track remaining predecessors for each task
    remaining_preds = pred_count[:]
    assigned = [False] * n
    active = [i for i in range(n) if remaining_preds[i] == 0]
    # Sort active by priority descending
    active.sort(key=lambda t: -priorities[t])
    active_set = set(active)

    station_count = 0
    remaining_time = 0
    assigned_count = 0

    while assigned_count < n:
        if remaining_time == 0:
            station_count += 1
            remaining_time = cycle_time

        # Find best available task that fits (scan sorted active list)
        best_task = -1
        for t in active:
            if not assigned[t] and times[t] <= remaining_time:
                best_task = t
                break

        if best_task == -1:
            # No task fits in current station, open new one
            station_count += 1
            remaining_time = cycle_time
            for t in active:
                if not assigned[t] and times[t] <= remaining_time:
                    best_task = t
                    break
            if best_task == -1:
                # Shouldn't happen if all tasks fit in cycle_time
                # Force assign the smallest unassigned task
                best_task = min((t for t in range(n) if not assigned[t]), key=lambda x: times[x])

        # Assign best_task
        assigned[best_task] = True
        assigned_count += 1
        remaining_time -= times[best_task]
        active_set.discard(best_task)

        # Update successors
        for s in successors[best_task]:
            remaining_preds[s] -= 1
            if remaining_preds[s] == 0:
                active.append(s)
                active_set.add(s)
                # Re-sort active list (n is small, this is fine)
                active.sort(key=lambda t: -priorities[t])

    return station_count


def _compute_ranks(times, predecessors):
    """Compute rank for each task: longest path from task to end."""
    n = len(times)
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
    
    # Topological order (reverse: from end to start)
    # Compute out-degree
    out_degree = [len(successors[i]) for i in range(n)]
    # Use Kahn's algorithm on reversed graph
    in_degree = [0] * n
    for i in range(n):
        for s in successors[i]:
            in_degree[s] += 1  # This is wrong for reversed
    
    # Actually, let me compute longest path from each node to any sink
    # Using topological sort
    in_deg = [len(predecessors[i]) for i in range(n)]
    from collections import deque
    queue = deque([i for i in range(n) if in_deg[i] == 0])
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for s in successors[u]:
            in_deg[s] -= 1
            if in_deg[s] == 0:
                queue.append(s)
    
    # Longest path from node to end: process in reverse topo order
    longest = [0.0] * n
    for u in reversed(topo_order):
        if successors[u]:
            longest[u] = times[u] + max(longest[s] for s in successors[u])
        else:
            longest[u] = times[u]
    
    return longest


def priority(instance):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)

    # Compute ranks
    ranks = _compute_ranks(times, predecessors)

    # Build successors for auxiliary features
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
    out_degree = np.array([len(successors[i]) for i in range(n)], dtype=float)
    in_degree = np.array([len(predecessors[i]) for i in range(n)], dtype=float)

    # Number of successors (recursive depth from node to sinks)
    # Compute "depth" = longest chain of successors
    depth = np.zeros(n)
    # Use memoized DFS via topological order
    in_deg = [len(predecessors[i]) for i in range(n)]
    from collections import deque
    queue = deque([i for i in range(n) if in_deg[i] == 0])
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for s in successors[u]:
            in_deg[s] -= 1
            if in_deg[s] == 0:
                queue.append(s)
    # depth[u] = 1 + max(depth[s]) for s in successors[u], 0 if no successors
    for u in reversed(topo_order):
        if successors[u]:
            depth[u] = 1.0 + max(depth[s] for s in successors[u])
        else:
            depth[u] = 0.0

    # Successor count (total descendants, not just immediate)
    desc_count = np.zeros(n)
    for u in reversed(topo_order):
        for s in successors[u]:
            desc_count[u] += 1 + desc_count[s]

    # Build candidate priority vectors
    candidates = []

    # Scheme 1: rank only
    candidates.append(ranks.copy())

    # Scheme 2: time only (LPT)
    candidates.append(times.copy())

    # Scheme 3: rank + alpha * time for various alpha
    for alpha in [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
        candidates.append(ranks + alpha * times)

    # Scheme 4: rank + beta * out_degree
    for beta in [0.5, 1.0, 2.0]:
        candidates.append(ranks + beta * out_degree)

    # Scheme 5: rank * (1 + out_degree)
    candidates.append(ranks * (1.0 + out_degree))

    # Scheme 6: rank + depth
    candidates.append(ranks + depth)

    # Scheme 7: rank + desc_count
    candidates.append(ranks + desc_count)

    # Scheme 8: (rank + time) * (1 + out_degree * 0.5)
    candidates.append((ranks + times) * (1.0 + 0.5 * out_degree))

    # Scheme 9: rank * (1 + depth * 0.1) + time
    candidates.append(ranks * (1.0 + 0.1 * depth) + times)

    # Scheme 10: rank + 0.5*time + 0.3*desc_count
    candidates.append(ranks + 0.5 * times + 0.3 * desc_count)

    # Evaluate all candidates
    best_score = float('inf')
    best_priorities = None
    for cand in candidates:
        score = _simulate_station_count(times, predecessors, cycle_time, cand)
        if score < best_score:
            best_score = score
            best_priorities = cand.copy()

    # Local search with multi-start and structured perturbations
    rng = np.random.default_rng(42)
    max_perturb_scale = max(1.0, np.std(ranks) * 0.3)

    improved = True
    stall_count = 0
    while improved and (time.time() < deadline - 5):
        improved = False
        stall_count += 1
        # Increase perturbation scale as we stall
        scale = max_perturb_scale * (1.0 + 0.2 * stall_count)

        for _ in range(80):
            if time.time() > deadline - 5:
                break

            # Random perturbation: pick 1-3 tasks and adjust
            num_perturb = int(rng.integers(1, 4))
            idxs = rng.choice(n, size=num_perturb, replace=False)
            new_priorities = best_priorities.copy()
            for i in idxs:
                new_priorities[i] += rng.normal(0, scale)

            score = _simulate_station_count(times, predecessors, cycle_time, new_priorities)
            if score < best_score:
                best_score = score
                best_priorities = new_priorities
                improved = True
                stall_count = 0

        # Try pair swaps
        if time.time() < deadline - 5:
            for _ in range(30):
                if time.time() > deadline - 5:
                    break
                i, j = rng.integers(0, n, size=2)
                if i == j:
                    continue
                new_priorities = best_priorities.copy()
                new_priorities[i], new_priorities[j] = new_priorities[j], new_priorities[i]
                score = _simulate_station_count(times, predecessors, cycle_time, new_priorities)
                if score < best_score:
                    best_score = score
                    best_priorities = new_priorities
                    improved = True
                    stall_count = 0

        # Multi-start: if stalled, try a fresh random restart from a random candidate
        if stall_count >= 3 and time.time() < deadline - 10:
            # Pick a random candidate and perturb it significantly
            base = candidates[int(rng.integers(0, len(candidates)))]
            new_priorities = base + rng.normal(0, max_perturb_scale, size=n)
            score = _simulate_station_count(times, predecessors, cycle_time, new_priorities)
            if score < best_score:
                best_score = score
                best_priorities = new_priorities
                improved = True
                stall_count = 0

    return best_priorities
# EVOLVE-BLOCK-END
