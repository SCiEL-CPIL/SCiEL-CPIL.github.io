import numpy as np


def priority(instance):
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)  # shape (n_activities, n_resources)
    capacities = np.asarray(instance["capacities"], dtype=float)  # shape (n_resources,)
    predecessors = instance["predecessors"]  # list of lists, predecessors[i] = list of pred indices

    n = len(durations)
    n_res = len(capacities)

    # Build successor lists
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)

    # Forward pass: earliest start/finish
    es = np.zeros(n)
    ef = np.zeros(n)
    for i in range(n):
        if predecessors[i]:
            es[i] = max(ef[p] for p in predecessors[i])
        ef[i] = es[i] + durations[i]

    # Project makespan (no resources)
    makespan = max(ef) if n > 0 else 0.0

    # Backward pass: latest start/finish
    lf = np.full(n, makespan)
    ls = np.full(n, makespan)
    # Process in reverse topological order
    # Use a simple approach: iterate until convergence or use topological sort
    # For simplicity, do iterative backward pass
    for _ in range(n + 1):
        for i in range(n - 1, -1, -1):
            if successors[i]:
                lf[i] = min(ls[s] for s in successors[i])
            else:
                lf[i] = makespan
            ls[i] = lf[i] - durations[i]

    # Total float
    total_float = lf - ef

    # Resource pressure: sum of demand/capacity for each activity
    # Avoid division by zero
    safe_cap = np.where(capacities > 0, capacities, 1.0)
    resource_pressure = np.sum(demands / safe_cap[np.newaxis, :], axis=1)

    # Successor count
    succ_count = np.array([len(successors[i]) for i in range(n)], dtype=float)

    # Normalize each component to [0, 1]
    def normalize(arr):
        arr = np.asarray(arr, dtype=float)
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.ones_like(arr) * 0.5
        return (arr - mn) / (mx - mn)

    # Invert total_float: low float = high criticality = high priority
    crit_score = 1.0 - normalize(total_float)
    res_score = normalize(resource_pressure)
    succ_score = normalize(succ_count)

    # Composite priority: weighted sum
    # Emphasize criticality most, then resource pressure, then successor count
    priority_score = 0.5 * crit_score + 0.3 * res_score + 0.2 * succ_score

    return priority_score
