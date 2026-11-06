"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 135.0  # wall-clock safety margin under the 150s evaluator limit


def priority(instance):
    t0 = time.time()

    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    if n == 0:
        return np.array([])
    cycle_time = float(instance.get("cycle_time", 40) or 40)
    if cycle_time <= 0:
        cycle_time = 1.0

    # Normalise predecessor sets (accept lists or sets, ints or np ints).
    preds = [set(int(p) for p in (instance.get("predecessors", [set() for _ in range(n)])[i] or ()))
             for i in range(n)]

    # Build successor lists.
    succs = [set() for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            if 0 <= p < n:
                succs[p].add(i)

    # Topological order (Kahn) for iterative path computation.
    indeg = np.zeros(n, dtype=int)
    for i in range(n):
        for p in preds[i]:
            if 0 <= p < n:
                indeg[i] += 1
    order = []
    frontier = list(range(n))
    for i in range(n):
        if indeg[i] > 0:
            frontier[i] = -1
    frontier = [i for i in range(n) if indeg[i] == 0]
    # Simple Kahn without a heap.
    queue = list(frontier)
    indeg = indeg.copy()
    while queue:
        cur = queue.pop(0)
        order.append(cur)
        for j in succs[cur]:
            indeg[j] -= 1
            if indeg[j] == 0:
                queue.append(j)
    # Guard against cycles: append any leftovers.
    if len(order) < n:
        for i in range(n):
            if i not in set(order):
                order.append(i)

    # Longest path from sources up to and including each task (predecessor path + own time).
    path_in = np.zeros(n, dtype=float)
    for i in order:
        best = times[i]
        for p in preds[i]:
            if path_in[p] + times[i] > best:
                best = path_in[p] + times[i]
        path_in[i] = best

    # Longest path from each task to sinks (own time + successor path).
    path_out = np.zeros(n, dtype=float)
    for i in reversed(order):
        best = times[i]
        for j in succs[i]:
            if times[i] + path_out[j] > best:
                best = times[i] + path_out[j]
        path_out[i] = best

    # Throughput criticality: the span a task occupies in the critical chain.
    through = path_in + path_out - times  # predecessor path + successor path (excludes own time double count)

    # Resource contention: sum of durations of all transitive predecessors and successors.
    # Tasks in time-dense interconnected subgraphs get higher priority.
    trans_pred_time = np.zeros(n, dtype=float)
    for i in order:
        best = times[i]
        for p in preds[i]:
            if trans_pred_time[p] + times[i] > best:
                best = trans_pred_time[p] + times[i]
        trans_pred_time[i] = best

    trans_succ_time = np.zeros(n, dtype=float)
    for i in reversed(order):
        best = times[i]
        for j in succs[i]:
            if times[i] + trans_succ_time[j] > best:
                best = times[i] + trans_succ_time[j]
        trans_succ_time[i] = best

    # Contention metric: total time in the dependency neighborhood
    contention = trans_pred_time + trans_succ_time - times

    # Flexibility: slack before successors (cycle - path_in) and after (cycle - path_out).
    slack_in = np.clip(cycle_time - path_in, 0.0, cycle_time)
    slack_out = np.clip(cycle_time - path_out, 0.0, cycle_time)
    min_slack = np.minimum(slack_in, slack_out)

    # Normalisation helpers.
    def norm(x):
        rng = x.max() - x.min()
        if rng < 1e-12:
            return np.zeros_like(x)
        return (x - x.min()) / rng

    if time.time() - t0 > _DEADLINE - 5:
        # Return a sensible early fallback: rank by processing time.
        return norm(times)

    # Combine signals:
    #  - Processing time (long tasks are hard to pack) -> strong weight.
    #  - Resource contention (time-dense interconnected subgraphs) -> maximize parallelism potential.
    #  - Throughput criticality (spans the makespan) -> identifies true bottlenecks.
    #  - Inverse flexibility (little slack) -> must be placed early to avoid blocking.
    w_time = 0.30
    w_contention = 0.30
    w_through = 0.25
    w_flex = 0.15

    priority_scores = (
        w_time * norm(times)
        + w_contention * norm(contention)
        + w_through * norm(through)
        + w_flex * norm(-min_slack)
    )

    # Ensure strict total ordering by adding a tiny deterministic tie-break
    # so the greedy "highest priority first" rule is stable and unique.
    priority_scores = priority_scores - np.arange(n) * 1e-9

    return priority_scores
# EVOLVE-BLOCK-END
