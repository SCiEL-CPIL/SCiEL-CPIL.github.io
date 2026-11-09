# EVOLVE-BLOCK-START
import time
import heapq
import numpy as np

_DEADLINE_SECONDS = 140.0
_START_TIME = time.time()


def _remaining_budget():
    return _DEADLINE_SECONDS - (time.time() - _START_TIME)


def _serial_sgs(durations, demands, capacities, predecessors, priority_vec):
    """
    Serial schedule-generation scheme with a given priority vector.
    Returns the makespan (float).
    """
    n = len(durations)
    if n == 0:
        return 0.0

    remaining_preds = np.zeros(n, dtype=int)
    successors = [[] for _ in range(n)]
    for j in range(n):
        pred_list = predecessors[j]
        if pred_list:
            remaining_preds[j] = len(pred_list)
            for p in pred_list:
                successors[p].append(j)

    # Max-heap via negation; tie-break by index for determinism.
    heap = []
    for j in range(n):
        if remaining_preds[j] == 0:
            heapq.heappush(heap, (-float(priority_vec[j]), j))

    start = np.zeros(n)
    finish = np.zeros(n)
    # Earliest time each resource is free (serial: one activity at a time).
    res_free = np.zeros(len(capacities))

    scheduled = 0
    while heap:
        _, j = heapq.heappop(heap)
        # Precedence constraint.
        ps = 0.0
        for p in predecessors[j]:
            if finish[p] > ps:
                ps = finish[p]
        # Resource constraint: earliest time all needed resources are free.
        rs = 0.0
        for k in range(len(capacities)):
            if demands[j][k] > 0 and res_free[k] > rs:
                rs = res_free[k]
        st = ps if ps > rs else rs
        start[j] = st
        finish[j] = st + durations[j]
        # Update resource free times.
        for k in range(len(capacities)):
            if demands[j][k] > 0 and finish[j] > res_free[k]:
                res_free[k] = finish[j]
        scheduled += 1
        # Release newly eligible successors.
        for v in successors[j]:
            remaining_preds[v] -= 1
            if remaining_preds[v] == 0:
                heapq.heappush(heap, (-float(priority_vec[v]), v))

    if scheduled < n:
        # Cycle / infeasible; return a large value so it is penalized.
        return 1e9
    return float(np.max(finish))


def priority(instance):
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    predecessors = instance["predecessors"]

    n = len(durations)
    if n == 0:
        return np.array([])

    # Build successors.
    successors = [[] for _ in range(n)]
    for j in range(n):
        if predecessors[j]:
            for p in predecessors[j]:
                successors[p].append(j)

    # In-degree for topological sort.
    in_degree = np.zeros(n, dtype=int)
    for j in range(n):
        if predecessors[j]:
            in_degree[j] = len(predecessors[j])

    queue = []
    for j in range(n):
        if in_degree[j] == 0:
            queue.append(j)

    top_order = []
    while queue:
        u = queue.pop(0)
        top_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    # Fallback if topo incomplete (shouldn't happen): append the rest.
    if len(top_order) < n:
        present = set(top_order)
        for j in range(n):
            if j not in present:
                top_order.append(j)

    rev_top_order = list(reversed(top_order))

    # Longest path from each node to the end (critical path tail).
    longest_path_to_end = np.zeros(n, dtype=float)
    for i in rev_top_order:
        max_succ = 0.0
        for j in successors[i]:
            cand = durations[j] + longest_path_to_end[j]
            if cand > max_succ:
                max_succ = cand
        longest_path_to_end[i] = durations[i] + max_succ

    # Resource pressure: max ratio (bottleneck) and sum ratio.
    max_ratio = np.zeros(n, dtype=float)
    sum_ratio = np.zeros(n, dtype=float)
    for j in range(n):
        mx = 0.0
        s = 0.0
        for k in range(len(capacities)):
            if capacities[k] > 0:
                r = demands[j][k] / capacities[k]
                if r > mx:
                    mx = r
                s += r
        max_ratio[j] = mx
        sum_ratio[j] = s

    # Successor count (fan-out): scheduling high-fan-out early can unblock many.
    succ_count = np.zeros(n, dtype=float)
    for j in range(n):
        succ_count[j] = len(successors[j])

    # Normalize components.
    def _norm(x):
        m = np.max(x)
        if m > 0:
            return x / m
        return np.zeros_like(x)

    norm_cp = _norm(longest_path_to_end)
    norm_max_ratio = _norm(max_ratio)
    norm_sum_ratio = _norm(sum_ratio)
    norm_succ = _norm(succ_count)

    # Deadline guard: if we're running out of time, return the proven heuristic.
    def _heuristic_score():
        return norm_cp * (1.0 + 0.6 * norm_max_ratio + 0.4 * norm_sum_ratio)

    if _remaining_budget() < 30.0:
        return _heuristic_score()

    # Local search over blending weights, evaluated by real SGS makespan.
    # score = w_cp * norm_cp * (1.0 + a * norm_max_ratio + b * norm_sum_ratio) + w_succ * norm_succ
    # We search (a, b, w_succ) with w_cp fixed at 1.0.
    best_score = _heuristic_score()
    best_ms = _serial_sgs(durations, demands, capacities, predecessors, best_score)

    # Candidate grid over (a, b, w_succ).
    a_vals = [0.0, 0.4, 0.6, 0.8, 1.2]
    b_vals = [0.0, 0.2, 0.4, 0.6]
    w_succ_vals = [0.0, 0.05, 0.1]

    # Precompute base = norm_cp, and the resource boost combos.
    base = norm_cp
    evaluated = 0
    max_evals = 300
    for a in a_vals:
        for b in b_vals:
            boost = 1.0 + a * norm_max_ratio + b * norm_sum_ratio
            for ws in w_succ_vals:
                if _remaining_budget() < 30.0:
                    break
                if evaluated >= max_evals:
                    break
                cand = base * boost + ws * norm_succ
                ms = _serial_sgs(durations, demands, capacities, predecessors, cand)
                evaluated += 1
                if ms < best_ms:
                    best_ms = ms
                    best_score = cand
            if _remaining_budget() < 30.0:
                break
        if _remaining_budget() < 30.0:
            break
        if evaluated >= max_evals:
            break

    # A second, finer pass around the best found weights if budget remains.
    if _remaining_budget() > 40.0 and evaluated < max_evals:
        # Refine around current best (a*, b*, ws*) with small steps.
        # Recover approximate a*, b*, ws* by re-scanning the grid cheaply is
        # unnecessary; instead do a coordinate refinement using the best_score
        # as anchor by perturbing the additive succ term and the boost scale.
        # We'll just try a few scaled variants of the current best.
        for scale in [0.8, 1.1, 1.25]:
            if _remaining_budget() < 30.0:
                break
            # Re-derive a candidate by scaling the resource boost portion.
            # Approximate: best_score = base*(1 + a*mr + b*sr) + ws*sc.
            # We can't easily decompose, so instead nudge the whole score's
            # resource component by recombining with a slightly different a,b.
            for a2 in [0.5, 0.7, 1.0]:
                for b2 in [0.3, 0.5]:
                    if _remaining_budget() < 30.0:
                        break
                    cand = base * (1.0 + a2 * norm_max_ratio + b2 * norm_sum_ratio)
                    ms = _serial_sgs(durations, demands, capacities, predecessors, cand)
                    if ms < best_ms:
                        best_ms = ms
                        best_score = cand

    return best_score
# EVOLVE-BLOCK-END
