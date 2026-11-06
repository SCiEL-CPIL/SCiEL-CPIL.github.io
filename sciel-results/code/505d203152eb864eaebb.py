"""Assembly Line Balancing - Model and Optimize approach (rewritten)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import defaultdict, deque


def _greedy_balance(times, predecessors, cycle_time, priority_scores):
    """
    Greedy station filling. Within each station, repeatedly place the
    highest-priority eligible task that fits until none fits (first-fit-decreasing
    style). Returns number of stations.
    """
    n = len(times)
    if n == 0:
        return 0

    successors = defaultdict(list)
    unassigned_preds = np.array([len(predecessors[i]) for i in range(n)], dtype=int)
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)

    # Feasibility: any single task exceeding cycle time => infeasible
    if np.any(times > cycle_time + 1e-9):
        return 10**9

    eligible = [i for i in range(n) if unassigned_preds[i] == 0]
    in_eligible = [False] * n
    for i in eligible:
        in_eligible[i] = True

    # Precompute a stable tie-breaker (task id) so sorting is deterministic
    order_key = np.arange(n, dtype=int)

    station_count = 1
    current_load = 0.0
    placed_total = 0

    while eligible:
        # Sort current eligible by priority desc, then task id asc
        candidates = sorted(eligible, key=lambda i: (-priority_scores[i], order_key[i]))
        placed_any = False
        for task in candidates:
            if times[task] <= cycle_time - current_load + 1e-9:
                current_load += times[task]
                eligible.remove(task)
                in_eligible[task] = False
                placed_any = True
                placed_total += 1
                for succ in successors[task]:
                    unassigned_preds[succ] -= 1
                    if unassigned_preds[succ] == 0:
                        eligible.append(succ)
                        in_eligible[succ] = True
                # A newly eligible task may also fit in THIS station; re-scan.
                # To keep it efficient, just continue scanning (candidates list
                # is stale for new tasks, so re-sort when a new task was added).
                if unassigned_preds[task] == 0:
                    pass
                # Re-sort to include newly eligible tasks for this station
                if len(eligible) > len(candidates):
                    candidates = sorted(eligible, key=lambda i: (-priority_scores[i], order_key[i]))
                # continue loop to try to place more in this station
                continue

        if not placed_any:
            # Station full (no eligible fits) -> start a new station
            station_count += 1
            current_load = 0.0

    return station_count


def _topo_and_features(times, predecessors, successors, n):
    """Topological order + critical path + downstream load + transitive counts."""
    in_degree = np.array([len(predecessors[i]) for i in range(n)], dtype=int)
    in_deg_copy = in_degree.copy()
    q = deque()
    for i in range(n):
        if in_deg_copy[i] == 0:
            q.append(i)

    topo_order = []
    while q:
        u = q.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_deg_copy[v] -= 1
            if in_deg_copy[v] == 0:
                q.append(v)

    if len(topo_order) != n:
        return None, None, None, None

    # Critical path: longest path from task to any sink
    cp = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        cp[u] = times[u]
        for v in successors[u]:
            val = times[u] + cp[v]
            if val > cp[u]:
                cp[u] = val

    # Downstream load (sum of times over all transitive successors incl self)
    dl = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        dl[u] = times[u]
        for v in successors[u]:
            dl[u] += dl[v]

    # Transitive successor count (number of all reachable successors)
    trans_succ = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        s = 0.0
        for v in successors[u]:
            s += 1.0 + trans_succ[v]
        trans_succ[u] = s

    return cp, dl, trans_succ, topo_order


def _build_heuristics(times, predecessors, successors, n, cycle_time):
    """Build a diverse set of candidate priority vectors."""
    cp, dl, trans_succ, topo_order = _topo_and_features(times, predecessors, successors, n)
    if cp is None:
        # Degenerate (shouldn't happen); fall back to time
        return [times.astype(float).copy()]

    num_succ = np.array([len(successors[i]) for i in range(n)], dtype=float)
    num_pred = np.array([len(predecessors[i]) for i in range(n)], dtype=float)

    cp_max = np.max(cp) + 1e-6
    dl_max = np.max(dl) + 1e-6
    ts_max = np.max(trans_succ) + 1e-6
    t_max = np.max(times) + 1e-6

    heuristics = []

    # 1. Critical path
    heuristics.append(cp.copy())
    # 2. Time + CP
    heuristics.append(times + cp)
    # 3. Downstream load
    heuristics.append(dl.copy())
    # 4. CP weighted by transitive successors (branching-aware CP)
    heuristics.append(cp * (1.0 + 0.2 * trans_succ / ts_max))
    # 5. Transitive successor count (branching complexity)
    heuristics.append(trans_succ.copy())
    # 6. Bottleneck slack: (cycle_time - time_i) / (1 + trans_succ) -> invert
    #    Favor tasks consuming most capacity relative to branching they unlock
    slack = (cycle_time - times) / (1.0 + trans_succ)
    heuristics.append(-slack.copy())
    # 7. Bottleneck slack with immediate successors
    slack_imm = (cycle_time - times) / (1.0 + num_succ)
    heuristics.append(-slack_imm.copy())
    # 8. CP / (1 + trans_succ)  -> criticality per branch
    heuristics.append(cp / (1.0 + trans_succ / ts_max))
    # 9. dl * (cp / cp_max)  -> load weighted by criticality
    heuristics.append(dl * (cp / cp_max))
    # 10. Pure time
    heuristics.append(times.copy())
    # 11. dl + cp
    heuristics.append(dl + cp)
    # 12. time * (cp / cp_max)
    heuristics.append(times * (cp / cp_max))
    # 13. CP - small penalty for many predecessors
    heuristics.append(cp - 0.05 * num_pred)
    # 14. Combined normalized: 0.5*CP + 0.3*DL + 0.2*trans_succ
    heuristics.append(0.5 * (cp / cp_max) + 0.3 * (dl / dl_max) + 0.2 * (trans_succ / ts_max))
    # 15. Rank-based on CP
    rank_cp = np.argsort(np.argsort(cp)).astype(float)
    heuristics.append(rank_cp)
    # 16. Rank-based on dl
    rank_dl = np.argsort(np.argsort(dl)).astype(float)
    heuristics.append(rank_dl)
    # 17. dl / (1 + trans_succ) -> load per branch
    heuristics.append(dl / (1.0 + trans_succ / ts_max))
    # 18. time + trans_succ (big tasks with lots of downstream)
    heuristics.append(times + 0.1 * trans_succ)

    return heuristics


def _local_search(priority, times, predecessors, cycle_time, n, deadline):
    """
    Pairwise-swap local search. First-improvement per pass, repeated until
    no improvement or deadline. Uses a fast greedy.
    """
    current_priority = priority.copy()
    current_score = _greedy_balance(times, predecessors, cycle_time, current_priority)
    best_priority = current_priority.copy()
    best_score = current_score

    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                return best_priority, best_score
            for j in range(i + 1, n):
                old_i, old_j = current_priority[i], current_priority[j]
                current_priority[i], current_priority[j] = old_j, old_i
                new_score = _greedy_balance(times, predecessors, cycle_time, current_priority)
                if new_score < current_score:
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_priority = current_priority.copy()
                    improved = True
                    if best_score <= 1:
                        return best_priority, best_score
                else:
                    current_priority[i], current_priority[j] = old_i, old_j
    return best_priority, best_score


def _random_walk(priority, times, predecessors, cycle_time, n, deadline):
    """
    Randomized perturbation + accept-improvement walk. Provides exploration
    beyond the local optimum found by local search.
    """
    current_priority = priority.copy()
    current_score = _greedy_balance(times, predecessors, cycle_time, current_priority)
    best_priority = current_priority.copy()
    best_score = current_score

    T = 1.0
    cooling = 0.95
    min_T = 0.02
    start = time.time()

    while T > min_T and time.time() < deadline:
        # Batch of random moves
        for _ in range(6):
            if time.time() > deadline:
                return best_priority, best_score
            i, j = random.sample(range(n), 2)
            old_i, old_j = current_priority[i], current_priority[j]
            current_priority[i], current_priority[j] = old_j, old_i
            new_score = _greedy_balance(times, predecessors, cycle_time, current_priority)
            delta = new_score - current_score
            if delta < 0 or (T > 0 and random.random() < np.exp(-delta / max(T, 1e-6))):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_priority = current_priority.copy()
                    if best_score <= 1:
                        return best_priority, best_score
            else:
                current_priority[i], current_priority[j] = old_i, old_j
        T *= cooling

    return best_priority, best_score


def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]

    n = len(times)
    if n == 0:
        return np.array([])

    start_time = time.time()
    deadline = start_time + 140.0  # safety margin under 150s kill

    # Build successor map
    successors = defaultdict(list)
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)

    # Build diverse heuristics
    heuristics = _build_heuristics(times, predecessors, successors, n, cycle_time)

    # Evaluate all heuristics quickly
    best_priority = None
    best_score = float('inf')
    scored = []
    for h in heuristics:
        if time.time() > deadline - 60:
            break
        score = _greedy_balance(times, predecessors, cycle_time, h)
        scored.append((score, h))
        if score < best_score:
            best_score = score
            best_priority = h.copy()
        if best_score <= 1:
            break

    if best_priority is None:
        best_priority = heuristics[0]
        best_score = _greedy_balance(times, predecessors, cycle_time, best_priority)

    # Local search from the best heuristic
    if time.time() < deadline - 20:
        best_priority, best_score = _local_search(
            best_priority, times, predecessors, cycle_time, n, deadline - 12
        )

    # Random walk / annealing from the best
    if time.time() < deadline - 6:
        best_priority, best_score = _random_walk(
            best_priority, times, predecessors, cycle_time, n, deadline - 3
        )

    # If time remains and best > 1, try local search from the 2nd/3rd best heuristic
    if time.time() < deadline - 8 and best_score > 1 and len(scored) >= 2:
        scored_sorted = sorted(scored, key=lambda x: x[0])
        for sc, h in scored_sorted[1:3]:
            if time.time() > deadline - 5:
                break
            if sc >= best_score:
                break
            # Quick local search from this alternative start
            pri, sc2 = _local_search(
                h, times, predecessors, cycle_time, n, deadline - 4
            )
            if sc2 < best_score:
                best_score = sc2
                best_priority = pri

    return best_priority
# EVOLVE-BLOCK-END
