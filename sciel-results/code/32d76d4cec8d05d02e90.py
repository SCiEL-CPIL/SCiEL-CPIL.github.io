"""Resource-Constrained Project Scheduling: fluid-flow bottleneck-pressure priority.

Family: dynamic capacity depletion / peak-congestion (fluid flow) heuristic.
priority(instance) -> one score per activity (higher = scheduled earlier).
"""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 140.0  # safety margin under the 150 s wall-clock kill


def _topo_order(pred_lists, n):
    """Return a topological order (list of ints)."""
    indeg = np.zeros(n, dtype=int)
    succ = [[] for _ in range(n)]
    for i in range(n):
        for p in pred_lists[i]:
            succ[p].append(i)
            indeg[i] += 1
    stack = [i for i in range(n) if indeg[i] == 0]
    order = []
    while stack:
        u = stack.pop()
        order.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                stack.append(v)
    if len(order) != n:  # cycle guard: fall back to natural order
        order = list(range(n))
    return order, succ


def priority(instance):
    t0 = time.time()
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)

    preds = instance.get("predecessors", None)
    if preds is None:
        preds = [[] for _ in range(n)]
    pred_lists = []
    for i in range(n):
        p = preds[i]
        if p is None:
            p = []
        pred_lists.append([int(x) for x in p])

    order, succ = _topo_order(pred_lists, n)

    # ---- Critical-path tail: longest path from each node to the sink ----
    tail = np.zeros(n, dtype=float)
    for u in reversed(order):
        best = 0.0
        for v in succ[u]:
            best = max(best, tail[v])
        tail[u] = durations[u] + best
    proj_len = float(max(tail)) if n else 1.0

    # ---- Per-resource normalized load pressure (fluid area / capacity) ----
    demands = np.asarray(instance.get("demands", np.zeros((n, 0))), dtype=float)
    capacities = np.asarray(instance.get("capacities", instance.get("resource_capacities", [])), dtype=float)
    nr = demands.shape[1] if demands.ndim == 2 else 0
    if nr == 0:
        res_pressure = np.zeros(n, dtype=float)
    else:
        if capacities.size < nr:
            capacities = np.full(nr, 1.0)
        # load area per resource per activity = demand * duration
        area = demands * durations[:, None]  # (n, nr)
        cap = np.where(capacities > 0, capacities, 1.0)
        norm = area / cap[None, :]
        # smooth peak-congestion surrogate: log-sum-exp over resources
        m = np.max(norm, axis=1, keepdims=True)
        lse = m[:, 0] + np.log(np.sum(np.exp(norm - m), axis=1))
        res_pressure = lse

    # ---- Quick greedy forward simulation to estimate real contention ----
    # Estimate, for each activity, how much it is delayed by resource scarcity.
    # This refines the static pressure with a contention-aware term (polish).
    if time.time() - t0 > _DEADLINE - 5.0:
        # out of time: return best-so-far combined score
        score = tail + res_pressure
        return score

    # Greedy serial SGS using current (tail+pressure) ranking to estimate
    # per-activity start delay caused by resource contention.
    rank = np.argsort(-(tail + res_pressure))
    start = np.zeros(n, dtype=float)
    finish = np.zeros(n, dtype=float)
    done = np.zeros(n, dtype=bool)
    if nr > 0:
        res_used = np.zeros(nr, dtype=float)
        # event list of (time, activity) releases
        releases = []
        for a in range(n):
            releases.append((0.0, a))
    sched_order = []
    remaining = set(range(n))
    # simple loop
    while remaining:
        # eligible: all preds done
        eligible = [a for a in remaining if all(done[p] for p in pred_lists[a])]
        if not eligible:
            break
        # pick highest rank (smallest position in rank) among those that fit resources
        eligible_sorted = sorted(eligible, key=lambda a: list(rank).index(a))
        picked = None
        for a in eligible_sorted:
            if nr == 0:
                picked = a
                break
            if np.all(res_used + demands[a] <= capacities + 1e-9):
                picked = a
                break
        if picked is None:
            # resource deadlock: wait until some release
            break
        # compute start time
        st = 0.0
        for p in pred_lists[picked]:
            st = max(st, finish[p])
        if nr > 0:
            # find earliest time at st where resources free up
            # gather release times of currently-running activities
            busy = [(finish[a], a) for a in range(n) if done[a] and finish[a] > st]
            # simulate
            t = st
            while True:
                # remove releases at <= t
                still = [(ft, a) for (ft, a) in busy if ft > t]
                if np.all(res_used_at_t(still, demands, capacities) + demands[picked] <= capacities + 1e-9):
                    break
                # advance to next release
                nxt = min(ft for (ft, a) in still) if still else t + 1e-9
                t = nxt
                busy = still
            st = t
        start[picked] = st
        finish[picked] = st + durations[picked]
        done[picked] = True
        remaining.discard(picked)
        sched_order.append(picked)
        if time.time() - t0 > _DEADLINE - 5.0:
            break

    # contention delay = start - earliest possible (tail-based) start
    # earliest start ignoring resources = max finish of preds (computed in sim already)
    # Use start as a proxy for how late it was pushed; higher start => more contention.
    delay = start.copy()
    # normalize delay
    dmax = float(np.max(delay)) if n else 1.0
    if dmax > 0:
        delay = delay / dmax

    # ---- Final combined smooth score (polish) ----
    # weight critical tail + resource pressure + contention delay
    score = tail + 0.5 * res_pressure + 0.3 * delay
    return score


def res_used_at_t(busy, demands, capacities):
    total = np.zeros(capacities.shape, dtype=float)
    for (ft, a) in busy:
        total += demands[a]
    return total
# EVOLVE-BLOCK-END
