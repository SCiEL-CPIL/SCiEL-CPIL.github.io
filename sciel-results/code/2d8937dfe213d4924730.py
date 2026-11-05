"""Assembly line balancing: return a per-task priority that minimizes stations
under the greedy 'highest-priority eligible task that fits' rule."""
# EVOLVE-BLOCK-START
import time
import heapq
import numpy as np

_DEADLINE = 150.0
_START = time.time()


def _time_left():
    return _DEADLINE - 15.0 - (time.time() - _START)


def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    if n == 0:
        return np.array([], dtype=float)
    preds = instance.get("predecessors", None)
    if preds is None:
        preds = [[] for _ in range(n)]
    preds = [list(p) for p in preds]

    # successors / in-degree
    succ = [[] for _ in range(n)]
    indeg = [0] * n
    for i in range(n):
        for p in preds[i]:
            if 0 <= p < n and p != i:
                succ[p].append(i)
                indeg[i] += 1

    # topological order (Kahn)
    zero = [i for i in range(n) if indeg[i] == 0]
    indeg2 = indeg[:]
    topo = []
    while zero:
        i = zero.pop()
        topo.append(i)
        for j in succ[i]:
            indeg2[j] -= 1
            if indeg2[j] == 0:
                zero.append(j)
    if len(topo) != n:  # cycle: fall back to static heuristic
        return times.astype(float)
    pos = [0] * n
    for k, t in enumerate(topo):
        pos[t] = k

    cycle = float(instance.get("cycle_time", 40))

    # greedy completion from a given set of done tasks
    def greedy_complete(done_set):
        done = set(done_set)
        avail = [i for i in range(n) if i not in done and all(p in done for p in preds[i])]
        stations = 0
        while avail:
            stations += 1
            cap = cycle
            # pick best (largest) task that fits
            chosen = None
            best_t = -1.0
            for i in avail:
                if times[i] <= cap and times[i] > best_t:
                    best_t = times[i]
                    chosen = i
            if chosen is None:
                # nothing fits; start new station with the largest remaining
                chosen = max(avail, key=lambda i: times[i])
            avail.remove(chosen)
            done.add(chosen)
            cap -= times[chosen]
            for j in succ[chosen]:
                if j not in done and all(p in done for p in preds[j]):
                    avail.append(j)
        return stations

    # baseline
    best_order = list(topo)
    best_cost = greedy_complete(set(range(n)))

    # beam search over task sequences minimizing station count
    if _time_left() > 2.0 and n > 1:
        beam = [((), frozenset())]  # (sequence, done_frozenset)
        BEAM = 100
        LA = 12
        for step in range(n):
            if _time_left() < 1.5:
                break
            cand = []
            for seq, done in beam:
                avail = [i for i in range(n)
                         if i not in done and all(p in done for p in preds[i])]
                if not avail:
                    break
                for a in avail:
                    ndone = done | frozenset((a,))
                    # look-ahead: greedy complete from ndone but only LA more tasks
                    done_set = set(ndone)
                    remaining = [i for i in range(n) if i not in done_set]
                    # quick lower-bound: stations so far = step+1
                    lb = step + 1
                    # look-ahead greedy for a few tasks
                    done_la = set(done_set)
                    avail_la = [i for i in remaining if all(p in done_la for p in preds[i])]
                    la_stations = 0
                    for _ in range(min(LA, len(remaining))):
                        if not avail_la:
                            break
                        la_stations += 1
                        cap = cycle
                        ch = None
                        bt = -1.0
                        for i in avail_la:
                            if times[i] <= cap and times[i] > bt:
                                bt = times[i]
                                ch = i
                        if ch is None:
                            ch = max(avail_la, key=lambda i: times[i])
                        avail_la.remove(ch)
                        done_la.add(ch)
                        cap -= times[ch]
                        for j in succ[ch]:
                            if j not in done_la and all(p in done_la for p in preds[j]):
                                avail_la.append(j)
                        if not avail_la:
                            break
                    # remaining tasks need at least ceil(rem_time / cycle) more stations
                    rem_time = sum(times[i] for i in range(n) if i not in done_la)
                    import math
                    rem_lb = int(math.ceil(rem_time / cycle - 1e-9)) if rem_time > 0 else 0
                    cost = la_stations + rem_lb
                    cand.append((cost, len(cand), seq + (a,), ndone))
            if not cand:
                break
            cand.sort(key=lambda x: (x[0], x[1]))
            # dedupe by done set
            seen = set()
            newbeam = []
            for cost, _, seq, ndone in cand:
                if ndone in seen:
                    continue
                seen.add(ndone)
                newbeam.append((seq, ndone))
                if len(newbeam) >= BEAM:
                    break
            beam = newbeam
            # evaluate current best full completion from beam heads occasionally
            for seq, done in beam[:5]:
                c = greedy_complete(set(done))
                if c < best_cost:
                    best_cost = c
                    best_order = list(seq)

    # final: build priority = position in best_order (earlier = higher priority)
    # ensure topological validity: best_order should already be topo-valid.
    # Safety: if best_order not topo-valid, merge with topo.
    # Build priority: task at index 0 gets highest priority.
    prio = np.zeros(n, dtype=float)
    for k, t in enumerate(best_order):
        prio[t] = float(n - k)  # earlier -> larger -> higher priority
    return prio
# EVOLVE-BLOCK-END
