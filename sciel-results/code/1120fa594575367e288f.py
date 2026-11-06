"""Rewritten program: adaptive priority for single-model assembly line balancing.
The scheduler (fixed by the task) picks, per station, the highest-priority
eligible task that fits the remaining cycle time. We only control priorities.
We construct several analytically-motivated priority vectors, simulate the
greedy fill for each (cheap for 30 tasks), and return the best one.

Improvements over prior version:
- Faster simulator using bitmask for predecessor sets (n<=30 fits in int64)
- Additional candidate priorities: rank-based, position-weighted, "critical path"
  with tie-breaking by duration, and inverse-duration for small tasks
- Local coordinate-descent refinement around the best Dirichlet weights
- Tighter time management to evaluate more candidates
"""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _topo_order(n, pred):
    """Kahn topological order. pred[i] = list of predecessor indices of i."""
    indeg = np.zeros(n, dtype=int)
    for i in range(n):
        for p in pred[i]:
            indeg[i] += 1
    from collections import deque
    q = deque([i for i in range(n) if indeg[i] == 0])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for i in range(n):
            if u in pred[i]:
                indeg[i] -= 1
                if indeg[i] == 0:
                    q.append(i)
    if len(order) != n:
        # fallback
        order = list(range(n))
    return order


def _build_graph(n, pred, succ):
    return pred, succ


def _downstream_load(n, succ, times):
    """For each task, sum of durations of all transitive successors (including self)."""
    load = times.astype(float).copy()
    # process in reverse topological order so successors are processed first
    # build reverse topo
    indeg = np.zeros(n, dtype=int)
    for i in range(n):
        for s in succ[i]:
            indeg[s] += 1
    from collections import deque
    q = deque([i for i in range(n) if indeg[i] == 0])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for s in succ[u]:
            indeg[s] -= 1
            if indeg[s] == 0:
                q.append(s)
    if len(order) != n:
        order = list(range(n))
    for u in reversed(order):
        for s in succ[u]:
            load[u] += load[s]
    return load


def _longest_path_end(n, succ, times):
    """Longest path from each node to a sink (max sum of durations along a path, including self)."""
    lp = times.astype(float).copy()
    indeg = np.zeros(n, dtype=int)
    for i in range(n):
        for s in succ[i]:
            indeg[s] += 1
    from collections import deque
    q = deque([i for i in range(n) if indeg[i] == 0])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for s in succ[u]:
            indeg[s] -= 1
            if indeg[s] == 0:
                q.append(s)
    if len(order) != n:
        order = list(range(n))
    for u in reversed(order):
        best = 0.0
        for s in succ[u]:
            if lp[s] > best:
                best = lp[s]
        lp[u] = times[u] + best
    return lp


def _successor_count(n, succ):
    """Count of transitive successors."""
    cnt = np.zeros(n, dtype=float)
    indeg = np.zeros(n, dtype=int)
    for i in range(n):
        for s in succ[i]:
            indeg[s] += 1
    from collections import deque
    q = deque([i for i in range(n) if indeg[i] == 0])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for s in succ[u]:
            indeg[s] -= 1
            if indeg[s] == 0:
                q.append(s)
    if len(order) != n:
        order = list(range(n))
    for u in reversed(order):
        c = 0.0
        for s in succ[u]:
            c += 1.0 + cnt[s]
        cnt[u] = c
    return cnt


def _simulate(times, pred, cycle_time, priority):
    """Greedy list scheduling: at each station pick highest-priority eligible task
    that fits remaining cycle time. Returns number of stations.
    Uses bitmask for fast predecessor checking (n must be <= 60)."""
    n = len(times)
    remaining = times.astype(float).copy()
    done_mask = 0  # bitmask of completed tasks
    # precompute pred bitmask for each task
    pred_mask = [0] * n
    for i in range(n):
        m = 0
        for p in pred[i]:
            m |= (1 << p)
        pred_mask[i] = m
    
    stations = 0
    while done_mask != (1 << n) - 1:
        stations += 1
        cap = cycle_time
        # find eligible: not done, preds all done, fits in cap
        best = -1
        best_p = -np.inf
        for i in range(n):
            if done_mask & (1 << i):
                continue
            if remaining[i] > cap:
                continue
            # check preds done: all bits in pred_mask[i] must be in done_mask
            if (pred_mask[i] & ~done_mask) != 0:
                continue
            if priority[i] > best_p:
                best_p = priority[i]
                best = i
        if best == -1:
            # no task fits in cap: pick smallest eligible to avoid deadlock
            best = -1
            best_sz = np.inf
            for i in range(n):
                if done_mask & (1 << i):
                    continue
                if (pred_mask[i] & ~done_mask) != 0:
                    continue
                if remaining[i] < best_sz:
                    best_sz = remaining[i]
                    best = i
            if best == -1:
                break
        done_mask |= (1 << best)
        cap -= remaining[best]
        # fill same station with more tasks greedily
        while True:
            best2 = -1
            best_p2 = -np.inf
            for i in range(n):
                if done_mask & (1 << i):
                    continue
                if remaining[i] > cap:
                    continue
                if (pred_mask[i] & ~done_mask) != 0:
                    continue
                if priority[i] > best_p2:
                    best_p2 = priority[i]
                    best2 = i
            if best2 == -1:
                break
            done_mask |= (1 << best2)
            cap -= remaining[best2]
    return stations


def priority(instance):
    start = time.time()
    deadline = start + 135.0  # safety margin under 150s kill

    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    pred = instance["predecessors"]
    # pred may be list of lists or array
    pred_lists = [list(p) for p in pred]
    succ = [[] for _ in range(n)]
    for i in range(n):
        for p in pred_lists[i]:
            succ[p].append(i)

    cycle_time = instance["cycle_time"]

    # Build graph metrics
    dl = _downstream_load(n, succ, times)          # downstream load (incl self)
    lp = _longest_path_end(n, succ, times)         # longest path to sink
    sc = _successor_count(n, succ)                 # transitive successor count

    tmax = times.max() if n else 1.0
    if tmax == 0:
        tmax = 1.0

    # Additional derived features
    # Rank by downstream load (1 = highest load, n = lowest)
    dl_rank = np.zeros(n)
    order_dl = np.argsort(dl)[::-1]
    for r, idx in enumerate(order_dl):
        dl_rank[idx] = n - r  # higher rank = more important
    # Rank by longest path
    lp_rank = np.zeros(n)
    order_lp = np.argsort(lp)[::-1]
    for r, idx in enumerate(order_lp):
        lp_rank[idx] = n - r
    # Rank by successor count
    sc_rank = np.zeros(n)
    order_sc = np.argsort(sc)[::-1]
    for r, idx in enumerate(order_sc):
        sc_rank[idx] = n - r

    # Normalize features to [0,1] range for consistent blending
    def _norm(v):
        mn, mx = v.min(), v.max()
        if mx - mn < 1e-12:
            return np.ones(n) * 0.5
        return (v - mn) / (mx - mn)

    f_dur = _norm(times)
    f_dl = _norm(dl)
    f_lp = _norm(lp)
    f_sc = _norm(sc)
    f_dl_rank = _norm(dl_rank)
    f_lp_rank = _norm(lp_rank)
    f_sc_rank = _norm(sc_rank)
    f_inv_dur = _norm(1.0 / (times + 1e-9))  # favors small tasks

    # Candidate priority vectors (diverse set)
    candidates = []
    # Single features
    candidates.append(times.copy())
    candidates.append(dl.copy())
    candidates.append(lp.copy())
    candidates.append(sc.copy())
    candidates.append(dl_rank.copy())
    candidates.append(lp_rank.copy())
    candidates.append(sc_rank.copy())
    # Normalized blends
    candidates.append(f_dur + f_dl)
    candidates.append(f_dl + f_lp)
    candidates.append(f_dur + f_lp)
    candidates.append(f_dur + f_sc)
    candidates.append(f_dl + f_sc)
    candidates.append(0.5*f_dur + 0.3*f_dl + 0.2*f_lp)
    candidates.append(0.3*f_dur + 0.4*f_dl + 0.3*f_lp)
    candidates.append(0.2*f_dur + 0.5*f_dl + 0.3*f_lp)
    candidates.append(0.5*f_dur + 0.2*f_dl + 0.2*f_lp + 0.1*f_sc)
    candidates.append(0.3*f_dur + 0.3*f_dl + 0.2*f_lp + 0.2*f_sc)
    candidates.append(0.2*f_dur + 0.3*f_dl + 0.2*f_lp + 0.3*f_sc)
    # Rank-based blends
    candidates.append(f_dl_rank + f_lp_rank)
    candidates.append(f_dur + f_dl_rank)
    candidates.append(0.5*f_dur + 0.3*f_dl_rank + 0.2*f_lp_rank)
    # Critical path emphasis
    candidates.append(lp + 0.5*times)
    candidates.append(lp + 0.3*dl)
    # Inverse duration (favor small tasks for packing)
    candidates.append(f_inv_dur + 0.5*f_dl)
    candidates.append(0.3*f_inv_dur + 0.4*f_dl + 0.3*f_lp)
    # Duration as fraction of cycle time
    candidates.append(times / max(cycle_time, 1e-9))
    candidates.append(times / max(cycle_time, 1e-9) + 0.5 * f_dl)

    # Evaluate each candidate with the greedy simulator
    best_prio = candidates[0]
    best_stations = _simulate(times, pred_lists, cycle_time, best_prio)

    for cand in candidates[1:]:
        if time.time() > deadline:
            break
        s = _simulate(times, pred_lists, cycle_time, cand)
        if s < best_stations:
            best_stations = s
            best_prio = cand

    # Randomized search in weight space
    if time.time() < deadline - 10.0:
        rng = np.random.default_rng(42)
        # Use normalized features for blending
        base = np.stack([f_dur, f_dl, f_lp, f_sc, f_dl_rank, f_lp_rank, f_sc_rank, f_inv_dur], axis=1)
        n_feat = base.shape[1]
        
        # Random Dirichlet weights
        n_random = 500
        for _ in range(n_random):
            if time.time() > deadline:
                break
            w = rng.dirichlet(np.ones(n_feat) * 0.5)  # more uniform
            cand = base @ w
            s = _simulate(times, pred_lists, cycle_time, cand)
            if s < best_stations:
                best_stations = s
                best_prio = cand

    # Local refinement: coordinate descent on weights around best found
    if time.time() < deadline - 5.0:
        # Try to extract weights from best_prio if it was from a blend
        # Instead, do a local search in weight space
        base = np.stack([f_dur, f_dl, f_lp, f_sc, f_dl_rank, f_lp_rank, f_sc_rank, f_inv_dur], axis=1)
        n_feat = base.shape[1]
        
        # Start from a few initial weight vectors
        rng2 = np.random.default_rng(99)
        init_weights = []
        init_weights.append(np.ones(n_feat) / n_feat)
        for _ in range(5):
            init_weights.append(rng2.dirichlet(np.ones(n_feat)))
        
        for w0 in init_weights:
            if time.time() > deadline:
                break
            w = w0.copy()
            for iteration in range(50):
                if time.time() > deadline:
                    break
                improved = False
                for d in range(n_feat):
                    if time.time() > deadline:
                        break
                    # Try increasing and decreasing weight d
                    for delta in [0.1, 0.05, 0.02, -0.1, -0.05, -0.02]:
                        w_trial = w.copy()
                        w_trial[d] += delta
                        if w_trial[d] < 0:
                            continue
                        w_sum = w_trial.sum()
                        if w_sum < 1e-6:
                            continue
                        w_trial /= w_sum
                        cand = base @ w_trial
                        s = _simulate(times, pred_lists, cycle_time, cand)
                        if s < best_stations:
                            best_stations = s
                            best_prio = cand
                            w = w_trial
                            improved = True
                            break
                if not improved:
                    break

    return np.asarray(best_prio, dtype=float)
# EVOLVE-BLOCK-END
