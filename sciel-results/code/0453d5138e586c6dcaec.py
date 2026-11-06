"""Serial schedule-generation priority: critical-path + resource-pressure + duration."""
import time
import numpy as np

# safety deadline (s)
_DEADLINE_S = 110.0


def _as_index_list(pred):
    """Return a list of int indices from various predecessor encodings."""
    if pred is None:
        return []
    if isinstance(pred, (list, tuple, set)):
        out = []
        for p in pred:
            out.append(int(p))
        return out
    if isinstance(pred, (int, np.integer)):
        return [int(pred)]
    if isinstance(pred, np.ndarray):
        if pred.size == 0:
            return []
        if pred.ndim == 0:
            return [int(pred)]
        return [int(x) for x in np.asarray(pred).ravel()]
    return []


def _get_key(instance, names, default=None):
    for n in names:
        if n in instance:
            return instance[n]
    return default


def priority(instance):
    t0 = time.time()

    dur = _get_key(instance, ["durations", "duration"], None)
    if dur is None:
        raise ValueError("durations missing")
    dur = np.asarray(dur, dtype=float).ravel()
    n = len(dur)

    preds = _get_key(instance, ["predecessors", "predecessor", "preds"], None)
    if preds is None:
        preds = [None] * n
    if isinstance(preds, (list, tuple, np.ndarray)) and len(preds) == n:
        pred_lists = [_as_index_list(preds[i]) for i in range(n)]
    else:
        pred_lists = [_as_index_list(preds)] * n

    # successors
    succ = [[] for _ in range(n)]
    indeg = [0] * n
    for i in range(n):
        for p in pred_lists[i]:
            if 0 <= p < n and p != i:
                succ[p].append(i)
                indeg[i] += 1

    # longest path from node to any sink (successor length) via DP on reverse topological order
    succ_len = np.zeros(n, dtype=float)
    # topological order (Kahn)
    from collections import deque
    q = deque([i for i in range(n) if indeg[i] == 0])
    indeg2 = indeg[:]
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for v in succ[u]:
            indeg2[v] -= 1
            if indeg2[v] == 0:
                q.append(v)
    # reverse topo
    for u in reversed(topo):
        best = 0.0
        for v in succ[u]:
            cand = succ_len[v]
            if cand > best:
                best = cand
        succ_len[u] = best

    # resource pressure
    dem = _get_key(instance, ["demands", "demand", "resource_demands"], None)
    cap = _get_key(instance, ["capacities", "capacity", "resource_capacities"], None)
    pressure = np.zeros(n, dtype=float)
    if dem is not None and cap is not None:
        cap_arr = np.asarray(cap, dtype=float).ravel()
        if cap_arr.size:
            cap_safe = np.where(cap_arr > 0, cap_arr, 1.0)
            dem_arr = np.asarray(dem, dtype=float)
            if dem_arr.ndim == 1:
                dem_arr = dem_arr.reshape(n, -1)
            elif dem_arr.ndim >= 2:
                pass
            if dem_arr.shape[0] == n and dem_arr.shape[1] >= 1:
                if dem_arr.shape[1] != cap_safe.shape[0]:
                    m = min(dem_arr.shape[1], cap_safe.shape[0])
                    dem_arr = dem_arr[:, :m]
                    cap_safe = cap_safe[:m]
                pressure = np.sum(dem_arr / cap_safe, axis=1)
                pressure = np.where(np.isfinite(pressure), pressure, 0.0)

    # deadline guard (cheap; computation above is O(n+e))
    if (time.time() - t0) > _DEADLINE_S:
        # fallback: return what we have so far
        return (succ_len + dur + 0.5 * pressure).astype(float)

    # Structural Change: Resource-Weighted Critical Path
    # Weight each activity's duration by its resource pressure to reflect bottleneck impact
    # Higher pressure => higher effective duration => higher priority if on critical path
    alpha_res = 0.5
    w_dur = dur * (1.0 + alpha_res * pressure)
    w_dur = np.where(np.isfinite(w_dur), w_dur, dur)

    # Recompute longest path (critical path length) using weighted durations
    # succ_len[u] will now store the total weighted duration of the critical path starting at u
    # We reuse the topo order and succ structure
    # Note: The previous succ_len calculation was: succ_len[u] = max(succ_len[v])
    # And prio = succ_len + dur.
    # Now we want prio = w_dur[u] + max(w_dur[v] + ...).
    # So we update succ_len to include the node's own weighted duration.
    
    # Reset succ_len to 0 to be safe, though we overwrite in the loop
    succ_len = np.zeros(n, dtype=float)
    
    for u in reversed(topo):
        best = 0.0
        for v in succ[u]:
            cand = succ_len[v]
            if cand > best:
                best = cand
        succ_len[u] = w_dur[u] + best

    prio = succ_len
    prio = np.where(np.isfinite(prio), prio, 0.0)
    return prio.astype(float)
