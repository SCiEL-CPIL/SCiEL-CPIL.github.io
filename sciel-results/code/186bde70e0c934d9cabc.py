"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque


def _as_index_list(pred):
    """Return a list of int indices from various predecessor encodings."""
    if pred is None:
        return []
    if isinstance(pred, (list, tuple, set)):
        out = []
        for p in pred:
            try:
                out.append(int(p))
            except (ValueError, TypeError):
                continue
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
    deadline = t0 + 110.0  # Safety margin below 120s limit

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

    # Build successors and in-degree
    succ = [[] for _ in range(n)]
    indeg = [0] * n
    for i in range(n):
        for p in pred_lists[i]:
            if 0 <= p < n and p != i:
                succ[p].append(i)
                indeg[i] += 1

    # Topological order (Kahn's algorithm)
    indeg2 = indeg[:]
    q = deque([i for i in range(n) if indeg2[i] == 0])
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for v in succ[u]:
            indeg2[v] -= 1
            if indeg2[v] == 0:
                q.append(v)

    # If topological sort is incomplete (cycle), fallback to LPT
    if len(topo) < n:
        return dur.copy()

    # Compute critical path length (longest path from node to any sink)
    cpl = np.zeros(n, dtype=float)
    for u in reversed(topo):
        best = 0.0
        for v in succ[u]:
            if cpl[v] > best:
                best = cpl[v]
        cpl[u] = dur[u] + best

    # Compute resource pressure
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
                # Use max ratio (bottleneck resource) rather than sum
                pressure = np.max(dem_arr / cap_safe, axis=1)
                pressure = np.where(np.isfinite(pressure), pressure, 0.0)

    # Check time
    if time.time() > deadline:
        # Fallback: return combined score
        scores = cpl + dur + 0.3 * pressure
        return np.where(np.isfinite(scores), scores, 0.0).astype(float)

    # Normalize components to [0, 1] for weighted combination
    if np.max(cpl) > 0:
        cpl_norm = cpl / np.max(cpl)
    else:
        cpl_norm = np.zeros(n)

    if np.max(dur) > 0:
        dur_norm = dur / np.max(dur)
    else:
        dur_norm = np.zeros(n)

    if np.max(pressure) > 0:
        pres_norm = pressure / np.max(pressure)
    else:
        pres_norm = np.zeros(n)

    # Weighted combination: emphasize critical path most, then duration, then resource pressure
    alpha = 0.55  # Weight for critical path
    beta = 0.30   # Weight for duration
    gamma = 0.15  # Weight for resource pressure

    scores = alpha * cpl_norm + beta * dur_norm + gamma * pres_norm
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores.astype(float)
# EVOLVE-BLOCK-END
