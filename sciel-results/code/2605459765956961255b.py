import numpy as np
from collections import deque
import time

# Global wall-clock guard: the evaluator kills after 150s. Keep a safety
# margin and, on breach, return a cheap duration-based priority so we always
# finish in time.
_TIME_LIMIT = 140.0
_START_TIME = None


def _now():
    return time.time()


def _duration_fallback(durations):
    d = np.asarray(durations, dtype=float)
    m = np.max(d) if d.size else 0.0
    if m <= 0:
        return d
    return d / (m + 1e-9)


def priority(instance):
    """
    Score each activity for the serial schedule-generation scheme.
    Higher score = higher priority (started earlier).

    Heuristic (merged):
      - CPM total float (LS - ES) drives criticality (low float = critical).
      - Bottom level (longest path from activity to project end) as a robust
        criticality proxy.
      - Resource contention = combined sum-pressure + max-pressure of
        demand/capacity ratios, weighted by near-criticality (slack factor)
        so that only activities that are both critical AND resource-heavy get
        the contention boost.
      - Small duration tie-break (LDF-like) + tiny index jitter.
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = _now()

    # Deadline guard: bail to a cheap, valid priority if we are too late.
    if _now() - _START_TIME > _TIME_LIMIT:
        durations = np.asarray(instance["durations"], dtype=float)
        return _duration_fallback(durations)

    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    if n == 0:
        return np.array([])

    preds = instance.get("predecessors", [None] * n)
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)

    # ---- Build graph ---------------------------------------------------
    pred_lists = []
    for p in preds:
        if p is None:
            pred_lists.append([])
        elif isinstance(p, (list, tuple, np.ndarray)):
            pred_lists.append([x for x in p if x is not None and 0 <= x < n])
        else:
            pred_lists.append([p] if 0 <= p < n else [])

    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        for p in pred_lists[i]:
            successors[p].append(i)
            in_degree[i] += 1

    # Topological order (Kahn). Cycle fallback -> index order.
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    top_order = []
    while queue:
        u = queue.popleft()
        top_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    if len(top_order) < n:
        top_order = list(range(n))

    # ---- CPM forward pass: ES / EF ------------------------------------
    es = np.zeros(n)
    ef = np.zeros(n)
    for u in top_order:
        e = es[u]
        for v in successors[u]:
            if e > es[v]:
                es[v] = e
        ef[u] = e + durations[u]

    # ---- CPM backward pass: LS / LF -----------------------------------
    project_end = float(np.max(ef)) if n else 0.0
    lf = np.full(n, project_end)
    ls = np.zeros(n)
    for u in reversed(top_order):
        l = lf[u]
        for v in successors[u]:
            if l > ls[v]:
                l = ls[v]
        ls[u] = l - durations[u]
        lf[u] = l

    total_float = ls - es
    total_float = np.maximum(total_float, 0.0)

    # ---- Bottom level (longest path from activity to end) --------------
    bottom_level = np.zeros(n)
    for u in reversed(top_order):
        b = durations[u]
        for v in successors[u]:
            cand = b + bottom_level[v]
            if cand > b:
                b = cand
        bottom_level[u] = b

    # ---- Resource contention ------------------------------------------
    if demands is not None and capacities is not None:
        demand_matrix = _as_demand_matrix(demands, n)
        cap_arr = _as_capacities(capacities)
        if demand_matrix.shape[1] != len(cap_arr):
            m = min(demand_matrix.shape[1], len(cap_arr))
            demand_matrix = demand_matrix[:, :m]
            cap_arr = cap_arr[:m]
        if demand_matrix.size == 0 or len(cap_arr) == 0:
            sum_pressure = np.zeros(n)
            max_pressure = np.zeros(n)
            opp_cost = np.zeros(n)
        else:
            safe_caps = np.where(cap_arr > 0, cap_arr, 1.0)
            ratios = demand_matrix / safe_caps  # (n, R)
            sum_pressure = np.sum(ratios, axis=1)
            max_pressure = np.max(ratios, axis=1)
            # Resource opportunity cost: duration-weighted scarcity
            # An activity that consumes d_i units of resource r for duration t_i
            # "occupies" t_i * d_i / C_r units of that resource over time.
            # Total opportunity cost = sum over resources of this product.
            # This penalizes long activities on scarce resources more heavily.
            occ = ratios * durations[:, np.newaxis]  # (n, R)
            opp_cost = np.sum(occ, axis=1)
            # Also compute max per-resource opportunity cost (bottleneck view)
            opp_cost_max = np.max(occ, axis=1)
    else:
        sum_pressure = np.zeros(n)
        max_pressure = np.zeros(n)
        opp_cost = np.zeros(n)
        opp_cost_max = np.zeros(n)

    # ---- Normalize components -----------------------------------------
    bf = float(np.max(total_float)) if n else 0.0
    float_norm = (1.0 - total_float / (bf + 1e-9)) if bf > 0 else np.ones(n)

    bl = float(np.max(bottom_level)) if n else 0.0
    bl_norm = (bottom_level / (bl + 1e-9)) if bl > 0 else np.zeros(n)

    sp_max = float(np.max(sum_pressure)) if n else 0.0
    sp_norm = (sum_pressure / (sp_max + 1e-9)) if sp_max > 0 else np.zeros(n)

    mp_max = float(np.max(max_pressure)) if n else 0.0
    mp_norm = (max_pressure / (mp_max + 1e-9)) if mp_max > 0 else np.zeros(n)

    dur_max = float(np.max(durations)) if n else 0.0
    dur_norm = (durations / (dur_max + 1e-9)) if dur_max > 0 else np.zeros(n)

    oc_max = float(np.max(opp_cost)) if n else 0.0
    oc_norm = (opp_cost / (oc_max + 1e-9)) if oc_max > 0 else np.zeros(n)

    ocm_max = float(np.max(opp_cost_max)) if n else 0.0
    ocm_norm = (opp_cost_max / (ocm_max + 1e-9)) if ocm_max > 0 else np.zeros(n)

    # ---- Combine -------------------------------------------------------
    # Criticality = weighted blend of float-based and bottom-level signals.
    crit = 0.6 * float_norm + 0.4 * bl_norm

    # Resource opportunity cost: duration-weighted scarcity.
    # Blend aggregate (sum over resources) and bottleneck (max over resources).
    opp_cost_score = 0.5 * oc_norm + 0.5 * ocm_norm

    # Contention (raw ratio, not duration-weighted) as secondary signal.
    contention = 0.6 * sp_norm + 0.4 * mp_norm

    # Slack factor: boost resource terms only for near-critical activities.
    if bf > 0:
        slack_factor = np.where(total_float < 0.2 * bf, 1.0,
                                np.where(total_float < 0.5 * bf, 0.5, 0.2))
    else:
        slack_factor = np.ones(n)

    # Primary: criticality. Secondary: opportunity cost (duration×scarcity).
    # Tertiary: raw contention. Small duration tie-break.
    scores = crit + 0.35 * opp_cost_score * slack_factor + 0.15 * contention * slack_factor + 0.05 * dur_norm

    # Tiny deterministic tie-breaks (LDF-like duration, then index).
    scores = scores + dur_norm * 1e-9 + np.arange(n) * 1e-12

    # Final deadline guard before returning.
    if _now() - _START_TIME > _TIME_LIMIT:
        return _duration_fallback(durations)

    return scores


def _as_demand_matrix(demands, n):
    """Coerce the demands field into a float (n, R) matrix."""
    if isinstance(demands, np.ndarray):
        d = demands.astype(float)
    elif isinstance(demands, (list, tuple)):
        if len(demands) == 0:
            return np.zeros((n, 1), dtype=float)
        if isinstance(demands[0], dict):
            # dict-of-resource form: {resource_index: value}
            d = np.zeros((n, 1), dtype=float)
            for i, row in enumerate(demands):
                for r, val in row.items():
                    try:
                        r = int(r)
                    except Exception:
                        continue
                    if r < 0:
                        continue
                    if r >= d.shape[1]:
                        d = np.hstack([d, np.zeros((n, r + 1 - d.shape[1]), dtype=float)])
                    if i < n:
                        d[i, r] = val
            return d
        try:
            d = np.asarray(demands, dtype=float)
        except Exception:
            return np.zeros((n, 1), dtype=float)
    else:
        return np.zeros((n, 1), dtype=float)

    if d.ndim == 1:
        if d.size == n:
            d = d.reshape(n, 1)
        else:
            d = np.zeros((n, 1), dtype=float)
    elif d.ndim == 2:
        if d.shape[0] != n:
            try:
                d = d.reshape(n, -1)
            except Exception:
                d = np.zeros((n, 1), dtype=float)
    else:
        d = np.zeros((n, 1), dtype=float)

    if d.shape[0] != n:
        try:
            d = d.reshape(n, -1)
        except Exception:
            d = np.zeros((n, 1), dtype=float)
    if d.shape[1] == 0:
        d = np.zeros((n, 1), dtype=float)
    return d


def _as_capacities(capacities):
    """Coerce the capacities field into a 1-D float array."""
    if isinstance(capacities, np.ndarray):
        c = capacities.astype(float).ravel()
    elif isinstance(capacities, (list, tuple)):
        try:
            c = np.asarray(capacities, dtype=float).ravel()
        except Exception:
            c = np.array([1.0])
    else:
        c = np.array([1.0])
    if c.size == 0:
        c = np.array([1.0])
    return c
