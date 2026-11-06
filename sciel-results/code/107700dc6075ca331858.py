"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = time.time() + 140.0  # safety margin well under the 150s wall-clock kill


def _resolve_pred(preds, n):
    """Coerce predecessor spec into a list of lists of int indices."""
    if isinstance(preds, np.ndarray):
        arr = preds
        if arr.ndim == 1:
            # maybe a flat list with -1 for none, or object array of lists
            if arr.dtype == object:
                return [list(p) for p in arr]
            return [[int(x) for x in p if x is not None and x >= 0] for p in arr]
        if arr.dtype == object:
            return [list(p) for p in arr]
        # numeric 2D: treat each row as pred indices, -1/None padding removed
        return [[int(x) for x in row if x >= 0] for row in arr]
    if isinstance(preds, dict):
        out = [[] for _ in range(n)]
        for k, v in preds.items():
            i = int(k)
            if 0 <= i < n:
                if isinstance(v, (list, tuple, np.ndarray)):
                    out[i] = [int(x) for x in v]
                else:
                    out[i] = [int(v)]
        return out
    # list/tuple of lists
    out = []
    for p in preds:
        if isinstance(p, (list, tuple, np.ndarray)):
            out.append([int(x) for x in p])
        else:
            out.append([int(p)])
    return out


def _topo_order(n, preds):
    """Kahn topological order; returns list of indices (0..n-1)."""
    indeg = np.zeros(n, dtype=int)
    succ = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            if 0 <= p < n:
                indeg[i] += 1
                succ[p].append(i)
    stack = [i for i in range(n) if indeg[i] == 0]
    order = []
    while stack:
        u = stack.pop()
        order.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                stack.append(v)
    if len(order) < n:
        # cycle: append the rest to avoid crash
        seen = set(order)
        for i in range(n):
            if i not in seen:
                order.append(i)
    return order


def priority(instance):
    dur = np.asarray(instance["durations"], dtype=float)
    n = len(dur)
    if n == 0:
        return np.zeros(0, dtype=float)

    preds = _resolve_pred(instance.get("predecessors", []), n)
    order = _topo_order(n, preds)
    succ = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            if 0 <= p < n:
                succ[p].append(i)

    # longest duration (critical path) from each node to the end
    longest = np.zeros(n, dtype=float)
    for u in reversed(order):
        best = 0.0
        for v in succ[u]:
            if longest[v] > best:
                best = longest[v]
        longest[u] = dur[u] + best

    # resource pressure: sum over resources of demand/capacity
    demands = instance.get("demands", None)
    caps = instance.get("capacities", None)
    pressure = np.zeros(n, dtype=float)
    if demands is not None and caps is not None:
        D = np.atleast_2d(np.asarray(demands, dtype=float))
        if D.shape[0] == n:
            C = np.asarray(caps, dtype=float)
            if C.ndim == 1:
                C = C.reshape(1, -1)
            C = C.T  # shape (n, R)
            ratio = D / np.maximum(C, 1e-9)
            pressure = ratio.sum(axis=1)
            # add a mild convexity so scarce resources dominate
            pressure = pressure + 0.5 * np.maximum(ratio, 0.0).max(axis=1)

    # Resource-Weighted Critical Path: multiplicative weighting by resource pressure
    # A node that is deep on the critical path AND consumes a scarce resource is
    # more impactful to schedule early (frees the bottleneck sooner).
    w = 1.0 + pressure
    score = longest * w

    # small deterministic tie-breaker to keep ordering stable
    score = score + np.linspace(0.0, 1e-6, n)

    if time.time() > _DEADLINE:
        # safety: return something valid immediately
        return np.asarray(dur, dtype=float)

    return np.asarray(score, dtype=float)
# EVOLVE-BLOCK-END
