"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def priority(instance):
    """Return one priority score per task (higher = scheduled earlier)."""
    t0 = time.time()
    DEADLINE = 115.0  # safety margin well under the 120s kill

    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    n = len(times)

    # Build immediate successor lists and in-degrees (for topological order).
    succ = [[] for _ in range(n)]
    indeg = np.zeros(n, dtype=int)
    for i in range(n):
        for p in preds[i]:
            succ[p].append(i)
            indeg[i] += 1

    # Topological order (Kahn).
    from collections import deque
    q = deque([i for i in range(n) if indeg[i] == 0])
    topo = []
    indeg_w = indeg.copy()
    while q:
        u = q.popleft()
        topo.append(u)
        for v in succ[u]:
            indeg_w[v] -= 1
            if indeg_w[v] == 0:
                q.append(v)
    # If cycles exist (shouldn't), append the rest.
    if len(topo) < n:
        topo += [i for i in range(n) if i not in set(topo)]

    # Longest path (critical path length) from each node to the end,
    # computed in reverse topological order.
    lpl = np.zeros(n, dtype=float)
    for u in reversed(topo):
        best = 0.0
        for v in succ[u]:
            cand = lpl[v]
            if cand > best:
                best = cand
        lpl[u] = times[u] + best

    # Total (transitive) successor count + total downstream work,
    # computed in reverse topological order (accumulate over immediate successors,
    # then deduplicate via sets for counts, but for work we sum lpl-free downstream).
    # We compute the set of all transitive successors per node (bounded by n=30).
    trans_succ = [set() for _ in range(n)]
    for u in reversed(topo):
        s = set()
        for v in succ[u]:
            s.add(v)
            s |= trans_succ[v]
        trans_succ[u] = s

    succ_count = np.array([len(s) for s in trans_succ], dtype=float)

    # Downstream work: sum of times over all transitive successors.
    down_work = np.array([sum(times[v] for v in s) for s in trans_succ], dtype=float)

    # Smooth surrogate for downstream criticality via soft-max (log-sum-exp)
    # over successor work, which is a smooth stand-in for max and rewards
    # both large total downstream work and concentrated (branching) downstream.
    # Guard against empty successor sets.
    def softmax_logsumexp(vals, temp):
        # vals: (n, k) possibly ragged -> handle per-row.
        out = np.zeros(len(vals), dtype=float)
        for i, row in enumerate(vals):
            if len(row) == 0:
                out[i] = 0.0
                continue
            row = np.asarray(row, dtype=float)
            m = np.max(row)
            out[i] = m + temp * np.log(np.sum(np.exp((row - m) / temp)))
        return out

    # Per-node vector of transitive successor times for soft-max.
    succ_time_rows = [times[list(s)] for s in trans_succ]
    temp = 3.0
    soft_down = softmax_logsumexp(succ_time_rows, temp)

    # Slack: cycle_time - lpl (tasks with small slack are critical).
    ct = float(instance.get("cycle_time", np.max(times) * 2))
    slack = ct - lpl
    slack = np.where(slack > 0, slack, 1e-6)

    # Normalize each component to [0,1]-ish for a stable blend.
    def norm(x):
        x = np.asarray(x, dtype=float)
        lo, hi = np.min(x), np.max(x)
        if hi - lo < 1e-12:
            return np.zeros_like(x)
        return (x - lo) / (hi - lo)

    n_time = norm(times)
    n_lpl = norm(lpl)
    n_succ = norm(succ_count)
    n_down = norm(down_work)
    n_soft = norm(soft_down)
    n_slack_inv = norm(1.0 / slack)  # small slack -> high

    # Blend: emphasize long tasks, critical path, and downstream branching.
    score = (
        1.00 * n_time
        + 0.90 * n_lpl
        + 0.70 * n_down
        + 0.60 * n_soft
        + 0.50 * n_succ
        + 0.40 * n_slack_inv
    )

    if time.time() > DEADLINE:
        # Fallback to a simple, guaranteed-fast score.
        return times.copy()

    return score
# EVOLVE-BLOCK-END
