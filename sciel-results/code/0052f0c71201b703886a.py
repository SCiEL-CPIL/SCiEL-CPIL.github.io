"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 140.0  # safety margin before 150s kill


def _compute_structural_priors(instance):
    """Compute static structural features to weight task priorities."""
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    n = len(times)
    ct = instance["cycle_time"]

    # Build successor counts and earliest-start (ES) from predecessors.
    succ_count = np.zeros(n, dtype=int)
    for i in range(n):
        for j in preds[i]:
            succ_count[j] += 1

    # Earliest start: ES[i] = max(ES[p] + times[p]) over predecessors p.
    # Use iterative relaxation (DAG, so finite).
    ES = np.zeros(n, dtype=float)
    # Topological order via Kahn
    indeg = np.array([len(preds[i]) for i in range(n)], dtype=int)
    from collections import deque
    q = deque([i for i in range(n) if indeg[i] == 0])
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for v in range(n):
            if u in preds[v]:
                indeg[v] -= 1
                if ES[u] + times[u] > ES[v]:
                    ES[v] = ES[u] + times[u]
                if indeg[v] == 0:
                    q.append(v)

    # Slack-like: how much idle time a task forces after it on its path.
    # Use succ_count as a "criticality" proxy: tasks with many successors
    # are more constraining; weight them higher.
    # Combine: base = times, boost by succ_count, and by (ct - ES - times)
    # being small (tight tasks) — but keep it simple & monotone.
    max_succ = max(1, int(succ_count.max()))
    max_time = max(1.0, float(times.max()))

    # priority = normalized_time * (1 + alpha * normalized_succ)
    # Higher successor count => higher priority (more dependent tasks wait).
    norm_time = times / max_time
    norm_succ = succ_count / max_succ
    alpha = 0.5
    prio = norm_time * (1.0 + alpha * norm_succ)

    return prio


def priority(instance):
    if time.time() > _DEADLINE - 5.0:
        # Fallback: plain LPT
        return np.asarray(instance["times"], dtype=float)
    try:
        return _compute_structural_priors(instance)
    except Exception:
        return np.asarray(instance["times"], dtype=float)
# EVOLVE-BLOCK-END
