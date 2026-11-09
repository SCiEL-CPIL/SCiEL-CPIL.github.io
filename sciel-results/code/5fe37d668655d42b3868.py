"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    n = len(times)
    ct = instance["cycle_time"]

    # Compute in-degrees
    in_deg = np.zeros(n, dtype=int)
    for j in range(n):
        for p in preds[j]:
            in_deg[p] += 1

    # Topological sort
    from collections import deque
    q = deque([i for i in range(n) if in_deg[i] == 0])
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for v in range(n):
            if u in preds[v]:
                in_deg[v] -= 1
                if in_deg[v] == 0:
                    q.append(v)

    # Successors
    succ = [[] for _ in range(n)]
    for j in range(n):
        for p in preds[j]:
            succ[p].append(j)

    # Compute position in topological order
    pos = [0] * n
    for i, node in enumerate(topo):
        pos[node] = i

    # Compute total successors (including self) via reverse topo
    total_succ = [1] * n
    for node in reversed(topo):
        for s in succ[node]:
            total_succ[node] += total_succ[s]

    # Compute earliest start time (EST) using forward pass
    est = np.zeros(n, dtype=float)
    for node in topo:
        for p in preds[node]:
            est[node] = max(est[node], est[p] + times[p])

    # Compute latest start time (LST) using backward pass
    lst = np.zeros(n, dtype=float)
    for node in reversed(topo):
        if not succ[node]:
            lst[node] = ct - times[node]
        else:
            lst[node] = min(lst[s] for s in succ[node]) - times[node]
        lst[node] = max(lst[node], 0.0)

    # Compute slack = LST - EST
    slack = lst - est

    # Compute out-degree
    out_deg = np.array([len(succ[i]) for i in range(n)], dtype=float)

    # Combine priorities:
    # We want tasks that are:
    # 1. Long (high time)
    # 2. Have many successors (high out_deg or total_succ)
    # 3. Have low slack (tight)
    # 4. Are early in topological order (low pos)

    # Normalize each component
    t_max = times.max() if times.max() > 0 else 1.0
    t_min = times.min() if times.min() < 1.0 else 0.0
    t_range = t_max - t_min if t_range > 0 else 1.0
    norm_time = (times - t_min) / t_range

    ts_max = max(total_succ) if max(total_succ) > 0 else 1
    ts_min = min(total_succ) if min(total_succ) < 1 else 0
    ts_range = ts_max - ts_min if ts_range > 0 else 1.0
    norm_ts = (np.array(total_succ, dtype=float) - ts_min) / ts_range

    sl_max = slack.max() if slack.max() > 0 else 1.0
    sl_min = slack.min() if slack.min() < 1.0 else 0.0
    sl_range = sl_max - sl_min if sl_range > 0 else 1.0
    norm_slack = (sl_max - slack) / sl_range  # invert: low slack = high priority

    pos_max = max(pos) if max(pos) > 0 else 1
    pos_min = min(pos) if min(pos) < 1 else 0
    pos_range = pos_max - pos_min if pos_range > 0 else 1.0
    norm_pos = (pos_max - np.array(pos, dtype=float)) / pos_range  # invert: early = high priority

    # Weighted combination
    w_time = 0.30
    w_ts = 0.25
    w_slack = 0.25
    w_pos = 0.20

    scores = (w_time * norm_time +
              w_ts * norm_ts +
              w_slack * norm_slack +
              w_pos * norm_pos)

    return scores
# EVOLVE-BLOCK-END
