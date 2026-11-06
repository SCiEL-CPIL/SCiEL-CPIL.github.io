# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque


def _topo_order(n, pred_lists):
    """Return topological order and successor adjacency."""
    succ = [[] for _ in range(n)]
    indeg = [0] * n
    for i in range(n):
        for p in pred_lists[i]:
            succ[p].append(i)
            indeg[i] += 1
    q = deque([i for i in range(n) if indeg[i] == 0])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return order, succ


def _downstream_stats(n, order, succ, times):
    """Compute longest_down and total_descendant_time via reverse topo."""
    ld = [0.0] * n
    td = [0.0] * n
    for u in reversed(order):
        best_ld = 0.0
        tot = 0.0
        for v in succ[u]:
            if times[v] + ld[v] > best_ld:
                best_ld = times[v] + ld[v]
            tot += times[v] + td[v]
        ld[u] = best_ld
        td[u] = tot
    return ld, td


def _greedy_stations(n, times, pred_lists, succ, cycle_time, priority_vec):
    """
    Simulate the greedy station-filling efficiently.
    Returns number of stations.
    """
    done = [False] * n
    remaining = list(range(n))
    stations = 0
    # Pre-sort by priority for faster selection? No, we filter first.
    
    while remaining:
        stations += 1
        cap = cycle_time
        
        # Find eligible tasks
        eligible = []
        new_remaining = []
        
        for t in remaining:
            is_eligible = True
            for p in pred_lists[t]:
                if not done[p]:
                    is_eligible = False
                    break
            
            if is_eligible:
                if times[t] <= cap:
                    eligible.append(t)
                else:
                    new_remaining.append(t)
            else:
                new_remaining.append(t)
        
        if not eligible:
            return 1000

        # Sort eligible by priority (desc), then time (desc)
        eligible.sort(key=lambda t: (-priority_vec[t], -times[t], t))
        
        # Assign tasks
        for t in eligible:
            if times[t] <= cap:
                cap -= times[t]
                done[t] = True
        
        remaining = new_remaining
        
    return stations


def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    pred_lists = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    if n == 0:
        return np.array([], dtype=float)

    t0 = time.time()
    DEADLINE = 140.0  # safety margin under 150s

    order, succ = _topo_order(n, pred_lists)
    ld, td = _downstream_stats(n, order, succ, times)

    # Upstream longest path
    lu = [0.0] * n
    for u in order:
        for v in succ[u]:
            c = lu[u] + times[u]
            if c > lu[v]:
                lu[v] = c

    # Candidate priority vectors (smooth surrogates for "best packing")
    cp_len = max(lu[i] + times[i] + ld[i] for i in range(n)) + 1e-9
    max_t = max(times) + 1e-9
    max_ld = max(ld) + 1e-9 if max(ld) > 0 else 1.0
    max_td = max(td) + 1e-9 if max(td) > 0 else 1.0
    max_lu = max(lu) + 1e-9 if max(lu) > 0 else 1.0
    max_succ = max([len(succ[i]) for i in range(n)]) + 1e-9
    max_pred = max([len(pred_lists[i]) for i in range(n)]) + 1e-9

    lu_a = np.array(lu, dtype=float)
    ld_a = np.array(ld, dtype=float)
    td_a = np.array(td, dtype=float)
    succ_a = np.array([len(succ[i]) for i in range(n)], dtype=float)
    pred_a = np.array([len(pred_lists[i]) for i in range(n)], dtype=float)
    cp_a = (lu_a + times + ld_a) / cp_len

    # Build a small set of candidate priority vectors
    candidates = []

    # 1. Pure time
    candidates.append(times.copy())
    
    # 2. Kiltin's Rank: Successors count, then time
    # Normalize successors and time to [0,1]
    norm_succ = succ_a / max_succ
    norm_time = times / max_t
    # Increased weight on successors based on findings
    candidates.append(norm_succ * 0.7 + norm_time * 0.3)
    
    # 3. Time + downstream (original)
    candidates.append(norm_time * 0.5 + ld_a / max_ld * 0.5)
    
    # 4. Time + CP proximity
    candidates.append(norm_time * 0.4 + cp_a * 0.4 + ld_a / max_ld * 0.2)
    
    # 5. Downstream-dominant
    candidates.append(ld_a / max_ld * 0.5 + td_a / max_td * 0.3 + norm_time * 0.2)
    
    # 6. Successor-count emphasis (bottleneck tasks first)
    candidates.append(norm_succ * 0.3 + norm_time * 0.3 + cp_a * 0.2 + ld_a / max_ld * 0.2)
    
    # 7. Inverse predecessor count (independent tasks first)
    candidates.append((1.0 - pred_a / max_pred) * 0.3 + norm_time * 0.4 + cp_a * 0.3)
    
    # 8. Smooth "rank by longest_down then time" via soft ordering
    combined = ld_a / max_ld * 0.6 + norm_time * 0.4
    idx = np.argsort(-combined)
    ranks = np.zeros(n)
    for r, t in enumerate(idx):
        ranks[t] = r
    max_r = max(n - 1, 1)
    soft_rank = 1.0 - (ranks / max_r) ** 0.7  # concave soft-rank
    candidates.append(soft_rank)
    
    # 9. Hybrid: time * cp-proximity (emphasize big tasks on critical path)
    candidates.append(norm_time * cp_a * 0.5 + norm_time * 0.3 + ld_a / max_ld * 0.2)
    
    # 10. Total descendant work emphasis
    candidates.append(td_a / max_td * 0.4 + norm_time * 0.3 + cp_a * 0.3)
    
    # 11. Pure CP proximity
    candidates.append(cp_a.copy())
    
    # 12. Slack-based: Prioritize tasks with less slack (closer to critical path)
    # Slack = cp_len - (lu + time + ld)
    slack_a = cp_len - (lu_a + times + ld_a)
    min_slack = min(slack_a)
    max_slack = max(slack_a)
    range_slack = max_slack - min_slack + 1e-9
    norm_slack = (slack_a - min_slack) / range_slack
    # Lower slack is better -> higher priority
    candidates.append((1.0 - norm_slack) * 0.5 + norm_time * 0.5)
    
    # 13. Positional Rank: Earlier in topo order? Or later?
    # Tasks with no successors (sinks) are often good to place early or late?
    # Let's try: Priority = 1 - (topo_index / n). Sinks (later in order) get lower priority?
    # Actually, usually we want to process "critical" tasks.
    # Let's try: Priority based on depth (lu). Higher lu = deeper = higher priority?
    candidates.append(lu_a / max_lu * 0.5 + norm_time * 0.5)
    
    # 14. Composite: Balanced
    candidates.append(norm_time * 0.3 + ld_a / max_ld * 0.3 + norm_succ * 0.2 + cp_a * 0.2)
    
    # 15. Positional Criticality: Ratio of earliest start to cycle time
    # Tasks with high lu relative to cycle_time are constrained early
    # We want to place them early? Or does it matter?
    # Let's try: Prioritize tasks that are "late" in the schedule (high lu)
    # This might help pack the end of the line better?
    # Or prioritize "early" tasks (low lu) to keep the start tight?
    # Let's try: Priority = norm_lu * 0.4 + norm_time * 0.6
    norm_lu = lu_a / max_lu
    candidates.append(norm_lu * 0.4 + norm_time * 0.6)
    
    # 16. Successor-Time Product: Emphasize bottlenecks that are also large
    # This combines the two strongest signals: fan-out and size
    succ_time_prod = norm_succ * norm_time
    candidates.append(succ_time_prod * 0.6 + cp_a * 0.2 + ld_a / max_ld * 0.2)
    
    # 17. Strong Successor Emphasis (Inspiration 1)
    # Prioritize tasks with many transitive successors to unlock downstream tasks early
    candidates.append(norm_succ * 0.8 + norm_time * 0.2)
    
    # 18. Successor-Downstream Hybrid
    # Combine fan-out with downstream path length
    candidates.append(norm_succ * 0.5 + ld_a / max_ld * 0.3 + norm_time * 0.2)
    
    # 19. "Bottleneck" Score: High successors AND high time
    # Non-linear interaction: if both are high, priority is very high
    candidates.append((norm_succ * norm_time) * 0.7 + cp_a * 0.3)
    
    # 20. Critical Path + Successors
    # Tasks on the critical path that also have many successors
    candidates.append(cp_a * 0.5 + norm_succ * 0.3 + norm_time * 0.2)
    
    # 21. Inverse Predecessors + Successors
    # Independent tasks with high fan-out
    candidates.append((1.0 - pred_a / max_pred) * 0.4 + norm_succ * 0.4 + norm_time * 0.2)
    
    # 22. Smooth Rank by Successors
    succ_idx = np.argsort(-succ_a)
    succ_ranks = np.zeros(n)
    for r, t in enumerate(succ_idx):
        succ_ranks[t] = r
    soft_succ_rank = 1.0 - (succ_ranks / max_r) ** 0.5
    candidates.append(soft_succ_rank * 0.6 + norm_time * 0.4)
    
    # 23. Weighted Sum of Ranks (Successors, Time, CP)
    # Rank each feature, normalize, sum
    time_idx = np.argsort(-times)
    time_ranks = np.zeros(n)
    for r, t in enumerate(time_idx):
        time_ranks[t] = r
    cp_idx = np.argsort(-cp_a)
    cp_ranks = np.zeros(n)
    for r, t in enumerate(cp_idx):
        cp_ranks[t] = r
        
    norm_succ_rank = 1.0 - succ_ranks / max_r
    norm_time_rank = 1.0 - time_ranks / max_r
    norm_cp_rank = 1.0 - cp_ranks / max_r
    
    candidates.append(norm_succ_rank * 0.5 + norm_time_rank * 0.3 + norm_cp_rank * 0.2)

    # Evaluate each candidate via greedy simulation; pick best (fewest stations)
    best_priority = candidates[0]
    best_stations = _greedy_stations(n, times, pred_lists, succ, cycle_time, candidates[0])

    for c in candidates[1:]:
        if time.time() - t0 > DEADLINE:
            break
        s = _greedy_stations(n, times, pred_lists, succ, cycle_time, c)
        if s < best_stations:
            best_stations = s
            best_priority = c

    return best_priority
# EVOLVE-BLOCK-END
