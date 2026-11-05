"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(instance):
    """
    Score each assembly task for the greedy assembly line balancing heuristic.
    
    The evaluator fills stations with the highest-priority eligible task that fits
    the cycle time. Lower total number of stations is better.
    
    We use a priority score that combines:
    1. Positional Weight (PW): Sum of times of the task and all its successors.
       This captures the total "work" that must be completed after this task,
       making it a strong indicator of bottlenecks and criticality.
    2. Longest Path (LP): The length of the longest path from the task to the end.
       This captures the critical path constraint.
    3. Task Duration: Longer tasks are prioritized to fill stations efficiently.
    """
    start_time = time.time()
    
    times = np.asarray(instance["times"], dtype=float)
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Build successor graph from predecessors
    pred = instance["predecessors"]
    succ = [[] for _ in range(n)]
    for i in range(n):
        for p in pred[i]:
            succ[p].append(i)
    
    # Topological sort using Kahn's algorithm
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for p in pred[i]:
            in_degree[i] += 1
    
    queue = [i for i in range(n) if in_degree[i] == 0]
    top_order = []
    while queue:
        # Sort by index for determinism
        queue.sort()
        u = queue.pop(0)
        top_order.append(u)
        for v in succ[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # Compute Positional Weight (PW) and Longest Path (LP)
    # PW[i] = times[i] + sum(PW[j] for j in succ[i])
    # LP[i] = times[i] + max(LP[j] for j in succ[i])
    # WSR[i] = sum(PW[j] * times[j] for j in succ[i]) - weighted successor rank
    pw = np.zeros(n, dtype=float)
    lp = np.zeros(n, dtype=float)
    wsr = np.zeros(n, dtype=float)  # weighted successor rank
    
    for u in reversed(top_order):
        pw_u = times[u]
        lp_u = times[u]
        wsr_u = 0.0
        if succ[u]:
            succ_pw = pw[succ[u]]
            succ_lp = lp[succ[u]]
            succ_times = times[succ[u]]
            pw_u += np.sum(succ_pw)
            lp_u += np.max(succ_lp)
            # Weighted successor rank: sum of (successor PW * successor time)
            # This captures both the workload and the criticality of successors
            wsr_u = np.sum(succ_pw * succ_times)
        pw[u] = pw_u
        lp[u] = lp_u
        wsr[u] = wsr_u

    # Deadline guard
    if time.time() - start_time > 100:
        return times

    # Normalize components
    def normalize(x):
        mn = np.min(x)
        mx = np.max(x)
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)
    
    norm_pw = normalize(pw)
    norm_lp = normalize(lp)
    norm_times = normalize(times)
    
    # Weighting:
    # PW is generally a stronger predictor of ALB performance than LP because
    # it accounts for branching (parallel successors).
    # LP is good for identifying the critical path.
    # Times handles the immediate packing efficiency.
    
    # Compute rank: number of immediate successors (more successors = more critical)
    rank = np.array([len(succ[i]) for i in range(n)], dtype=float)
    norm_rank = normalize(rank)
    
    # Normalize weighted successor rank
    norm_wsr = normalize(wsr)
    
    # Compute depth: number of predecessors (tasks with more predecessors are more constrained)
    pred_count = np.array([len(pred[i]) for i in range(n)], dtype=float)
    norm_depth = normalize(pred_count)
    
    # Weighting strategy:
    # PW is the strongest predictor - it captures total downstream workload
    # LP identifies critical path bottlenecks
    # WSR (weighted successor rank) captures branching complexity with workload weighting
    # NS (rank) is a good tie-breaker for parallelism
    # Times helps with immediate packing efficiency
    # Depth: tasks with fewer predecessors can be placed earlier, reducing idle time
    
    w_pw = 0.50
    w_lp = 0.15
    w_wsr = 0.15
    w_rank = 0.10
    w_times = 0.08
    w_depth = 0.02  # Small negative weight: prefer fewer predecessors (earlier placement)
    
    score = (w_pw * norm_pw + 
             w_lp * norm_lp + 
             w_wsr * norm_wsr +
             w_rank * norm_rank +
             w_times * norm_times -
             w_depth * norm_depth)
             
    # Tie-breaker: favor longer tasks when scores are very close
    # This helps pack stations more tightly
    score = score + 1e-5 * times
    
    if time.time() - start_time > 110:
        return times
    
    return score
# EVOLVE-BLOCK-END
