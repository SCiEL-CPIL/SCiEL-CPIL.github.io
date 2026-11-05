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
    pw = np.zeros(n, dtype=float)
    lp = np.zeros(n, dtype=float)
    
    for u in reversed(top_order):
        pw_u = times[u]
        lp_u = times[u]
        if succ[u]:
            pw_u += np.sum(pw[succ[u]])
            lp_u += np.max(lp[succ[u]])
        pw[u] = pw_u
        lp[u] = lp_u

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
    
    w_pw = 0.5
    w_lp = 0.3
    w_times = 0.2
    
    score = (w_pw * norm_pw + 
             w_lp * norm_lp + 
             w_times * norm_times)
             
    # Tie-breaker: slightly favor longer tasks if scores are very close
    # This helps in packing when PW/LP are similar
    score = score + 1e-6 * times
    
    if time.time() - start_time > 120:
        return times
    
    return score
# EVOLVE-BLOCK-END
