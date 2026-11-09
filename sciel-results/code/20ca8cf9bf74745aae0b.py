"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(instance):
    """
    Score each assembly task using a tightest-fit-inspired static heuristic.
    
    Based on the bipartite matching/slack directive: approximate 'negative slack'
    by prioritizing tasks that pack tightly into the cycle time. We use:
    1. Processing time (LPT): larger tasks first to reduce station count
    2. Fit score: reward times close to cycle_time/k for k=2,3,4 (common remaining capacities)
    3. Rank component: ensure precedence-critical tasks are prioritized
    
    The static proxy for 'tightest fit' combines:
    - High processing time (reduces total stations)
    - Good fit with typical remaining capacities (minimizes leftover waste)
    - Critical path rank (respects precedence structure)
    """
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    
    n = len(times)
    
    if n == 0:
        return np.array([])
    
    # Build successor list
    succs = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            if p < n:  # guard against invalid indices
                succs[p].append(i)
    
    # Compute RPW (Ranked Positional Weight) for rank component
    in_degree = [0] * n
    for i in range(n):
        for p in preds[i]:
            if p < n:
                in_degree[i] += 1
    
    from collections import deque
    q = deque()
    for i in range(n):
        if in_degree[i] == 0:
            q.append(i)
    
    topo_order = []
    in_degree_temp = in_degree[:]
    while q:
        node = q.popleft()
        topo_order.append(node)
        for s in succs[node]:
            in_degree_temp[s] -= 1
            if in_degree_temp[s] == 0:
                q.append(s)
    
    # Fallback if cycle detected
    if len(topo_order) != n:
        # Simple LPT with fit score
        fit_scores = np.array([_fit_score(t, cycle_time) for t in times])
        return times + 0.1 * fit_scores
    
    # Compute RPW in reverse topological order
    rpW = np.zeros(n, dtype=float)
    for i in reversed(topo_order):
        rpW[i] = times[i]
        for s in succs[i]:
            rpW[i] += rpW[s]
    
    # Compute fit scores: how well each task time fits into common remaining capacities
    fit_scores = np.array([_fit_score(t, cycle_time) for t in times])
    
    # Normalize components to comparable scales
    # Priority = weighted combination:
    # - RPW (critical path importance)
    # - Processing time (LPT effect)
    # - Fit score (tightest fit approximation)
    
    # Normalize each component to [0, 1] range
    rpW_norm = (rpW - rpW.min()) / (rpW.max() - rpW.min() + 1e-10)
    time_norm = (times - times.min()) / (times.max() - times.min() + 1e-10)
    fit_norm = (fit_scores - fit_scores.min()) / (fit_scores.max() - fit_scores.min() + 1e-10)
    
    # Weighted combination: emphasize time and fit for tightest packing
    # LPT is strong for bin packing; fit score approximates slack minimization
    priority_scores = 0.5 * time_norm + 0.3 * fit_norm + 0.2 * rpW_norm
    
    # Scale up to ensure meaningful differences
    priority_scores = priority_scores * (times.max() + 1)
    
    return priority_scores


def _fit_score(time_val, cycle_time):
    """
    Compute how well a task time fits into common remaining capacities.
    Higher score = better fit (less leftover waste expected).
    
    Checks proximity to cycle_time/k for k=2,3,4,5 and also to cycle_time itself.
    A task that exactly fills a fraction of cycle time has minimal slack.
    """
    if time_val <= 0 or cycle_time <= 0:
        return 0.0
    
    # If task time exceeds cycle time, it's problematic (shouldn't happen in valid instances)
    if time_val > cycle_time:
        return -1.0
    
    # Score based on closeness to cycle_time/k for k = 2, 3, 4, 5
    # Also consider cycle_time itself (k=1)
    best_score = 0.0
    for k in range(1, 6):
        target = cycle_time / k
        # Only consider if target is reasonable (not too small)
        if target < 1.0:
            continue
        # Distance from target
        dist = abs(time_val - target)
        # Normalize distance by target to make it scale-invariant
        rel_dist = dist / (target + 1e-10)
        # Score: higher when closer to target
        # Use exponential decay: score = exp(-rel_dist * 2)
        score = np.exp(-rel_dist * 2.0)
        if score > best_score:
            best_score = score
    
    # Bonus for being close to cycle_time itself (fills entire station)
    if time_val > cycle_time * 0.8:
        best_score = max(best_score, 1.0)
    
    return best_score
# EVOLVE-BLOCK-END
