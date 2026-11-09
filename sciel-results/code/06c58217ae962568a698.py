"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque


def priority(instance):
    """
    Compute priority scores by simulating a greedy serial schedule-generation scheme.
    
    During the simulation, at each time step, eligible activities are ranked by a
    composite score that combines:
    1. Critical path length (CPL) - how much work remains after this activity
    2. Resource fit score - how well the activity fits into remaining resources
    
    The resource fit score is based on the maximum ratio of demand to available
    capacity across all resources the activity uses. Activities that fit comfortably
    (low ratio) are favored, while those that would cause bottlenecks (high ratio)
    are penalized.
    
    The returned scores are the composite scores used during simulation, providing
    direct discrimination between activities.
    """
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float) if "demands" in instance else np.asarray(instance["durations"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    predecessors = instance["predecessors"]
    
    n = len(durations)
    n_res = len(capacities)
    
    # Build successor list for CPL computation
    successors = [[] for _ in range(n)]
    for i in range(n):
        preds = predecessors[i] if i < len(predecessors) else []
        for p in preds:
            if p < n:
                successors[p].append(i)
    
    # Compute Critical Path Length (CPL) for each activity
    in_degree = [0] * n
    for i in range(n):
        preds = predecessors[i] if i < len(predecessors) else []
        in_degree[i] = len(preds)
    
    # Topological sort
    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)
    
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)
    
    # If not all nodes are in topo_order (cycle), handle gracefully
    if len(topo_order) < n:
        remaining = set(range(n)) - set(topo_order)
        topo_order.extend(sorted(remaining))
    
    # Compute CPL in reverse topological order
    cpl = np.zeros(n, dtype=float)
    for i in reversed(topo_order):
        max_succ_cpl = 0.0
        for succ in successors[i]:
            if cpl[succ] > max_succ_cpl:
                max_succ_cpl = cpl[succ]
        cpl[i] = durations[i] + max_succ_cpl
    
    # Simulate greedy serial schedule
    scheduled = [False] * n
    start_times = [0.0] * n
    finish_times = [0.0] * n
    resource_usage = np.zeros(n_res, dtype=float)
    current_time = 0.0
    composite_scores = np.zeros(n, dtype=float)
    
    # Track which activities are currently running
    active = []  # list of (activity_id, finish_time)
    
    while len([s for s in scheduled if s]) < n:
        # Find eligible activities: all predecessors are finished
        eligible = []
        for i in range(n):
            if scheduled[i]:
                continue
            all_preds_done = True
            preds = predecessors[i] if i < len(predecessors) else []
            for p in preds:
                if not scheduled[p] or finish_times[p] > current_time:
                    all_preds_done = False
                    break
            if all_preds_done:
                eligible.append(i)
        
        if not eligible:
            # Advance time to next event (earliest finish of active activities)
            if active:
                min_finish = min(finish for _, finish in active)
                current_time = min_finish
                # Remove finished activities
                new_active = []
                for aid, fin in active:
                    if fin > current_time:
                        new_active.append((aid, fin))
                    else:
                        resource_usage -= demands[aid]
                        resource_usage = np.maximum(resource_usage, 0)
                active = new_active
            else:
                # Should not happen if there are no cycles, but handle it
                break
            continue
        
        # Rank eligible activities by composite score
        best_idx = -1
        best_score = -float('inf')
        
        for i in eligible:
            # Compute resource fit: maximum ratio of demand to available capacity
            max_ratio = 0.0
            for r in range(n_res):
                if capacities[r] > 0:
                    available = capacities[r] - resource_usage[r]
                    if available <= 0:
                        max_ratio = float('inf')
                        break
                    ratio = demands[i, r] / available
                    if ratio > max_ratio:
                        max_ratio = ratio
            
            # Composite score: normalize CPL and penalize poor resource fit
            # Use a harmonic-like combination: high CPL and low max_ratio are good
            if max_ratio == float('inf'):
                score = -1e9  # Very low priority if infeasible
            else:
                # Normalize CPL to [0, 1]
                max_cpl = max(cpl) if max(cpl) > 0 else 1.0
                norm_cpl = cpl[i] / max_cpl
                # Resource fit score: 1/(1+max_ratio) is high when fit is good
                fit_score = 1.0 / (1.0 + max_ratio)
                # Combine: weight CPL more, but resource fit matters significantly
                score = 0.7 * norm_cpl + 0.3 * fit_score
            
            if score > best_score:
                best_score = score
                best_idx = i
        
        if best_idx == -1:
            best_idx = eligible[0]
        
        # Store the composite score for this activity
        composite_scores[best_idx] = best_score
        
        # Schedule the best activity
        scheduled[best_idx] = True
        start_times[best_idx] = current_time
        finish_times[best_idx] = current_time + durations[best_idx]
        resource_usage += demands[best_idx]
        active.append((best_idx, finish_times[best_idx]))
    
    return composite_scores
# EVOLVE-BLOCK-END
