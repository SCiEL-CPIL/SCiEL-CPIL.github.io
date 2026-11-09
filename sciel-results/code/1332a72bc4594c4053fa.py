"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(instance):
    """
    Compute priority scores by simulating a greedy serial schedule-generation scheme.
    
    During the simulation, at each time step, eligible activities are ranked by a
    composite score that combines:
    1. Critical path length (CPL) - how much work remains after this activity
    2. Current resource slack fit - how well the activity fits into remaining resources
    
    Activities that fit comfortably into the current resource envelope are favored,
    while those that would cause immediate bottlenecks are penalized.
    
    The returned scores reflect the order in which activities were scheduled during
    this simulation, with earlier-scheduled activities getting higher priority scores.
    """
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["durations"], dtype=float) if "demands" not in instance else np.asarray(instance["demands"], dtype=float)
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
    # CPL[i] = duration[i] + max(CPL[j] for j in successors[i])
    # Use topological order (reverse)
    in_degree = [0] * n
    for i in range(n):
        preds = predecessors[i] if i < len(predecessors) else []
        in_degree[i] = len(preds)
    
    # Topological sort
    from collections import deque
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
        # Add remaining nodes
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
    resource_usage = np.zeros(n_res, dtype=float)  # current resource usage
    current_time = 0.0
    schedule_order = []
    
    # Track which activities are currently running
    active = []  # list of (activity_id, finish_time)
    
    while len(schedule_order) < n:
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
            # Compute resource tightness: how much of the available slack this activity would consume
            tightness = 0.0
            for r in range(n_res):
                if n_res > 0 and capacities[r] > 0:
                    available_slack = capacities[r] - resource_usage[r]
                    if available_slack <= 0:
                        # This resource is already fully used
                        tightness += 10.0  # heavy penalty
                    else:
                        # Fraction of slack that would be consumed
                        tightness += demands[i, r] / available_slack
                else:
                    tightness += 0.0
            
            # Composite score: higher CPL is better, lower tightness is better
            # Weight CPL more heavily, but penalize high tightness
            score = cpl[i] * 5.0 - tightness * 2.0
            
            if score > best_score:
                best_score = score
                best_idx = i
        
        if best_idx == -1:
            # Fallback: pick the first eligible
            best_idx = eligible[0]
        
        # Schedule the best activity
        scheduled[best_idx] = True
        start_times[best_idx] = current_time
        finish_times[best_idx] = current_time + durations[best_idx]
        resource_usage += demands[best_idx]
        active.append((best_idx, finish_times[best_idx]))
        schedule_order.append(best_idx)
    
    # Assign priority scores based on schedule order
    # Earlier in the schedule = higher priority
    scores = np.zeros(n, dtype=float)
    for rank, act_id in enumerate(schedule_order):
        # Higher score for earlier positions
        scores[act_id] = float(n - rank)
    
    return scores
# EVOLVE-BLOCK-END
