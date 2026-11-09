"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque


def priority(instance):
    """
    Compute priority scores by simulating a greedy serial schedule-generation scheme.
    
    The simulation uses a dynamic heuristic that combines Critical Path Length (CPL)
    and a resource bottleneck penalty. The penalty is based on the ratio of the 
    activity's demand to the currently available slack for each resource.
    
    The returned scores reflect the order in which activities were scheduled in the 
    simulation, ensuring that the evaluator's scheduler replicates the simulated schedule.
    """
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float) if "demands" in instance else np.asarray(instance["durations"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    predecessors = instance["predecessors"]
    
    n = len(durations)
    n_res = len(capacities)
    
    # Build successor list and in-degree
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        preds = predecessors[i] if i < len(predecessors) else []
        in_degree[i] = len(preds)
        for p in preds:
            if p < n:
                successors[p].append(i)
    
    # Compute Critical Path Length (CPL)
    # Topological sort
    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)
            
    topo_order = []
    temp_in_degree = in_degree[:] # Copy for topo sort
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            temp_in_degree[succ] -= 1
            if temp_in_degree[succ] == 0:
                queue.append(succ)
                
    if len(topo_order) < n:
        remaining = set(range(n)) - set(topo_order)
        topo_order.extend(sorted(remaining))
        
    cpl = np.zeros(n, dtype=float)
    for i in reversed(topo_order):
        max_succ_cpl = 0.0
        for succ in successors[i]:
            if cpl[succ] > max_succ_cpl:
                max_succ_cpl = cpl[succ]
        cpl[i] = durations[i] + max_succ_cpl
        
    max_cpl = max(cpl) if max(cpl) > 0 else 1.0

    # Simulation State
    scheduled = [False] * n
    start_times = [0.0] * n
    finish_times = [0.0] * n
    resource_usage = np.zeros(n_res, dtype=float)
    current_time = 0.0
    
    # Track remaining predecessors for eligibility
    rem_preds = in_degree[:]
    
    # Eligible set: activities with rem_preds == 0
    eligible = [i for i in range(n) if rem_preds[i] == 0]
    
    active = []  # list of (activity_id, finish_time)
    schedule_order = []
    
    while len(schedule_order) < n:
        # Find feasible activities among eligible
        feasible = []
        for i in eligible:
            is_feasible = True
            for r in range(n_res):
                if resource_usage[r] + demands[i, r] > capacities[r] + 1e-9:
                    is_feasible = False
                    break
            if is_feasible:
                feasible.append(i)
                
        if not feasible:
            # Advance time to next finish
            if not active:
                # Deadlock or error, break to avoid infinite loop
                break
            min_finish = min(fin for _, fin in active)
            current_time = min_finish
            new_active = []
            for aid, fin in active:
                if fin > current_time + 1e-9:
                    new_active.append((aid, fin))
                else:
                    resource_usage -= demands[aid]
                    resource_usage = np.maximum(resource_usage, 0)
                    # Update successors
                    for succ in successors[aid]:
                        rem_preds[succ] -= 1
                        if rem_preds[succ] == 0:
                            eligible.append(succ)
            active = new_active
            continue
        
        # Score feasible activities
        best_idx = -1
        best_score = -float('inf')
        
        for i in feasible:
            # Calculate resource bottleneck penalty
            max_ratio = 0.0
            for r in range(n_res):
                available = capacities[r] - resource_usage[r]
                if available > 1e-9:
                    ratio = demands[i, r] / available
                    if ratio > max_ratio:
                        max_ratio = ratio
                else:
                    # Should be infeasible if demand > 0, but handle edge case
                    if demands[i, r] > 1e-9:
                        max_ratio = float('inf')
                        break
            
            if max_ratio == float('inf'):
                score = -1e9
            else:
                # Normalize CPL
                norm_cpl = cpl[i] / max_cpl
                # Penalty: higher ratio is worse
                # Scale penalty to be comparable to CPL
                penalty = max_ratio * max_cpl * 0.5 
                score = norm_cpl * max_cpl - penalty
            
            if score > best_score:
                best_score = score
                best_idx = i
                
        if best_idx == -1:
            best_idx = feasible[0]
            
        # Schedule best_idx
        scheduled[best_idx] = True
        start_times[best_idx] = current_time
        finish_times[best_idx] = current_time + durations[best_idx]
        resource_usage += demands[best_idx]
        active.append((best_idx, finish_times[best_idx]))
        schedule_order.append(best_idx)
        
        # Remove from eligible
        eligible.remove(best_idx)
        
        # Do NOT advance time. Try to schedule another at current_time.
        
    # Generate static scores based on schedule order
    scores = np.zeros(n, dtype=float)
    for k, idx in enumerate(schedule_order):
        # Higher score for earlier scheduled activities
        scores[idx] = -k
        
    return scores
# EVOLVE-BLOCK-END
