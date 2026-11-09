import numpy as np
import time

def priority(instance):
    t_start = time.time()
    DEADLINE = 120.0
    SAFETY_MARGIN = 5.0

    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)  # (n, n_res)
    capacities = np.asarray(instance["capacities"], dtype=float)  # (n_res,)
    predecessors = instance["predecessors"]  # list of lists
    
    n = len(durations)
    n_res = len(capacities)
    
    # Build successor lists
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
    
    # Forward pass: ES, EF
    es = np.zeros(n)
    ef = np.zeros(n)
    for i in range(n):
        for p in predecessors[i]:
            es[i] = max(es[i], ef[p])
        ef[i] = es[i] + durations[i]
    
    project_end = np.max(ef)
    
    # Backward pass: LS, LF
    ls = np.zeros(n)
    lf = np.zeros(n)
    for i in range(n-1, -1, -1):
        if not successors[i]:
            lf[i] = project_end
        else:
            lf[i] = min(ls[s] for s in successors[i])
        ls[i] = lf[i] - durations[i]
    
    # Total float
    total_float = ls - es
    
    # Tail: distance from EF to project end
    tail = project_end - ef
    
    # Resource demand intensity
    if n_res > 0:
        demand_ratios = demands / np.maximum(capacities, 1e-9)
        max_demand_ratio = np.max(demand_ratios, axis=1)
        sum_demand_ratio = np.sum(demand_ratios, axis=1)
    else:
        max_demand_ratio = np.zeros(n)
        sum_demand_ratio = np.zeros(n)
    
    # Normalize features
    max_float = np.max(total_float) if np.max(total_float) > 0 else 1.0
    max_tail = np.max(tail) if np.max(tail) > 0 else 1.0
    max_dur = np.max(durations) if np.max(durations) > 0 else 1.0
    max_ratio = np.max(max_demand_ratio) if np.max(max_demand_ratio) > 0 else 1.0
    max_sum_ratio = np.max(sum_demand_ratio) if np.max(sum_demand_ratio) > 0 else 1.0
    
    # Check deadline
    if time.time() - t_start > DEADLINE - SAFETY_MARGIN:
        return (max_float - total_float) / max_float
    
    # Dynamic Look-Ahead Simulation:
    # Run a serial schedule simulation using a baseline rule to estimate
    # each activity's completion time under resource constraints.
    # Use the estimated completion times (inverted) as priority scores.
    
    # Baseline rule for simulation: prioritize by (tail, duration, resource demand)
    # This approximates what a good scheduler would do
    
    # Build a priority ordering for the simulation
    # We'll use a simple rule: higher tail + higher duration + higher resource demand = higher priority
    sim_priority = (
        0.5 * tail / max_tail +
        0.3 * durations / max_dur +
        0.2 * sum_demand_ratio / max_sum_ratio
    )
    
    # Serial schedule simulation
    # Track resource usage over time
    # We'll simulate time in integer steps (durations are likely integers or we can round)
    # For efficiency, we'll track resource usage as a dictionary: time -> resource usage
    
    # Start time for each activity
    start_times = np.zeros(n)
    
    # Track which activities are completed
    completed = np.zeros(n, dtype=bool)
    
    # Track resource usage: for each time step, how much of each resource is used
    # Since n=30 and durations are likely small, we can use a list of lists
    # Or better: track active activities and their end times
    
    # Use a simpler approach: simulate event by event
    # At each time step, check which activities can start
    
    # We need to track:
    # 1. Which activities are not yet started
    # 2. Which activities are in progress (and when they end)
    # 3. Current resource usage
    
    # For efficiency with n=30, we'll do a simple loop
    
    # Sort activities by simulation priority (descending)
    # But we need to respect precedence constraints
    
    # Let's do a proper serial schedule generation:
    # At each step, find eligible activities (all predecessors completed)
    # Among eligible, pick the one with highest sim_priority
    # Start it if resources allow
    
    # Track state
    remaining = list(range(n))
    in_progress = {}  # activity_id -> end_time
    resource_usage = [0.0] * n_res  # current resource usage
    
    # We'll simulate time step by step
    # But to be efficient, we'll jump to the next event time
    
    current_time = 0.0
    scheduled_count = 0
    
    while scheduled_count < n:
        # Check if time has exceeded deadline
        if time.time() - t_start > DEADLINE - SAFETY_MARGIN:
            # Return fallback
            break
        
        # Find eligible activities: all predecessors completed
        eligible = []
        for i in remaining:
            pred_done = True
            for p in predecessors[i]:
                if not completed[p]:
                    pred_done = False
                    break
            if pred_done:
                eligible.append(i)
        
        if not eligible:
            # No eligible activities, jump to next completion time
            if in_progress:
                next_end = min(in_progress.values())
                # Complete all activities that end at next_end
                to_complete = [i for i, end in in_progress.items() if end <= next_end]
                for i in to_complete:
                    completed[i] = True
                    for r in range(n_res):
                        resource_usage[r] -= demands[i, r]
                    del in_progress[i]
                current_time = next_end
            else:
                # Should not happen if there's a valid schedule
                break
            continue
        
        # Among eligible, find the one with highest sim_priority that can be scheduled
        best_activity = -1
        best_priority = -1
        for i in eligible:
            # Check if resources are available
            can_schedule = True
            for r in range(n_res):
                if resource_usage[r] + demands[i, r] > capacities[r] + 1e-9:
                    can_schedule = False
                    break
            if can_schedule:
                if sim_priority[i] > best_priority:
                    best_priority = sim_priority[i]
                    best_activity = i
        
        if best_activity == -1:
            # No eligible activity can be scheduled due to resource constraints
            # Jump to next completion time
            if in_progress:
                next_end = min(in_progress.values())
                to_complete = [i for i, end in in_progress.items() if end <= next_end]
                for i in to_complete:
                    completed[i] = True
                    for r in range(n_res):
                        resource_usage[r] -= demands[i, r]
                    del in_progress[i]
                current_time = next_end
            else:
                break
            continue
        
        # Schedule best_activity
        start_times[best_activity] = current_time
        in_progress[best_activity] = current_time + durations[best_activity]
        for r in range(n_res):
            resource_usage[r] += demands[best_activity, r]
        remaining.remove(best_activity)
        scheduled_count += 1
    
    # If simulation completed, use estimated completion times as priority
    # Activities that finish later in the simulation should have lower priority
    # So priority = max_completion - completion_time
    
    if scheduled_count == n:
        completion_times = start_times + durations
        max_completion = np.max(completion_times)
        # Higher priority for activities that finish earlier in the simulation
        # This means they are "more critical" or "more urgent"
        priority_scores = max_completion - completion_times
        # Normalize
        max_p = np.max(priority_scores) if np.max(priority_scores) > 0 else 1.0
        priority_scores = priority_scores / max_p
    else:
        # Fallback to CPM-based priority
        w_float = 0.45
        w_tail = 0.30
        w_dur = 0.10
        w_res_max = 0.08
        w_res_sum = 0.07
        
        priority_scores = (
            w_float * (max_float - total_float) / max_float +
            w_tail * tail / max_tail +
            w_dur * durations / max_dur +
            w_res_max * max_demand_ratio / max_ratio +
            w_res_sum * sum_demand_ratio / max_sum_ratio
        )
        priority_scores = np.maximum(priority_scores, 0.0)
    
    return priority_scores
