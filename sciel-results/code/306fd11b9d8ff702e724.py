"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque


def priority(instance):
    """
    Compute a dynamic resource-weighted criticality score.
    
    New Approach:
    1. Compute CPM Earliest Start (ES) and Criticality (Longest Path to End).
    2. Compute Resource Pressure (Demand/Capacity ratio).
    3. Simulate a schedule using 'Longest Duration' as priority to identify
       which activities are naturally bottlenecked by resource contention.
    4. Measure the 'Observed Delay' for each activity relative to its CPM ES.
    5. Combine these into a score: Criticality * (1 + Relative Delay) * Pressure.
    """
    durations = np.asarray(instance["durations"], dtype=float)
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    predecessors = instance.get("predecessors", None)
    
    n = len(durations)
    
    # Handle edge cases
    if demands is None or capacities is None:
        return durations.copy()
    
    demands = np.asarray(demands)
    capacities = np.asarray(capacities)
    
    if demands.ndim == 1:
        demands = demands.reshape(-1, 1)
    if capacities.ndim == 2:
        capacities = capacities.flatten()
    
    n_resources = len(capacities)
    
    # Build predecessor and successor sets
    if predecessors is None:
        pred_sets = [set() for _ in range(n)]
    else:
        pred_sets = [set() for _ in range(n)]
        for i, preds in enumerate(predecessors):
            if preds is not None:
                pred_sets[i] = set(preds)
    
    succ_sets = [set() for _ in range(n)]
    for i in range(n):
        for p in pred_sets[i]:
            succ_sets[p].add(i)
    
    # Topological Sort
    in_degree = [len(pred_sets[i]) for i in range(n)]
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    top_order = []
    while queue:
        node = queue.popleft()
        top_order.append(node)
        for s in succ_sets[node]:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)
    
    # 1. Compute CPM Earliest Start (ES) and Criticality (Longest Path to End)
    es = [0.0] * n
    for node in top_order:
        if pred_sets[node]:
            es[node] = max(es[p] + durations[p] for p in pred_sets[node])
    
    longest_to_end = [0.0] * n
    for node in reversed(top_order):
        if succ_sets[node]:
            longest_to_end[node] = max(longest_to_end[s] for s in succ_sets[node]) + durations[node]
        else:
            longest_to_end[node] = durations[node]
    
    # 2. Compute Resource Pressure
    # Use max ratio to capture the most constrained resource for that activity
    pressure = np.zeros(n)
    for i in range(n):
        max_ratio = 0.0
        for r in range(n_resources):
            if capacities[r] > 0:
                ratio = demands[i][r] / capacities[r]
                if ratio > max_ratio:
                    max_ratio = ratio
        pressure[i] = max_ratio
    
    # 3. Simulate with 'Longest Duration' priority to find bottlenecks
    # Long activities are often the ones that block resources for the longest time,
    # causing delays for others.
    sim_priority = durations.copy()
    scheduled = [False] * n
    start_times = [0.0] * n
    end_times = [0.0] * n
    remaining_preds = [len(pred_sets[i]) for i in range(n)]
    
    time = 0.0
    scheduled_count = 0
    active_activities = []
    resource_usage = [0.0] * n_resources
    
    while scheduled_count < n:
        eligible = [i for i in range(n) if not scheduled[i] and remaining_preds[i] == 0]
        
        if not eligible:
            if active_activities:
                time = min(end for end, _ in active_activities)
                new_active = []
                for end, idx in active_activities:
                    if end <= time:
                        for r in range(n_resources):
                            resource_usage[r] -= demands[idx][r]
                        for s in succ_sets[idx]:
                            remaining_preds[s] -= 1
                    else:
                        new_active.append((end, idx))
                active_activities = new_active
            else:
                break
            continue
        
        # Sort by Longest Duration (descending)
        eligible_sorted = sorted(eligible, key=lambda x: -sim_priority[x])
        
        scheduled_any = False
        for act in eligible_sorted:
            can_schedule = True
            for r in range(n_resources):
                if resource_usage[r] + demands[act][r] > capacities[r] + 1e-9:
                    can_schedule = False
                    break
            
            if can_schedule:
                start_times[act] = time
                end_times[act] = time + durations[act]
                
                scheduled[act] = True
                scheduled_count += 1
                for r in range(n_resources):
                    resource_usage[r] += demands[act][r]
                active_activities.append((end_times[act], act))
                scheduled_any = True
                break
        
        if not scheduled_any:
            if active_activities:
                time = min(end for end, _ in active_activities)
                new_active = []
                for end, idx in active_activities:
                    if end <= time:
                        for r in range(n_resources):
                            resource_usage[r] -= demands[idx][r]
                        for s in succ_sets[idx]:
                            remaining_preds[s] -= 1
                    else:
                        new_active.append((end, idx))
                active_activities = new_active
            else:
                break
    
    # 4. Measure Observed Delay relative to CPM ES
    # If an activity started much later than its CPM ES, it was delayed by resources.
    observed_delay = np.zeros(n)
    for i in range(n):
        observed_delay[i] = max(0.0, start_times[i] - es[i])
    
    # Normalize components
    max_lte = max(longest_to_end) if max(longest_to_end) > 0 else 1.0
    norm_lte = np.array([lte / max_lte for lte in longest_to_end])
    
    max_delay = max(observed_delay) if max(observed_delay) > 0 else 1.0
    # Relative delay: how much delay compared to its own duration?
    # Or just normalized delay. Let's use normalized delay but scale by criticality.
    norm_delay = np.array([d / max_delay for d in observed_delay])
    
    max_pressure = max(pressure) if max(pressure) > 0 else 1.0
    norm_pressure = pressure / max_pressure
    
    # 5. Combine Scores
    # We want to prioritize activities that are:
    # - Critical (high LTE)
    # - Bottlenecked (high delay in the duration-based sim)
    # - Resource Heavy (high pressure)
    
    # The delay in the simulation indicates that if we don't prioritize this activity
    # (or its predecessors), it gets pushed back.
    # However, the simulation used Duration priority.
    # If a high-criticality activity had high delay in the duration-priority sim,
    # it means duration-priority is bad for it, so we should prioritize it more.
    
    # Score = Criticality * (1 + DelayFactor) * PressureFactor
    # DelayFactor: How much was it delayed?
    # PressureFactor: How much resource does it take?
    
    # Let's use a weighted sum with interaction terms
    # Base: CPM Criticality
    # Boost: If it was delayed, boost it.
    # Boost: If it consumes lots of resources, boost it.
    
    w_crit = 0.5
    w_delay = 0.3
    w_press = 0.2
    
    # Interaction: Critical AND Delayed is very bad -> prioritize high
    # Critical AND High Pressure is very bad -> prioritize high
    
    scores = w_crit * norm_lte + w_delay * norm_delay + w_press * norm_pressure
    
    # Add interaction term: Criticality * Delay
    # This captures "Activities that are critical but got delayed"
    interaction = norm_lte * norm_delay
    max_inter = max(interaction) if max(interaction) > 0 else 1.0
    norm_inter = interaction / max_inter
    
    scores = scores + 0.2 * norm_inter
    
    # Normalize final scores to [0, 1] for consistency, though SGS only needs relative order
    max_score = max(scores) if max(scores) > 0 else 1.0
    scores = scores / max_score
    
    return scores
# EVOLVE-BLOCK-END
