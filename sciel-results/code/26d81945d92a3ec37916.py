"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict, deque


def priority(instance):
    """
    Compute a dynamic resource-weighted criticality score for each activity.
    
    The approach:
    1. Extract instance data (durations, demands, capacities, predecessors).
    2. Compute static CPM-based criticality (backward pass) to identify activities on the critical path.
    3. Simulate a greedy serial schedule (using a simple priority like longest duration) to identify
       which activities actually cause resource contention/bottlenecks.
    4. Combine static criticality with empirical resource bottleneck detection and demand-to-capacity ratios.
    
    Returns a numpy array of scores, one per activity. Higher score = higher priority.
    """
    # Extract instance data
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    # Demands: list of lists, each activity has demands for each resource
    demands = instance["demands"]  # shape (n, num_resources)
    num_resources = len(instance["capacities"]) if "capacities" in instance else (len(demands[0]) if n > 0 else 0)
    capacities = np.asarray(instance["capacities"], dtype=float)
    
    # Predecessors: list of lists, each activity has a list of predecessor indices
    predecessors = instance["predecessors"]
    
    if n == 0:
        return np.array([])
    
    # Build adjacency lists for successors
    successors = defaultdict(list)
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)
    
    # --- Step 1: Compute static CPM criticality ---
    # Forward pass to compute earliest start times
    earliest_start = np.zeros(n)
    in_degree = np.array([len(preds) for preds in predecessors])
    topo_order = []
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # If not all nodes are in topo_order, there's a cycle; handle gracefully
    if len(topo_order) < n:
        # Fallback: just use durations
        return durations.copy()
    
    # Forward pass
    for u in topo_order:
        for v in successors[u]:
            earliest_start[v] = max(earliest_start[v], earliest_start[u] + durations[u])
    
    # Project makespan estimate
    makespan_estimate = max(earliest_start[i] + durations[i] for i in range(n))
    
    # Backward pass to compute latest start times
    latest_finish = np.full(n, makespan_estimate)
    for u in reversed(topo_order):
        for v in successors[u]:
            latest_finish[u] = min(latest_finish[u], latest_finish[v])
    
    latest_start = latest_finish - durations
    
    # Total float
    total_float = latest_start - earliest_start
    
    # Static criticality: normalize by (1 - float/makespan) or similar
    # Activities with low float are more critical
    if makespan_estimate > 0:
        static_criticality = 1.0 - total_float / makespan_estimate
    else:
        static_criticality = np.ones(n)
    
    # --- Step 2: Simulate a greedy serial schedule to identify resource bottlenecks ---
    # Use a simple greedy: highest duration first among eligible activities
    # Track how often each activity is "delayed" due to resource contention
    
    # Resource usage over time
    # We'll simulate and track for each activity whether it was delayed due to resources
    
    # Simple greedy simulation
    eligible = [i for i in range(n) if len(predecessors[i]) == 0]
    scheduled = set()
    time = 0.0
    resource_usage = np.zeros(num_resources)  # current resource usage
    
    # Track delay due to resources for each activity
    resource_delay = np.zeros(n)
    
    # For efficiency, we'll track events
    # But for simplicity with 30 activities, we can do a step-by-step simulation
    
    # Activity states: not_started, in_progress, completed
    state = ['not_started'] * n
    start_time = np.full(n, np.inf)
    end_time = np.full(n, np.inf)
    
    # Event-based simulation
    # At each event time, check if any activity can start
    
    # We need to track when activities finish
    # Let's do a simpler approach: iterate through time
    
    # Since we only have 30 activities, we can do this efficiently
    # Track which activities are ready (all predecessors done) but not yet started
    # and which are in progress
    
    remaining_preds = {i: set(predecessors[i]) for i in range(n)}
    in_progress = []  # list of (end_time, activity_index)
    
    # Priority queue for eligible activities (we'll use a simple list and sort)
    # For the simulation, we'll use longest duration as priority
    
    while len(scheduled) < n:
        # Check if any in-progress activities have finished
        still_in_progress = []
        for end_t, act_idx in in_progress:
            if end_t <= time:
                scheduled.add(act_idx)
                # Free resources
                for r in range(num_resources):
                    resource_usage[r] -= demands[act_idx][r]
                # Check if any activities are now eligible
                for v in successors[act_idx]:
                    remaining_preds[v].discard(act_idx)
            else:
                still_in_progress.append((end_t, act_idx))
        
        in_progress = still_in_progress
        
        # Find eligible activities: all predecessors in scheduled, not yet started
        eligible_acts = []
        for i in range(n):
            if i not in scheduled and state[i] == 'not_started' and len(remaining_preds[i]) == 0:
                eligible_acts.append(i)
        
        # If no eligible activities but some are in progress, advance time to next completion
        if not eligible_acts:
            if in_progress:
                next_time = min(end_t for end_t, _ in in_progress)
                time = next_time
                continue
            else:
                # No eligible and nothing in progress - shouldn't happen if no cycles
                break
        
        # Try to start eligible activities in order of priority (longest duration first)
        eligible_acts.sort(key=lambda x: durations[x], reverse=True)
        
        for act in eligible_acts:
            if act in scheduled:
                continue
            # Check if resources are available
            can_start = True
            for r in range(num_resources):
                if resource_usage[r] + demands[act][r] > capacities[r] + 1e-9:
                    can_start = False
                    break
            
            if can_start:
                # Start the activity
                state[act] = 'in_progress'
                start_time[act] = time
                end_t = time + durations[act]
                for r in range(num_resources):
                    resource_usage[r] += demands[act][r]
                in_progress.append((end_t, act))
            else:
                # Activity is delayed due to resource contention
                # We'll count this as a resource bottleneck signal
                resource_delay[act] += 1
    
    # --- Step 3: Compute demand-to-capacity ratio for each activity ---
    # For each activity, compute the max ratio of demand to capacity across resources
    demand_to_capacity = np.zeros(n)
    for i in range(n):
        if num_resources > 0 and len(capacities) > 0:
            ratios = []
            for r in range(num_resources):
                if capacities[r] > 0:
                    ratios.append(demands[i][r] / capacities[r])
            if ratios:
                demand_to_capacity[i] = max(ratios)
    
    # --- Step 4: Combine scores ---
    # Normalize each component to [0, 1]
    def normalize(arr):
        if len(arr) == 0:
            return arr
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val - min_val < 1e-9:
            return np.ones_like(arr)
        return (arr - min_val) / (max_val - min_val)
    
    static_crit_norm = normalize(static_criticality)
    resource_delay_norm = normalize(resource_delay)
    demand_cap_norm = normalize(demand_to_capacity)
    
    # Combine: weight static criticality heavily, add resource bottleneck and demand-to-capacity
    # Higher score = higher priority
    score = (0.5 * static_crit_norm + 
             0.3 * resource_delay_norm + 
             0.2 * demand_cap_norm)
    
    # Add a small bonus for long duration activities on the critical path
    duration_norm = normalize(durations)
    score += 0.1 * duration_norm * static_crit_norm
    
    return score
# EVOLVE-BLOCK-END
