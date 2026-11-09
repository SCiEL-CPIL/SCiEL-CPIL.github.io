"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(instance):
    """
    Dynamic Look-Ahead Simulation heuristic for RCPSP priority.
    
    Runs a fast serial schedule simulation using a baseline rule to estimate
    each activity's completion time under resource constraints. Returns inverted
    estimated completion times as priority scores (higher score = earlier finish = higher priority).
    """
    start_time = time.time()
    DEADLINE = 110.0  # Safety margin before 120s limit
    
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
    
    # Compute in-degrees for topological ordering
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for p in predecessors[i]:
            in_degree[i] += 1
    
    # Topological sort using Kahn's algorithm
    from collections import deque
    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)
    
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # If not all nodes are in topo order, there's a cycle; fall back to simple priority
    if len(topo_order) != n:
        return durations.copy()
    
    # Forward pass: compute earliest start times (ES)
    es = np.zeros(n)
    for u in topo_order:
        for v in successors[u]:
            es[v] = max(es[v], es[u] + durations[u])
    
    # Backward pass: compute latest finish times (LF) for critical path
    # First, find the project duration (makespan without resources)
    project_duration = np.max(es + durations) if n > 0 else 0
    
    lf = np.full(n, project_duration)
    # Process in reverse topological order
    for u in reversed(topo_order):
        for v in successors[u]:
            lf[u] = min(lf[u], lf[v] - durations[u])
    
    # Total float = LF - EF = LF - (ES + duration)
    ef = es + durations
    total_float = lf - ef
    
    # Now run a fast serial schedule simulation with a baseline priority rule
    # to estimate actual completion times under resource constraints
    
    # Baseline rule: use a combination of criticality and duration
    # For the simulation, we'll use longest processing time (LPT) as baseline
    baseline_priority = durations.copy()
    
    # Run serial schedule generation
    simulated_completion = _run_serial_schedule(
        n, durations, demands, capacities, predecessors, successors,
        baseline_priority, es, lf, total_float
    )
    
    # Check time
    if time.time() - start_time > DEADLINE * 0.7:
        # Fall back to simpler heuristic
        pass
    
    # Invert completion times: earlier completion = higher priority
    # Add a small offset to avoid zero/negative issues
    max_completion = np.max(simulated_completion) if n > 0 else 1
    priority_scores = max_completion - simulated_completion
    
    # Combine with CPM-based features for robustness
    # Normalize components
    if max_completion > 0:
        priority_scores = priority_scores / max_completion
    else:
        priority_scores = np.zeros(n)
    
    # Add criticality weight (lower float = more critical)
    max_float = np.max(total_float) if n > 0 else 1
    if max_float > 0:
        criticality = 1.0 - total_float / max_float
    else:
        criticality = np.ones(n)
    
    # Add resource demand intensity
    if n_res > 0:
        demand_ratio = np.sum(demands / capacities[None, :], axis=1)
        max_demand_ratio = np.max(demand_ratio) if n > 0 else 1
        if max_demand_ratio > 0:
            demand_score = demand_ratio / max_demand_ratio
        else:
            demand_score = np.zeros(n)
    else:
        demand_score = np.zeros(n)
    
    # Combine: simulated completion (primary), criticality, demand
    final_priority = 0.5 * priority_scores + 0.3 * criticality + 0.2 * demand_score
    
    return final_priority


def _run_serial_schedule(n, durations, demands, capacities, predecessors, successors, 
                         baseline_priority, es, lf, total_float):
    """
    Run a fast serial schedule generation scheme.
    Returns estimated completion time for each activity.
    """
    # Track state
    completed = [False] * n
    completion_times = np.zeros(n)
    
    # Available resources at current time
    # We'll simulate time step by step, but for speed, we'll jump to event times
    
    # Initially, eligible activities are those with no predecessors
    eligible = [i for i in range(n) if len(predecessors[i]) == 0]
    
    # Current resource usage
    resource_usage = np.zeros(len(capacities))
    
    # Active activities: (completion_time, activity_index)
    import heapq
    active_heap = []
    
    # Time pointer
    current_time = 0.0
    
    scheduled_count = 0
    
    while scheduled_count < n:
        # Check if time limit would be exceeded - but this is called from priority, 
        # so we trust it's fast enough
        
        # Determine next event time: either an activity completes, or we can start new ones
        # First, process any completions at current_time
        while active_heap and active_heap[0][0] <= current_time + 1e-9:
            comp_time, act_idx = heapq.heappop(active_heap)
            # Free resources
            resource_usage -= demands[act_idx]
            # Ensure no negative due to floating point
            resource_usage = np.maximum(resource_usage, 0)
        
        # Now, find eligible activities that can be started
        # An activity is eligible if all predecessors are completed
        # And its resource demand fits within available capacity
        
        newly_eligible = []
        for i in range(n):
            if not completed[i] and len(predecessors[i]) > 0:
                # Check if all predecessors are completed
                all_done = True
                for p in predecessors[i]:
                    if not completed[p]:
                        all_done = False
                        break
                if all_done:
                    newly_eligible.append(i)
        
        # Also include activities that were initially eligible but not yet scheduled
        # We need to track which activities are "available" (all preds done)
        
        # Let's use a different approach: maintain a set of available activities
        # This is getting complex; let's use a simpler iterative approach
        
        # Actually, let's restart with a cleaner approach
        break
    
    # Simpler approach: iterative scheduling with time jumps
    completed = [False] * n
    completion_times = np.zeros(n)
    resource_usage = np.zeros(len(capacities))
    
    # Track which activities have all predecessors completed
    pred_count_remaining = np.array([len(predecessors[i]) for i in range(n)], dtype=int)
    
    current_time = 0.0
    scheduled = 0
    
    # Use a priority queue for scheduling
    import heapq
    
    # Eligible set: activities with all predecessors done
    # We'll recompute eligibility each iteration for simplicity (n=30, so it's fine)
    
    while scheduled < n:
        # Find all eligible activities (not completed, all preds completed)
        eligible = []
        for i in range(n):
            if not completed[i] and pred_count_remaining[i] == 0:
                eligible.append(i)
        
        if not eligible:
            # No eligible activities; jump to next completion time
            if scheduled == 0:
                # Should not happen if graph is valid
                break
            
            # Find the next completion time
            min_comp_time = np.inf
            for i in range(n):
                if not completed[i]:
                    # This activity is still in progress or not started
                    # Actually, we need to track active activities
                    pass
            
            # We need to track active activities properly
            # Let's use a different state representation
            break
        
        # Sort eligible by baseline priority (highest first)
        eligible.sort(key=lambda x: -baseline_priority[x])
        
        # Try to schedule as many as possible, in priority order
        newly_started = []
        for i in eligible:
            if time.time() - start_time > 105:  # Safety
                break
            # Check if resources are available
            if np.all(resource_usage + demands[i] <= capacities + 1e-9):
                # Start this activity
                resource_usage += demands[i]
                start_time_act = current_time
                comp_time_act = current_time + durations[i]
                completion_times[i] = comp_time_act
                newly_started.append(i)
        
        # Mark newly started as completed (in terms of being scheduled)
        # Actually, they're not completed yet, just started
        # We need to track active activities
        
        # This approach is getting messy. Let me use a cleaner simulation.
        break
    
    # Clean simulation approach
    return _simulate_clean(n, durations, demands, capacities, predecessors, 
                           successors, baseline_priority)


def _simulate_clean(n, durations, demands, capacities, predecessors, successors, 
                    baseline_priority):
    """Clean serial schedule simulation."""
    import heapq
    
    completed = [False] * n
    completion_times = np.zeros(n)
    resource_usage = np.zeros(len(capacities))
    pred_remaining = np.array([len(predecessors[i]) for i in range(n)], dtype=int)
    
    # Active activities: heap of (completion_time, activity_index)
    active_heap = []
    
    current_time = 0.0
    scheduled_count = 0
    
    while scheduled_count < n:
        # Process completions at or before current_time
        while active_heap and active_heap[0][0] <= current_time + 1e-9:
            comp_time, act_idx = heapq.heappop(active_heap)
            completed[act_idx] = True
            completion_times[act_idx] = comp_time
            resource_usage -= demands[act_idx]
            resource_usage = np.maximum(resource_usage, 0)
            # Decrement pred count for successors
            for v in successors[act_idx]:
                pred_remaining[v] -= 1
        
        # Find eligible activities
        eligible = []
        for i in range(n):
            if not completed[i] and pred_remaining[i] == 0:
                # Check if not already active
                # We need to know if i is in active_heap
                # For simplicity, let's check if it's been started
                # Actually, we need a "started" flag
                eligible.append(i)
        
        # Remove activities that are already active from eligible
        active_set = set(act_idx for _, act_idx in active_heap)
        eligible = [i for i in eligible if i not in active_set]
        
        # Sort by priority
        eligible.sort(key=lambda x: -baseline_priority[x])
        
        # Schedule as many as possible
        started_this_round = 0
        for i in eligible:
            if np.all(resource_usage + demands[i] <= capacities + 1e-9):
                # Start activity i
                resource_usage += demands[i]
                comp_time = current_time + durations[i]
                heapq.heappush(active_heap, (comp_time, i))
                started_this_round += 1
        
        if started_this_round > 0:
            scheduled_count += started_this_round
        
        # Determine next time to advance to
        if active_heap:
            next_time = active_heap[0][0]
        else:
            # No active activities; if there are still unscheduled, something is wrong
            if scheduled_count < n:
                # Check if there are eligible activities that can't be started due to resources
                # This shouldn't happen in a valid instance
                break
            else:
                break
        
        if next_time > current_time:
            current_time = next_time
        elif started_this_round == 0 and active_heap:
            # Stuck: no eligible can start, but activities are active
            # Advance to next completion
            current_time = active_heap[0][0]
    
    return completion_times
# EVOLVE-BLOCK-END
