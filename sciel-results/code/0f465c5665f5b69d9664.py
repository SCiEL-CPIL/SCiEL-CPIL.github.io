"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict, deque
import time


def _build_graph(n, predecessors):
    """Build adjacency and in-degree structures."""
    successors = defaultdict(list)
    indeg = np.zeros(n, dtype=int)
    for i in range(n):
        preds = predecessors[i] if i < len(predecessors) else []
        for p in preds:
            successors[p].append(i)
            indeg[i] += 1
    return successors, indeg


def _longest_path_to_end(n, successors, durations):
    """Compute longest path from each node to project end (reverse topological)."""
    # Topological sort
    in_deg = np.zeros(n, dtype=int)
    for u in range(n):
        for v in successors[u]:
            in_deg[v] += 1
    
    queue = deque([i for i in range(n) if in_deg[i] == 0])
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_deg[v] -= 1
            if in_deg[v] == 0:
                queue.append(v)
    
    # Reverse topological: longest path from node to end
    dist_to_end = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        if not successors[u]:
            dist_to_end[u] = durations[u]
        else:
            dist_to_end[u] = durations[u] + max(dist_to_end[v] for v in successors[u])
    
    return dist_to_end


def _longest_path_from_start(n, successors, durations):
    """Compute longest path from project start to each node (forward topological)."""
    in_deg = np.zeros(n, dtype=int)
    for u in range(n):
        for v in successors[u]:
            in_deg[v] += 1
    
    queue = deque([i for i in range(n) if in_deg[i] == 0])
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_deg[v] -= 1
            if in_deg[v] == 0:
                queue.append(v)
    
    dist_from_start = np.zeros(n, dtype=float)
    for u in topo_order:
        if not any(v == u for v in range(n) if successors[v]):
            # source node
            pass
        # Find predecessors of u
        preds = [v for v in range(n) if u in successors[v]]
        if not preds:
            dist_from_start[u] = durations[u]
        else:
            dist_from_start[u] = durations[u] + max(dist_from_start[v] for v in preds)
    
    return dist_from_start


def _resource_contention_score(n, demands, capacities, durations):
    """
    For each activity, compute a resource contention score based on
    how much it competes for scarce resources relative to total demand.
    """
    n_resources = len(capacities)
    # Total demand per resource across all activities
    total_demand = np.zeros(n_resources, dtype=float)
    for i in range(n):
        for r in range(n_resources):
            if demands[i, r] > 0:
                total_demand[r] += demands[i, r]
    
    # For each activity, max utilization ratio among its resources
    contention = np.zeros(n, dtype=float)
    for i in range(n):
        max_ratio = 0.0
        for r in range(n_resources):
            if capacities[r] > 0 and demands[i, r] > 0:
                # Ratio of this activity's demand to capacity,
                # weighted by how scarce the resource is (total demand / capacity)
                scarcity = total_demand[r] / capacities[r]
                ratio = demands[i, r] / capacities[r]
                max_ratio = max(max_ratio, ratio * (1.0 + scarcity * 0.5))
        contention[i] = max_ratio
    
    return contention


def _serial_schedule(n, durations, demands, capacities, predecessors, successors, priority_scores):
    """
    Serial schedule generation scheme.
    Returns makespan.
    """
    n_resources = len(capacities)
    
    # Track resource availability over time
    # We'll use an event-based approach
    scheduled = [False] * n
    completion_times = [0.0] * n
    start_times = [0.0] * n
    
    # Predecessor completion tracking
    pred_done = [0] * n
    for i in range(n):
        preds = predecessors[i] if i < len(predecessors) else []
        pred_done[i] = len(preds)
    
    # Resource usage over time: dict of time -> resource usage
    # We'll simulate time
    current_time = 0.0
    remaining = n
    
    # Resource usage at current time
    resource_usage = np.zeros(n_resources, dtype=float)
    
    # Events: list of (time, event_type, data)
    # event_type: 0 = start, 1 = finish
    # We need to track which activities are running
    
    running = []  # list of (activity_id, finish_time)
    
    # For efficiency, track resource usage as a dict of time -> delta
    # Actually, let's use a simpler approach: at each step, find the earliest time
    # when resources become available for the highest priority eligible activity
    
    while remaining > 0:
        # Find eligible activities: all predecessors completed
        eligible = []
        for i in range(n):
            if not scheduled[i] and pred_done[i] == 0:
                eligible.append(i)
        
        if not eligible:
            # Shouldn't happen in a valid DAG, but break to avoid infinite loop
            break
        
        # Sort eligible by priority (higher first)
        eligible.sort(key=lambda x: priority_scores[x], reverse=True)
        
        # Try to schedule the highest priority eligible activity
        # Find the earliest time we can start it given resource constraints
        # We need to check resource availability at current_time and beyond
        
        # For simplicity in serial SGS: at current_time, check if resources are available
        # If not, advance time to the next event where resources free up
        
        # Get resource availability at current_time
        # We track resource usage via the running activities
        # At current_time, some activities may be finishing
        
        # Remove activities that finish at or before current_time
        still_running = []
        for act, ft in running:
            if ft <= current_time:
                for r in range(n_resources):
                    resource_usage[r] -= demands[act, r]
            else:
                still_running.append((act, ft))
        running = still_running
        
        # Now try to schedule eligible activities
        scheduled_any = False
        for i in eligible:
            # Check if resources are available at current_time
            can_start = True
            for r in range(n_resources):
                if resource_usage[r] + demands[i, r] > capacities[r] + 1e-9:
                    can_start = False
                    break
            
            if can_start:
                # Schedule activity i at current_time
                scheduled[i] = True
                start_times[i] = current_time
                finish = current_time + durations[i]
                completion_times[i] = finish
                running.append((i, finish))
                
                for r in range(n_resources):
                    resource_usage[r] += demands[i, r]
                
                # Update pred_done for successors
                for v in successors[i]:
                    pred_done[v] -= 1
                
                remaining -= 1
                scheduled_any = True
                
                # Continue trying to schedule more at the same time
                # (greedy: schedule as many as possible at this time)
                # Re-check eligible (some may have become eligible)
                # But in serial SGS, typically one at a time. Let's do one at a time
                # for simplicity, then break
                break
        
        if not scheduled_any:
            # Need to advance time to next event
            if running:
                next_time = min(ft for _, ft in running)
                current_time = next_time
            else:
                # No running activities but can't schedule? Shouldn't happen
                current_time += 1.0
        else:
            # We scheduled one activity; loop to try more at same time
            # But the 'break' above means we exit the inner loop
            # The outer while loop will re-evaluate
            pass
    
    makespan = max(completion_times) if remaining == 0 else float('inf')
    return makespan


def priority(instance):
    """
    Score each activity for the serial schedule generation scheme.
    Higher score = higher priority (scheduled earlier).
    
    Strategy: Combine critical path length (longest path to end) with
    resource contention. Activities on the critical path that also demand
    scarce resources get the highest priority.
    """
    start_time = time.time()
    deadline = start_time + 120.0  # 120s limit, keep margin
    
    n = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    predecessors = instance["predecessors"]
    
    # Build graph
    successors, indeg = _build_graph(n, predecessors)
    
    # Compute longest path to end (criticality)
    dist_to_end = _longest_path_to_end(n, successors, durations)
    
    # Compute resource contention
    contention = _resource_contention_score(n, demands, capacities, durations)
    
    # Normalize both to [0, 1]
    max_dist = np.max(dist_to_end) if np.max(dist_to_end) > 0 else 1.0
    max_contention = np.max(contention) if np.max(contention) > 0 else 1.0
    
    norm_dist = dist_to_end / max_dist
    norm_contention = contention / max_contention
    
    # Combine: weight critical path heavily, with resource contention as tiebreaker
    # Higher weight on critical path since it directly determines makespan
    alpha = 0.7  # weight for critical path
    beta = 0.3   # weight for resource contention
    
    scores = alpha * norm_dist + beta * norm_contention
    
    # Add small duration component to break ties
    max_dur = np.max(durations) if np.max(durations) > 0 else 1.0
    scores += 0.1 * (durations / max_dur)
    
    # Re-normalize
    max_score = np.max(scores) if np.max(scores) > 0 else 1.0
    scores = scores / max_score
    
    return scores
# EVOLVE-BLOCK-END
