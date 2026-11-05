"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _serial_schedule(instance, priority_scores):
    """
    Run a serial schedule-generation scheme.
    Returns makespan.
    """
    n = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    demands = instance.get("demands", None)  # shape (n, R)
    capacities = instance.get("capacities", None)  # shape (R,)
    predecessors = instance.get("predecessors", None)  # list of lists
    
    if predecessors is None:
        predecessors = [[] for _ in range(n)]
    
    # Build successor list
    successors = [[] for _ in range(n)]
    in_degree = np.zeros(n, dtype=int)
    for j in range(n):
        for i in predecessors[j]:
            successors[i].append(j)
            in_degree[j] += 1
    
    # Track completion times
    completion = np.zeros(n, dtype=float)
    available = [j for j in range(n) if in_degree[j] == 0]
    
    # Resource usage over time: we'll track events
    # For simplicity, use a discrete-event approach with sorted event times
    # But for 30 activities, we can just track resource availability per time step
    
    # Actually, let's use a cleaner approach:
    # Maintain a list of (end_time, resource_index, amount) for when resources are freed
    # For each resource, track a list of (end_time, amount)
    
    R = len(capacities) if capacities is not None else 0
    resource_free = [[] for _ in range(R)]  # for each resource, list of (end_time, amount)
    current_time = 0.0
    remaining_in_degree = in_degree.copy()
    completed_count = 0
    
    while completed_count < n:
        # Find eligible activities (all predecessors completed)
        eligible = [j for j in range(n) if remaining_in_degree[j] == 0 and completion[j] == 0 and j not in [x for x in range(n) if completion[x] > 0]]
        
        # Actually, let's track which are done
        done = set(completion[completion > 0].astype(int)) if n > 0 else set()
        # Hmm, completion starts at 0, so we need a separate flag
        # Let me redo this more carefully
        
    # Let me rewrite with a cleaner state
    return _serial_schedule_clean(instance, priority_scores)


def _serial_schedule_clean(instance, priority_scores):
    n = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    predecessors = instance.get("predecessors", None)
    
    if predecessors is None:
        predecessors = [[] for _ in range(n)]
    
    if demands is None:
        demands = np.zeros((n, 0))
    if capacities is None:
        capacities = np.zeros(0)
    
    R = len(capacities)
    
    # State
    start_time = np.full(n, -1.0)
    completion_time = np.full(n, -1.0)
    remaining_preds = np.array([len(p) for p in predecessors], dtype=int)
    
    # For each resource, maintain a list of (release_time, amount_freed)
    # We'll use a simple approach: for each resource, track intervals of usage
    # Since n=30, we can afford O(n^2 * R) checks
    
    # Resource usage: for resource r, list of (start, end, amount)
    resource_intervals = [[] for _ in range(R)]
    
    def can_start(j, t):
        """Check if activity j can start at time t given resource constraints."""
        if R == 0:
            return True
        for r in range(R):
            if demands[j, r] == 0:
                continue
            # Check overlap with existing intervals
            for (s, e, amt) in resource_intervals[r]:
                if t < e and t + durations[j] > s:
                    # Overlap exists
                    if amt + demands[j, r] > capacities[r]:
                        return False
        return True
    
    def find_earliest_start(j, min_t):
        """Find the earliest time >= min_t when activity j can start."""
        if R == 0:
            return min_t
        t = min_t
        max_iterations = n * 10 + 100  # safety bound
        for _ in range(max_iterations):
            if can_start(j, t):
                return t
            # Find the earliest end time of any conflicting interval
            earliest_end = float('inf')
            for r in range(R):
                if demands[j, r] == 0:
                    continue
                for (s, e, amt) in resource_intervals[r]:
                    if t < e and t + durations[j] > s:
                        if amt + demands[j, r] > capacities[r]:
                            earliest_end = min(earliest_end, e)
            if earliest_end == float('inf'):
                return t  # Should not happen if can_start failed
            t = earliest_end
        return t
    
    # Serial SGS: repeatedly pick the highest priority eligible activity
    # Eligible = all predecessors completed
    # The start time is max(completion of preds, earliest feasible time)
    
    scheduled = [False] * n
    current_time = 0.0
    
    for _ in range(n):
        # Find eligible activities
        eligible = []
        for j in range(n):
            if scheduled[j]:
                continue
            # Check all preds are scheduled
            all_preds_done = True
            for p in predecessors[j]:
                if not scheduled[p]:
                    all_preds_done = False
                    break
            if all_preds_done:
                eligible.append(j)
        
        if not eligible:
            break  # Should not happen if precedence is acyclic
        
        # Pick highest priority
        best_j = max(eligible, key=lambda j: priority_scores[j])
        
        # Determine earliest start based on predecessors
        pred_completion = 0.0
        for p in predecessors[best_j]:
            pred_completion = max(pred_completion, completion_time[p])
        
        # Find earliest feasible start
        t_start = find_earliest_start(best_j, pred_completion)
        
        start_time[best_j] = t_start
        completion_time[best_j] = t_start + durations[best_j]
        
        # Add resource usage
        for r in range(R):
            if demands[best_j, r] > 0:
                resource_intervals[r].append((t_start, completion_time[best_j], demands[best_j, r]))
        
        scheduled[best_j] = True
        current_time = max(current_time, completion_time[best_j])
    
    makespan = max(completion_time) if n > 0 else 0.0
    return makespan


def priority(instance):
    """
    Score each activity using resource-weighted criticality.
    Combines CPM critical path with resource bottleneck detection.
    """
    start_wall = time.time()
    DEADLINE = 140.0  # 10s margin before 150s kill
    
    n = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    predecessors = instance.get("predecessors", None)
    
    if predecessors is None:
        predecessors = [[] for _ in range(n)]
    
    if demands is None:
        demands = np.zeros((n, 0))
    if capacities is None:
        capacities = np.zeros(0)
    
    R = len(capacities)
    
    # Step 1: Compute CPM critical path (backward pass for latest start times)
    # Forward pass: earliest start and completion
    es = np.zeros(n, dtype=float)
    ec = np.zeros(n, dtype=float)
    
    # Topological order
    in_degree = np.array([len(p) for p in predecessors], dtype=int)
    successors = [[] for _ in range(n)]
    for j in range(n):
        for i in predecessors[j]:
            successors[i].append(j)
    
    from collections import deque
    queue = deque([j for j in range(n) if in_degree[j] == 0])
    topo_order = []
    temp_in_degree = in_degree.copy()
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            temp_in_degree[v] -= 1
            if temp_in_degree[v] == 0:
                queue.append(v)
    
    # Forward pass
    for u in topo_order:
        for v in successors[u]:
            es[v] = max(es[v], ec[u])
        ec[u] = es[u] + durations[u]
    
    project_duration = max(ec) if n > 0 else 0.0
    
    # Backward pass: latest start and completion
    ls = np.full(n, project_duration, dtype=float)
    lc = np.full(n, project_duration, dtype=float)
    
    for u in reversed(topo_order):
        for v in successors[u]:
            ls[u] = min(ls[u], ls[v])
            lc[u] = min(lc[u], lc[v])
        # ls[u] is already set; lc[u] = ls[u] + durations[u]
    
    # Total float
    total_float = np.zeros(n, dtype=float)
    for i in range(n):
        total_float[i] = ls[i] - es[i]
    
    # Step 2: Resource demand-to-capacity ratio
    if R > 0 and np.any(capacities > 0):
        # For each activity, compute max demand/capacity ratio across resources
        # Avoid division by zero
        safe_caps = np.where(capacities > 0, capacities, 1.0)
        ratio = np.max(demands / safe_caps, axis=1)
    else:
        ratio = np.zeros(n, dtype=float)
    
    # Step 3: Simulate a greedy schedule to identify bottlenecks
    # Use a simple priority (e.g., longest duration) to get a baseline schedule
    # Then identify which activities cause resource contention
    
    # For now, use a static heuristic combining CPM criticality and resource ratio
    # Criticality: inverse of total float (activities with less float are more critical)
    # We use 1/(float + epsilon) to avoid division by zero
    
    epsilon = 1e-6
    criticality = 1.0 / (total_float + epsilon)
    
    # Normalize
    max_crit = np.max(criticality) if n > 0 else 1.0
    if max_crit > 0:
        criticality_norm = criticality / max_crit
    else:
        criticality_norm = criticality
    
    max_ratio = np.max(ratio) if n > 0 else 1.0
    if max_ratio > 0:
        ratio_norm = ratio / max_ratio
    else:
        ratio_norm = ratio
    
    # Combine: weight criticality more heavily, but boost high-resource-demand activities
    # score = alpha * criticality_norm + (1 - alpha) * ratio_norm
    # Try alpha = 0.7
    
    alpha = 0.7
    scores = alpha * criticality_norm + (1 - alpha) * ratio_norm
    
    # Add a small duration component to break ties
    max_dur = np.max(durations) if n > 0 else 1.0
    if max_dur > 0:
        dur_norm = durations / max_dur
    else:
        dur_norm = durations
    
    scores = 0.9 * scores + 0.1 * dur_norm
    
    # Check time
    elapsed = time.time() - start_wall
    if elapsed > DEADLINE:
        return scores
    
    # Optional: refine with a simulation-based bottleneck detection
    # Run a few greedy schedules with different priorities and see which activities
    # tend to be delayed due to resource contention
    
    # For now, return the static score
    return scores
# EVOLVE-BLOCK-END
