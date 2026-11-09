"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq

_DEADLINE_SECONDS = 140.0  # Safety margin before 150s kill


def _build_graph(predecessors, n):
    """Build predecessor and successor lists."""
    if isinstance(predecessors, np.ndarray):
        pred_lists = [list(predecessors[i]) for i in range(n)]
    else:
        pred_lists = [list(p) for p in predecessors]
    
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in pred_lists[i]:
            if i not in successors[p]:
                successors[p].append(i)
    
    return pred_lists, successors


def _topological_order(pred_lists, successors, n):
    """Compute topological order. Returns None if cycle detected."""
    in_degree = [len(pred_lists[i]) for i in range(n)]
    queue = [i for i in range(n) if in_degree[i] == 0]
    topo_order = []
    while queue:
        u = queue.pop(0)
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    if len(topo_order) < n:
        return None
    return topo_order


def _compute_cpm_metrics(durations, pred_lists, successors, topo_order):
    """Compute head, tail, and slack values."""
    n = len(durations)
    
    head = np.zeros(n, dtype=float)
    for i in topo_order:
        if pred_lists[i]:
            head[i] = durations[i] + max(head[p] for p in pred_lists[i])
        else:
            head[i] = durations[i]
    
    tail = np.zeros(n, dtype=float)
    for i in reversed(topo_order):
        if not successors[i]:
            tail[i] = durations[i]
        else:
            tail[i] = durations[i] + max(tail[j] for j in successors[i])
    
    makespan_ignoring_resources = max(tail) if n > 0 else 0.0
    
    slack = np.zeros(n, dtype=float)
    for i in range(n):
        slack[i] = makespan_ignoring_resources - head[i] - tail[i] + durations[i]
    
    return head, tail, slack


def _normalize(arr):
    """Normalize array to [0, 1]."""
    arr = np.asarray(arr, dtype=float)
    mn, mx = np.min(arr), np.max(arr)
    if mx - mn < 1e-10:
        return np.ones_like(arr) * 0.5
    return (arr - mn) / (mx - mn)


def _compute_priority(instance):
    """
    Compute a priority score for each activity to minimize makespan.
    
    Strategy:
    1. Compute CPM metrics: Head, Tail, Slack.
    2. Simulate a greedy serial schedule using dynamic resource-aware scoring.
    3. Derive final priority scores from the simulation order and static metrics.
    
    The dynamic scoring considers current resource slack to penalize activities
    that would cause immediate resource bottlenecks.
    """
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)  # shape: (n_activities, n_resources)
    capacities = np.asarray(instance["capacities"], dtype=float)  # shape: (n_resources,)
    predecessors = instance["predecessors"]
    
    n = len(durations)
    if n == 0:
        return np.array([])
    
    pred_lists, successors = _build_graph(predecessors, n)
    topo_order = _topological_order(pred_lists, successors, n)
    
    if topo_order is None:
        return durations.copy()
    
    head, tail, slack = _compute_cpm_metrics(durations, pred_lists, successors, topo_order)
    
    # Pre-compute static components
    n_resources = len(capacities)
    inv_caps = np.zeros(n_resources)
    for r in range(n_resources):
        if capacities[r] > 0:
            inv_caps[r] = 1.0 / capacities[r]
    
    norm_tail = _normalize(tail)
    norm_slack_inv = _normalize(-slack)
    norm_dur = _normalize(durations)
    
    # Simulate greedy serial schedule with dynamic resource-aware scoring
    # This gives us an ordering that accounts for resource constraints
    simulated_order = _simulate_greedy_schedule(
        durations, demands, capacities, pred_lists, successors, 
        norm_tail, norm_slack_inv, norm_dur
    )
    
    # If simulation failed, fall back to static scoring
    if simulated_order is None:
        # Static fallback with resource pressure
        resource_pressure = np.zeros(n, dtype=float)
        for i in range(n):
            pressure = 0.0
            for r in range(n_resources):
                if capacities[r] > 0:
                    ratio = demands[i][r] * inv_caps[r]
                    pressure += ratio * ratio
                else:
                    if demands[i][r] > 0:
                        pressure += 100.0
            resource_pressure[i] = pressure
        
        norm_pressure = _normalize(resource_pressure)
        
        w_tail = 0.40
        w_slack = 0.35
        w_pressure = 0.20
        w_dur = 0.05
        
        priority_scores = (w_tail * norm_tail + 
                           w_slack * norm_slack_inv + 
                           w_pressure * norm_pressure + 
                           w_dur * norm_dur)
        return priority_scores
    
    # Convert simulated order to priority scores
    # Activities scheduled earlier get higher priority
    # Use a smooth ranking based on position in the simulated order
    priority_scores = np.zeros(n, dtype=float)
    for rank, activity in enumerate(simulated_order):
        # Higher rank (earlier) = higher priority
        # Use exponential decay to emphasize early positions
        priority_scores[activity] = 1.0 - (rank / max(n - 1, 1))
    
    # Blend with static CPM metrics to ensure robustness
    # The simulated order already incorporates resource awareness
    # Blend to avoid over-reliance on a single simulation path
    w_sim = 0.60
    w_tail = 0.20
    w_slack = 0.15
    w_dur = 0.05
    
    priority_scores = (w_sim * priority_scores + 
                       w_tail * norm_tail + 
                       w_slack * norm_slack_inv + 
                       w_dur * norm_dur)
    
    return priority_scores


def _simulate_greedy_schedule(durations, demands, capacities, pred_lists, successors,
                              norm_tail, norm_slack_inv, norm_dur):
    """
    Simulate a greedy serial schedule using dynamic resource-aware scoring.
    Returns the order in which activities were scheduled, or None on failure.
    """
    n = len(durations)
    n_resources = len(capacities)
    
    # Track remaining predecessors for each activity
    remaining_preds = [len(pred_lists[i]) for i in range(n)]
    # Track current resource usage
    current_usage = np.zeros(n_resources, dtype=float)
    # Track end times of activities
    end_times = np.zeros(n, dtype=float)
    
    # Start time for each activity (when it becomes eligible)
    start_times = np.zeros(n, dtype=float)
    
    # Eligible activities: those with all predecessors completed
    # Use a min-heap for efficient selection
    # Each entry: (-score, activity_id)
    eligible_heap = []
    
    # Initially, activities with no predecessors are eligible
    for i in range(n):
        if remaining_preds[i] == 0:
            # Compute initial score
            score = _compute_dynamic_score(i, durations, demands, capacities,
                                          norm_tail, norm_slack_inv, norm_dur,
                                          current_usage, start_times)
            heapq.heappush(eligible_heap, (-score, i))
    
    simulated_order = []
    scheduled_count = 0
    current_time = 0.0
    
    while scheduled_count < n:
        if not eligible_heap:
            # No eligible activities but not all scheduled - should not happen if no cycles
            # Advance time to next completion
            min_end = float('inf')
            for i in range(n):
                if end_times[i] > 0 and i not in simulated_order:
                    min_end = min(min_end, end_times[i])
            if min_end == float('inf'):
                return None
            current_time = min_end
        
        # Select highest priority eligible activity
        # Note: we need to re-score as time advances and resources change
        best_score = -float('inf')
        best_activity = -1
        best_heap_entry = None
        
        while eligible_heap:
            neg_score, activity = heapq.heappop(eligible_heap)
            # Verify this activity is still eligible (should be, but check)
            if activity not in simulated_order and remaining_preds[activity] == 0:
                # Re-compute score with current state
                new_score = _compute_dynamic_score(activity, durations, demands, capacities,
                                                  norm_tail, norm_slack_inv, norm_dur,
                                                  current_usage, start_times)
                if new_score > best_score:
                    best_score = new_score
                    best_activity = activity
                    best_heap_entry = (activity, new_score)
                # Push back other eligible activities with updated scores
                # Actually, we should push them back with their new scores
                # But this is expensive. Let's just pick the best from current heap
                # and re-evaluate if needed
                # For now, we'll push back activities that weren't selected
                # This is a simplification
                # Better approach: just use the heap as-is but re-score when popping
                # Let's push back all non-selected activities
                if activity != best_activity:
                    heapq.heappush(eligible_heap, (-new_score, activity))
                break
            # If not eligible, discard
            continue
        
        if best_activity == -1:
            # Fallback: advance time
            min_end = float('inf')
            for i in range(n):
                if end_times[i] > 0 and i not in simulated_order:
                    min_end = min(min_end, end_times[i])
            if min_end == float('inf'):
                return None
            current_time = min_end
            # Re-evaluate all unscheduled activities with no remaining preds
            # Clear heap and rebuild
            eligible_heap = []
            for i in range(n):
                if i not in simulated_order and remaining_preds[i] == 0:
                    score = _compute_dynamic_score(i, durations, demands, capacities,
                                                  norm_tail, norm_slack_inv, norm_dur,
                                                  current_usage, start_times)
                    heapq.heappush(eligible_heap, (-score, i))
            continue
        
        # Schedule the best activity
        activity = best_activity
        simulated_order.append(activity)
        scheduled_count += 1
        
        # Determine start time
        # Must be after all predecessors complete
        earliest_start = current_time
        for p in pred_lists[activity]:
            earliest_start = max(earliest_start, end_times[p])
        
        # Check resource availability
        # For simplicity, assume activity starts at earliest_start
        # In a real RCPSP, we might need to wait for resources
        # But the serial SGS typically starts at the earliest feasible time
        start_time = earliest_start
        
        # Compute end time
        end_time = start_time + durations[activity]
        
        # Update resource usage (simplified: assume instantaneous allocation)
        # In reality, resources are held for the duration
        # For scoring purposes, we track peak usage
        for r in range(n_resources):
            current_usage[r] += demands[activity][r]
        
        end_times[activity] = end_time
        start_times[activity] = start_time
        
        # Update remaining preds for successors
        for s in successors[activity]:
            remaining_preds[s] -= 1
            if remaining_preds[s] == 0:
                score = _compute_dynamic_score(s, durations, demands, capacities,
                                              norm_tail, norm_slack_inv, norm_dur,
                                              current_usage, start_times)
                heapq.heappush(eligible_heap, (-score, s))
        
        # Advance time if necessary
        current_time = end_time
    
    return simulated_order


def _compute_dynamic_score(activity, durations, demands, capacities,
                           norm_tail, norm_slack_inv, norm_dur,
                           current_usage, start_times):
    """
    Compute a dynamic priority score for an activity given current resource state.
    
    Combines:
    1. Static CPM metrics (tail, slack, duration)
    2. Dynamic resource slack: penalize activities that would cause bottlenecks
    """
    n_resources = len(capacities)
    
    # Base score from static metrics
    base_score = 0.50 * norm_tail[activity] + \
                 0.30 * norm_slack_inv[activity] + \
                 0.20 * norm_dur[activity]
    
    # Dynamic resource slack penalty
    # For each resource, compute how much slack would remain after scheduling
    # Penalize activities that consume a large fraction of available capacity
    resource_penalty = 0.0
    for r in range(n_resources):
        if capacities[r] > 0:
            available = capacities[r] - current_usage[r]
            if available <= 0:
                # Resource already exhausted - heavy penalty
                resource_penalty += 5.0
            else:
                demand_ratio = demands[activity][r] / available
                # Convex penalty: emphasizes high utilization
                resource_penalty += demand_ratio * demand_ratio
        else:
            if demands[activity][r] > 0:
                resource_penalty += 5.0
    
    # Normalize resource penalty to a reasonable range
    # Scale it down so it doesn't dominate the base score
    resource_penalty_scaled = 0.30 * resource_penalty / (n_resources + 1)
    
    # Final score: base minus resource penalty
    # Higher is better
    score = base_score - resource_penalty_scaled
    
    return score


def priority(instance):
    """
    Score each activity for the serial schedule-generation scheme.
    Higher score = higher priority (scheduled first when eligible).
    """
    start_time = time.time()
    
    try:
        scores = _compute_priority(instance)
        elapsed = time.time() - start_time
        if elapsed > _DEADLINE_SECONDS:
            # Fallback
            return np.asarray(instance["durations"], dtype=float)
        return scores
    except Exception:
        # Fallback: longest processing time
        try:
            return np.asarray(instance["durations"], dtype=float)
        except Exception:
            n = len(instance.get("durations", []))
            return np.ones(n, dtype=float)
# EVOLVE-BLOCK-END
