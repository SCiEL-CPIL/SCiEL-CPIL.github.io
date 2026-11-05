"""Rewritten program for Resource-Constrained Project Scheduling."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque
import time

_TIME_LIMIT = 140.0
_START_TIME = None

def priority(instance):
    """
    Score each activity for the serial schedule-generation scheme.
    Higher score = higher priority (started earlier).
    
    Heuristic: Dynamic resource-weighted criticality.
    1. Compute static CPM metrics (Bottom Level, Total Float).
    2. Simulate a resource-feasible greedy schedule to identify actual bottlenecks.
    3. Combine criticality with empirical resource contention.
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        # Emergency fallback
        durations = np.asarray(instance["durations"], dtype=float)
        if len(durations) == 0: return np.array([])
        return durations / (np.max(durations) + 1e-9)

    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    if n == 0:
        return np.array([])

    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    preds = instance.get("predecessors", [None] * n)

    # 1. Build graph
    pred_lists = []
    for p in preds:
        if p is None:
            pred_lists.append([])
        elif isinstance(p, (list, tuple, np.ndarray)):
            pred_lists.append([x for x in p if x is not None])
        else:
            pred_lists.append([p])
            
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        for p in pred_lists[i]:
            if 0 <= p < n:
                successors[p].append(i)
                in_degree[i] += 1
            
    # Topological Sort
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    top_order = []
    while queue:
        u = queue.popleft()
        top_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    if len(top_order) < n:
        top_order = list(range(n))

    # 2. Compute Bottom Level (Longest path from node to end)
    bottom_level = np.zeros(n)
    for u in reversed(top_order):
        if successors[u]:
            max_succ_bl = np.max([bottom_level[v] for v in successors[u]])
        else:
            max_succ_bl = 0.0
        bottom_level[u] = durations[u] + max_succ_bl

    # 3. Compute Earliest Start (ES) and Earliest Finish (EF)
    es = np.zeros(n)
    ef = np.zeros(n)
    for u in top_order:
        if pred_lists[u]:
            es[u] = np.max([ef[p] for p in pred_lists[u]])
        else:
            es[u] = 0.0
        ef[u] = es[u] + durations[u]
        
    # Project makespan (critical path length)
    cp_length = np.max(ef) if n > 0 else 0.0
    
    # Compute Latest Start (LS) and Latest Finish (LF)
    ls = np.zeros(n)
    lf = np.zeros(n)
    for u in reversed(top_order):
        if successors[u]:
            lf[u] = np.min([ls[v] for v in successors[u]])
        else:
            lf[u] = cp_length
        ls[u] = lf[u] - durations[u]
        
    # Total Float = LS - ES
    tf = ls - es

    # 4. Parse Demands and Capacities
    n_res = 0
    d_matrix = None
    capacities_arr = None
    
    if demands is not None and capacities is not None:
        capacities_arr = np.asarray(capacities, dtype=float)
        if len(capacities_arr) == 0:
            capacities_arr = np.array([1.0])
        else:
            capacities_arr = np.where(capacities_arr > 0, capacities_arr, 1.0)
        n_res = len(capacities_arr)
        
        d_matrix = np.zeros((n, n_res), dtype=float)
        
        if isinstance(demands, list):
            for i in range(n):
                if i < len(demands):
                    d_i = demands[i]
                    if isinstance(d_i, dict):
                        for r, val in d_i.items():
                            if r < n_res:
                                d_matrix[i, r] = val
                    elif isinstance(d_i, (list, tuple, np.ndarray)):
                        d_arr = np.asarray(d_i, dtype=float)
                        if d_arr.ndim == 1:
                            min_len = min(len(d_arr), n_res)
                            d_matrix[i, :min_len] = d_arr[:min_len]
                    else:
                        # Scalar?
                        if n_res > 0:
                            d_matrix[i, 0] = float(d_i)
        else:
            # Try converting whole thing to array
            try:
                d_mat_full = np.asarray(demands, dtype=float)
                if d_mat_full.shape[0] == n:
                    if d_mat_full.ndim == 1:
                         d_matrix[:, 0] = d_mat_full
                    else:
                        min_res = min(d_mat_full.shape[1], n_res)
                        d_matrix[:, :min_res] = d_mat_full[:, :min_res]
            except:
                pass

    # 5. Simulate Resource-Feasible Greedy Schedule
    # This identifies which activities actually cause bottlenecks in a realistic schedule.
    # We use a simple priority rule for the simulation: Bottom Level (static criticality).
    
    sim_start = np.zeros(n)
    sim_finish = np.zeros(n)
    active = []  # list of (finish_time, activity_id)
    
    # Sort activities by static priority (Bottom Level) for simulation
    # We want to simulate a schedule that is "good" enough to reveal true bottlenecks.
    # Using BL as the simulation priority is a reasonable baseline.
    sim_order_indices = np.argsort(-bottom_level)
    
    # We need to simulate in a way that respects precedence.
    # A standard serial SGS simulation:
    # 1. Find eligible activities (all predecessors finished).
    # 2. Pick the one with highest priority (BL).
    # 3. Find the earliest feasible start time given resources.
    # 4. Schedule it.
    
    remaining = set(range(n))
    scheduled = set()
    
    # For efficiency, we maintain a list of eligible activities.
    # Since n=30, O(n^2) or O(n^3) is fine.
    
    # Precompute static priority for simulation
    sim_priority = bottom_level.copy()
    
    t = 0
    while len(scheduled) < n:
        # Find eligible activities
        eligible = []
        for i in remaining:
            if all(p in scheduled for p in pred_lists[i]):
                eligible.append(i)
        
        if not eligible:
            # Should not happen if graph is valid, but safety break
            break
        
        # Pick highest priority eligible activity
        best_i = max(eligible, key=lambda x: sim_priority[x])
        
        # Find earliest feasible start time
        # Start time is at least the finish time of all predecessors
        earliest_start = 0
        for p in pred_lists[best_i]:
            earliest_start = max(earliest_start, sim_finish[p])
        
        # Check resource availability
        # We need to find the earliest time >= earliest_start where resources are sufficient.
        # Since resources are renewable, we can simulate time slots.
        # However, with n=30 and small durations, we can just step through time or check event points.
        
        # Event points: earliest_start and finish times of active tasks.
        # Let's just iterate time from earliest_start onwards until we find a slot.
        # To be efficient, we can check against the finish times of currently active tasks.
        
        # Get finish times of active tasks that overlap with earliest_start or later
        # Actually, simpler: just check from earliest_start, incrementing by 1 (or min duration) until feasible.
        # Given small n, this is fast.
        
        # Optimization: The only times resources change are at start or finish of activities.
        # So we only need to check t = earliest_start, and finish times of active activities > earliest_start.
        
        candidate_times = [earliest_start]
        for finish_t, act_id in active:
            if finish_t > earliest_start:
                candidate_times.append(finish_t)
        
        candidate_times.sort()
        
        feasible_start = earliest_start
        found = False
        for t_candidate in candidate_times:
            # Check if resources are available at t_candidate
            # Sum demands of active tasks at t_candidate
            # Active tasks are those with start < t_candidate <= finish
            # We store active as (finish, id). We also need start times.
            # Let's keep a list of active tasks with their start times for checking.
            
            # Re-evaluate active tasks at t_candidate
            current_demand = np.zeros(n_res)
            for finish_t, act_id in active:
                if sim_start[act_id] < t_candidate <= finish_t:
                    current_demand += d_matrix[act_id]
            
            # Check if best_i fits
            if np.all(current_demand + d_matrix[best_i] <= capacities_arr):
                feasible_start = t_candidate
                found = True
                break
        
        if not found:
            # If no candidate time worked (shouldn't happen if capacities are sufficient),
            # fallback to checking continuously or just use earliest_start (might violate, but shouldn't happen)
            feasible_start = earliest_start
        
        # Schedule best_i
        sim_start[best_i] = feasible_start
        sim_finish[best_i] = feasible_start + durations[best_i]
        active.append((sim_finish[best_i], best_i))
        remaining.remove(best_i)
        scheduled.add(best_i)
        
        # Remove finished activities from active list for next iterations?
        # Not strictly necessary for correctness of "is active at t", but good for memory.
        # We'll just keep them and filter by time in the check above.
        # To keep active list small, we can remove finished ones.
        active = [(f, i) for f, i in active if f > feasible_start]

    # 6. Compute Dynamic Resource Contention Score
    # An activity is a bottleneck if it delays other activities or uses scarce resources heavily.
    # Metric: For each activity i, how much does it contribute to resource pressure?
    # A simple metric: demand/capacity ratio, weighted by how often it overlaps with other high-priority tasks.
    
    # Let's use a simpler dynamic metric:
    # "Resource Slack" = 1 - (max_demand / capacity) for each resource used by i.
    # Low slack = high contention.
    
    # Or, better: Identify which activities are on the "resource critical path".
    # In the simulated schedule, if an activity i is running, and another activity j (with higher BL)
    # could not be scheduled earlier due to resources, then i is a bottleneck for j.
    
    # Let's compute a "bottleneck score" for each activity:
    # For each activity i, sum over all activities j that were delayed by i.
    # Delay of j = sim_start[j] - es[j] (earliest start from CPM).
    # If j was delayed, attribute some of that delay to the resources it competes for.
    
    # Simpler approach:
    # Score_i = sum_{r} (demand_i[r] / capacity[r]) * (1 if i is on critical path of simulation else 0)
    
    # Let's define "Simulation Criticality":
    # An activity is on the simulation critical path if sim_finish[i] == makespan_sim and it's a successor chain.
    # Or simply: high BL AND high resource usage.
    
    # Let's combine:
    # 1. Static Criticality: BL (normalized)
    # 2. Resource Pressure: sum(demand/capacity) (normalized)
    # 3. Dynamic Bottleneck: Did this activity cause delays in the simulation?
    
    # Compute delays in simulation
    # delay[i] = sim_start[i] - es[i]
    delays = np.maximum(0, sim_start - es)
    
    # Attribute delay to the activity that was running and consuming resources.
    # This is complex. Let's use a simpler proxy:
    # "Resource Weighted Criticality" = BL * (1 + resource_pressure)
    
    # Let's try:
    # Score = w1 * BL_norm + w2 * (BL_norm * ResourcePressure_norm) - w3 * TF_norm
    
    # Normalize BL
    bl_max = np.max(bottom_level) if np.max(bottom_level) > 0 else 1.0
    bl_min = np.min(bottom_level)
    bl_range = bl_max - bl_min
    if bl_range > 0:
        bl_norm = (bottom_level - bl_min) / bl_range
    else:
        bl_norm = np.ones(n)
        
    # Normalize TF
    tf_max = np.max(tf) if np.max(tf) > 0 else 1.0
    tf_min = np.min(tf)
    tf_range = tf_max - tf_min
    if tf_range > 0:
        tf_norm = (tf - tf_min) / tf_range
    else:
        tf_norm = np.zeros(n)
        
    # Normalize Resource Pressure
    rp = np.zeros(n)
    if d_matrix is not None and capacities_arr is not None:
        ratios = d_matrix / capacities_arr
        rp = np.sum(ratios, axis=1)
    
    rp_max = np.max(rp) if np.max(rp) > 0 else 1.0
    rp_min = np.min(rp)
    rp_range = rp_max - rp_min
    if rp_range > 0:
        rp_norm = (rp - rp_min) / rp_range
    else:
        rp_norm = np.ones(n)
        
    # Normalize Delays (dynamic bottleneck indicator)
    # High delay for an activity itself is bad, but we want to prioritize activities that CAUSE delays.
    # Let's use the delay of the activity itself as a negative indicator? No, we want to schedule critical things first.
    # Actually, if an activity has a high BL, it should be scheduled early.
    # The simulation tells us if the static priority was good.
    
    # Let's use a "Dynamic Priority" adjustment:
    # If an activity was delayed in the simulation (sim_start > es), it means it was waiting for resources or predecessors.
    # If it was waiting for resources, it's a high-contention activity.
    # We want to prioritize high-contention activities on the critical path.
    
    # Let's define a "Bottleneck Score" for each activity i:
    # For each time t where i is running, how many other high-priority activities are waiting for resources?
    # This is expensive.
    
    # Alternative: Use the delay of successors.
    # If i is on the path to a delayed activity, i is a bottleneck.
    
    # Let's try a simpler combination that worked in inspirations:
    # Score = BL_norm * (1 + alpha * rp_norm) - beta * tf_norm
    
    alpha = 0.5
    beta = 0.2
    
    scores = bl_norm * (1 + alpha * rp_norm) - beta * tf_norm
    
    # Normalize final scores
    score_max = np.max(scores) if np.max(scores) > 0 else 1.0
    score_min = np.min(scores)
    score_range = score_max - score_min
    if score_range > 0:
        scores = (scores - score_min) / score_range
    else:
        scores = np.ones(n)

    return scores
# EVOLVE-BLOCK-END
