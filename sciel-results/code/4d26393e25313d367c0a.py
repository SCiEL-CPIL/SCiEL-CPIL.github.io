"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict, deque

def priority(instance):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    
    # 1. Parse Instance
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    predecessors = instance.get("predecessors", None)
    
    # Build adjacency list for predecessors and successors
    pred_list = [list() for _ in range(n)]
    succ_list = [list() for _ in range(n)]
    
    if predecessors is not None:
        if isinstance(predecessors, dict):
            for act, preds in predecessors.items():
                act_idx = int(act)
                for p in preds:
                    p_idx = int(p)
                    if act_idx not in succ_list[p_idx]:
                        succ_list[p_idx].append(act_idx)
                    if p_idx not in pred_list[act_idx]:
                        pred_list[act_idx].append(p_idx)
        elif isinstance(predecessors, (list, tuple, np.ndarray)):
            for i, preds in enumerate(predecessors):
                if preds is not None:
                    if isinstance(preds, (list, tuple, np.ndarray)):
                        for p in preds:
                            p_idx = int(p)
                            if i not in succ_list[p_idx]:
                                succ_list[p_idx].append(i)
                            if p_idx not in pred_list[i]:
                                pred_list[i].append(p_idx)
                    elif isinstance(preds, (int, np.integer)):
                        p_idx = int(preds)
                        if i not in succ_list[p_idx]:
                            succ_list[p_idx].append(i)
                        if p_idx not in pred_list[i]:
                            pred_list[i].append(p_idx)
    
    # Build demands structure
    n_resources = 0
    demand_matrix = None
    cap_vector = None
    
    if demands is not None:
        if isinstance(demands, (list, tuple, np.ndarray)):
            d_arr = np.asarray(demands)
            if d_arr.ndim == 2:
                demand_matrix = d_arr.astype(float)
                n_resources = d_arr.shape[1]
            elif d_arr.ndim == 1:
                demand_matrix = d_arr.reshape(-1, 1).astype(float)
                n_resources = 1
        elif isinstance(demands, dict):
            res_keys = list(demands.keys())
            n_resources = len(res_keys)
            demand_matrix = np.zeros((n, n_resources))
            for j, rk in enumerate(res_keys):
                val = demands[rk]
                if isinstance(val, (list, tuple, np.ndarray)):
                    demand_matrix[:, j] = np.asarray(val, dtype=float)
                else:
                    demand_matrix[:, j] = float(val)
    
    if capacities is not None:
        if isinstance(capacities, (list, tuple, np.ndarray)):
            cap_vector = np.asarray(capacities, dtype=float)
            if cap_vector.ndim == 0:
                cap_vector = np.full(n_resources, float(cap_vector)) if n_resources > 0 else np.array([1.0])
        elif isinstance(capacities, dict):
            cap_vector = np.zeros(n_resources)
            if n_resources > 0:
                if len(capacities) == 1:
                    cap_vector = np.full(n_resources, float(list(capacities.values())[0]))
                else:
                    for j in range(n_resources):
                        if j in capacities:
                            cap_vector[j] = float(capacities[j])
                        elif str(j) in capacities:
                            cap_vector[j] = float(capacities[str(j)])
                        else:
                            cap_vector[j] = float(list(capacities.values())[j]) if j < len(capacities) else 1.0
    
    if demand_matrix is None:
        demand_matrix = np.zeros((n, 0))
        cap_vector = np.array([])
        n_resources = 0
    
    if cap_vector is None or len(cap_vector) != n_resources:
        cap_vector = np.ones(n_resources)

    # 2. Topological Sort & Initial Path Calculations
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        in_degree[i] = len(pred_list[i])
        
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    temp_in_deg = in_degree.copy()
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for j in succ_list[node]:
            temp_in_deg[j] -= 1
            if temp_in_deg[j] == 0:
                queue.append(j)
                
    if len(topo_order) < n:
        for i in range(n):
            if i not in topo_order:
                topo_order.append(i)

    # Forward Pass: Earliest Start
    earliest_start = np.zeros(n)
    for node in topo_order:
        if pred_list[node]:
            earliest_start[node] = max(earliest_start[p] + durations[p] for p in pred_list[node])
            
    project_end = max(earliest_start[i] + durations[i] for i in range(n)) if n > 0 else 0
    
    # Backward Pass: Latest Start
    latest_finish = np.full(n, project_end)
    for node in reversed(topo_order):
        if succ_list[node]:
            latest_finish[node] = min(earliest_start[s] for s in succ_list[node])
        else:
            latest_finish[node] = project_end
            
    latest_start = latest_finish - durations
    total_float = latest_start - earliest_start
    total_float = np.maximum(total_float, 0)
    
    # Longest Path from node to end
    longest_path = np.zeros(n)
    for node in reversed(topo_order):
        if succ_list[node]:
            longest_path[node] = durations[node] + max(longest_path[s] for s in succ_list[node])
        else:
            longest_path[node] = durations[node]

    # 3. Compute "Resource Bottleneck Centrality"
    # This is the core of the new strategy.
    # For each activity i, calculate the sum over all resources r of:
    #   (demand[i][r] / capacity[r]) * (number of downstream activities that depend on i)
    # This identifies activities that:
    # 1. Consume a large fraction of a scarce resource (high demand/capacity ratio)
    # 2. Gate a large portion of the remaining project (many downstream dependents)
    
    # First, compute the number of downstream activities for each node (including itself? No, just dependents).
    # Let's compute "downstream_count[i]" = number of nodes j such that i is an ancestor of j.
    # We can do this with a topological sort (reverse order) and dynamic programming.
    
    downstream_count = np.zeros(n)
    # Process in reverse topological order
    for node in reversed(topo_order):
        # downstream_count[node] = 1 (itself) + sum of downstream counts of its immediate successors?
        # No, that would double count.
        # Actually, the number of nodes that depend on i is the size of the set of all successors (transitive).
        # Let's use a different approach: for each node, count how many times it appears as a predecessor in the transitive closure.
        # Or simpler: for each node j, find all its ancestors and increment their count.
        # But that's O(n^2). With n=30, it's fine.
        
        # Alternative: 
        # downstream_count[i] = sum over all j != i of 1 if i is an ancestor of j.
        # We can compute this by: for each j, traverse its ancestors and increment a counter.
        
        pass
    
    # Let's compute downstream_count properly.
    # downstream_count[i] = number of nodes j such that there is a path from i to j.
    # We can use the successor list and do a BFS/DFS from each node, but that's O(n^2 * (n+m)).
    # With n=30, it's very fast.
    
    downstream_count = np.zeros(n)
    for i in range(n):
        # BFS/DFS from i to find all reachable nodes
        visited = set()
        stack = list(succ_list[i])
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                for s in succ_list[node]:
                    if s not in visited:
                        stack.append(s)
        downstream_count[i] = len(visited)
    
    # Now compute the bottleneck centrality score
    # score[i] = sum over resources r of (demand[i][r] / capacity[r]) * downstream_count[i]
    
    bottleneck_score = np.zeros(n)
    for i in range(n):
        for r in range(n_resources):
            if cap_vector[r] > 0:
                bottleneck_score[i] += (demand_matrix[i][r] / cap_vector[r]) * downstream_count[i]
            else:
                # If capacity is 0, any demand is infinite bottleneck.
                if demand_matrix[i][r] > 0:
                    bottleneck_score[i] += float('inf')
    
    # Normalize bottleneck score
    def normalize(arr):
        mn = np.min(arr)
        mx = np.max(arr)
        if mx > mn:
            return (arr - mn) / (mx - mn)
        return np.zeros_like(arr)
    
    norm_bottleneck = normalize(bottleneck_score)
    
    # Combine with other signals?
    # The directive says to shift focus from static critical path to resource bottleneck centrality.
    # But we might still want some critical path info.
    # Let's try a pure bottleneck centrality first, or blend it.
    
    # Let's blend with critical path (longest_path) and low float.
    norm_longest_path = normalize(longest_path)
    norm_low_float = 1.0 - normalize(total_float)
    
    # Weights: emphasize bottleneck centrality heavily.
    alpha = 0.7  # Weight for bottleneck centrality
    beta = 0.2   # Weight for critical path
    gamma = 0.1  # Weight for low float
    
    current_priority = alpha * norm_bottleneck + beta * norm_longest_path + gamma * norm_low_float
    
    # 4. Optional: Iterative Refinement (if time permits)
    # We can simulate the schedule with this priority and see if we can improve.
    # But the core idea is the bottleneck centrality. Let's try a few iterations.
    
    max_iters = 5
    for it in range(max_iters):
        if time.time() > deadline - 5.0:
            break
            
        # Simulate Serial Schedule Generation
        start_times, end_times, scheduled_order = simulate_schedule(
            durations, pred_list, succ_list, demand_matrix, cap_vector, current_priority, n
        )
        
        if scheduled_order is None or len(scheduled_order) < n:
            break
            
        # Calculate Dynamic Blocking Score (same as before)
        dynamic_score = np.zeros(n)
        actual_delay = start_times - earliest_start
        actual_delay = np.maximum(actual_delay, 0)
        
        for i in range(n):
            if succ_list[i]:
                s_delay_sum = sum(actual_delay[s] for s in succ_list[i])
                dynamic_score[i] = s_delay_sum
            else:
                dynamic_score[i] = actual_delay[i]
        
        norm_dynamic = normalize(dynamic_score)
        
        # Blend: The dynamic score tells us what actually happened in the last simulation.
        # We want to fix the bottlenecks identified.
        # So, high dynamic score = high priority (to try and resolve the bottleneck earlier).
        
        # Also keep the bottleneck centrality and static signals.
        alpha_dyn = 0.4  # Weight for dynamic
        alpha_bot = 0.4  # Weight for bottleneck
        alpha_static = 0.2  # Weight for static (critical path + float)
        
        new_priority = alpha_dyn * norm_dynamic + alpha_bot * norm_bottleneck + alpha_static * (0.5 * norm_longest_path + 0.5 * norm_low_float)
        
        # Check for convergence or improvement
        if it > 2:
            # Simple check: correlation between old and new
            if np.corrcoef(current_priority, new_priority)[0,1] > 0.99:
                current_priority = new_priority
                break
        
        current_priority = new_priority
    
    # Final return
    # Ensure we return a valid array
    if len(current_priority) != n:
        return current_priority
        
    return current_priority


def simulate_schedule(durations, pred_list, succ_list, demand_matrix, cap_vector, priority, n):
    """
    Serial Schedule Generation Scheme.
    Returns: start_times, end_times, scheduled_order
    """
    n_resources = demand_matrix.shape[1]
    
    # Track resource usage over time
    # Since time is continuous/discrete, we use a list of (time, resource_delta) events or a dict.
    # For 30 activities, a simple list of active activities at any time t is fine.
    # Or, since we schedule one by one, we can just check availability at the proposed start time.
    
    # Proposed start time for each activity
    # An activity is eligible if all predecessors are finished.
    # We pick the highest priority eligible activity.
    
    scheduled = [False] * n
    start_times = np.zeros(n)
    end_times = np.zeros(n)
    scheduled_order = []
    
    # Resource usage: list of (end_time, resource_index, amount)
    # To check if resource r is available at time t for amount d:
    # Sum of amounts of all activities using r that started before t and end after t.
    
    # For efficiency with small n, we can just iterate through scheduled activities.
    
    remaining = set(range(n))
    
    while remaining:
        # Find eligible activities
        eligible = []
        for i in remaining:
            if all(scheduled[p] for p in pred_list[i]):
                eligible.append(i)
                
        if not eligible:
            # Deadlock or cycle? Break.
            break
            
        # Pick highest priority
        best = max(eligible, key=lambda x: priority[x])
        
        # Find earliest start time for best
        # 1. Predecessor constraint
        es_pred = 0
        for p in pred_list[best]:
            es_pred = max(es_pred, end_times[p])
            
        # 2. Resource constraint
        es_res = es_pred
        # We need to find the earliest time t >= es_pred such that for all resources r,
        # usage_r(t) + demand[best][r] <= cap[r]
        
        # Get all relevant end times of currently scheduled activities that overlap with es_pred
        # Actually, we just need to check at es_pred, and if it fails, jump to the next event time.
        
        # Collect all end times of scheduled activities
        # To be efficient, we can just check es_pred, then if conflict, find the min end time of conflicting activities.
        
        t = es_pred
        # Max iterations to avoid infinite loop in case of weird data
        max_attempts = n + 1
        attempts = 0
        while attempts < max_attempts:
            attempts += 1
            # Check resource availability at t
            feasible = True
            for r in range(n_resources):
                demand_needed = demand_matrix[best][r]
                if demand_needed == 0:
                    continue
                # Check usage at time t
                usage = 0.0
                for i in range(n):
                    if i == best or not scheduled[i]:
                        continue
                    # If activity i is active at time t
                    if start_times[i] < t and end_times[i] > t:
                        usage += demand_matrix[i][r]
                    # Note: If end_times[i] == t, it is not active at t (assuming [start, end) or start inclusive, end exclusive)
                    # Standard RCPSP: [start, end). So if end == t, it's free.
                    
                if usage + demand_needed > cap_vector[r] + 1e-9:
                    feasible = False
                    # Find the earliest end time among conflicting activities
                    # We need to jump to the next time a resource frees up.
                    # Find min end_time of activities i such that i is active at t and uses resource r.
                    min_end = float('inf')
                    for i in range(n):
                        if i == best or not scheduled[i]:
                            continue
                        if start_times[i] < t and end_times[i] > t and demand_matrix[i][r] > 0:
                            if end_times[i] < min_end:
                                min_end = end_times[i]
                    if min_end == float('inf'):
                        # Should not happen if feasible was False
                        t = t + 1.0
                    else:
                        t = min_end
                    break # Re-check all resources at new t
            
            if feasible:
                break
                
        start_times[best] = t
        end_times[best] = t + durations[best]
        scheduled[best] = True
        scheduled_order.append(best)
        remaining.remove(best)
        
    return start_times, end_times, scheduled_order
# EVOLVE-BLOCK-END
