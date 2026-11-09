import numpy as np
import time

def priority(instance):
    t_start = time.time()
    DEADLINE = 150.0
    SAFETY_MARGIN = 10.0

    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)  # (n, n_res)
    capacities = np.asarray(instance["capacabilities"], dtype=float) if "capacabilities" in instance else np.asarray(instance["capacities"], dtype=float)
    predecessors = instance["predecessors"]  # list of lists
    
    n = len(durations)
    n_res = len(capacities)
    
    if n == 0:
        return []
    
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
    
    project_end = np.max(ef) if n > 0 else 0.0
    
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
    
    # Resource demand intensity: total demand relative to capacity
    if n_res > 0:
        demand_ratios = demands / np.maximum(capacities, 1e-9)  # (n, n_res)
        max_demand_ratio = np.max(demand_ratios, axis=1)  # (n,)
        # Sum of demand ratios (total resource pressure)
        sum_demand_ratio = np.sum(demand_ratios, axis=1)  # (n,)
    else:
        max_demand_ratio = np.zeros(n)
        sum_demand_ratio = np.zeros(n)
    
    # Normalize features
    max_float = np.max(total_float) if np.max(total_float) > 0 else 1.0
    max_tail = np.max(tail) if np.max(tail) > 0 else 1.0
    max_dur = np.max(durations) if np.max(durations) > 0 else 1.0
    max_ratio = np.max(max_demand_ratio) if np.max(max_demand_ratio) > 0 else 1.0
    max_sum_ratio = np.max(sum_demand_ratio) if np.max(sum_demand_ratio) > 0 else 1.0
    
    # Check deadline before computing final score
    if time.time() - t_start > DEADLINE - SAFETY_MARGIN:
        # Return a simple fallback based on criticality
        return (max_float - total_float) / max_float
    
    # Build conflict graph for resource contention
    # Two activities conflict if they share a resource where their combined demand exceeds capacity
    # AND they could potentially overlap in time (i.e., not strictly ordered by precedence)
    
    # First, determine which pairs are NOT ordered by precedence (can potentially overlap)
    # Compute transitive closure of precedence (which activity precedes which)
    # For n=30, this is cheap
    precedes = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for p in predecessors[i]:
            precedes[p][i] = True
    
    # Transitive closure (Floyd-Warshall style, n=30 is tiny)
    for k in range(n):
        for i in range(n):
            if precedes[i][k]:
                precedes[i] |= precedes[k]
    
    # Build conflict adjacency matrix
    # A[i][j] = 1 if activities i and j can overlap AND conflict on some resource
    conflict = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i+1, n):
            # Skip if one precedes the other (they can't overlap)
            if precedes[i][j] or precedes[j][i]:
                continue
            # Check resource conflict: if they share a resource where combined demand > capacity
            # Only consider renewable resources (all are renewable here)
            if n_res > 0:
                for r in range(n_res):
                    if demands[i][r] + demands[j][r] > capacities[r]:
                        conflict[i][j] = 1.0
                        conflict[j][i] = 1.0
                        break
    
    # Compute eigenvector centrality of the conflict graph
    # The dominant eigenvector of the adjacency matrix gives centrality scores
    # Activities with high centrality are structurally dominant bottlenecks
    try:
        # Use power iteration for the dominant eigenvector (faster than full eigen for sparse)
        # Initialize with uniform vector
        vec = np.ones(n) / n
        for _ in range(50):
            new_vec = conflict @ vec
            norm = np.linalg.norm(new_vec)
            if norm < 1e-15:
                break
            new_vec /= norm
            if np.linalg.norm(new_vec - vec) < 1e-10:
                break
            vec = new_vec
        
        # Normalize eigenvector centrality to [0, 1]
        max_centrality = np.max(vec) if np.max(vec) > 0 else 1.0
        eig_centrality = vec / max_centrality
    except Exception:
        eig_centrality = np.zeros(n)
    
    # Check deadline
    if time.time() - t_start > DEADLINE - SAFETY_MARGIN:
        # Fallback to static CPM-based priority
        priority_scores = (
            0.45 * (max_float - total_float) / max_float +
            0.30 * tail / max_tail +
            0.10 * durations / max_dur +
            0.08 * max_demand_ratio / max_ratio +
            0.07 * sum_demand_ratio / max_sum_ratio
        )
        return np.maximum(priority_scores, 0.0)
    
    # Dynamic Look-Ahead Simulation:
    # Run a fast serial schedule simulation with a baseline rule to estimate
    # each activity's actual completion time under resource constraints.
    
    def simulate_serial_schedule(rule_func):
        """Run serial schedule generation with a given priority rule.
        Returns estimated completion times for each activity."""
        scheduled = [False] * n
        start_times = np.zeros(n)
        end_times = np.zeros(n)
        current_time = 0.0
        
        for _ in range(n):
            if time.time() - t_start > DEADLINE - SAFETY_MARGIN:
                break
            
            # Find eligible activities (all predecessors completed)
            eligible = []
            for i in range(n):
                if not scheduled[i]:
                    all_preds_done = all(scheduled[p] for p in predecessors[i])
                    if all_preds_done:
                        eligible.append(i)
            
            if not eligible:
                break
            
            # Apply the rule to pick the highest priority eligible activity
            best_activity = max(eligible, key=lambda i: rule_func(i, start_times, end_times, scheduled))
            
            # Determine start time considering resource constraints
            earliest_start = current_time
            for p in predecessors[best_activity]:
                earliest_start = max(earliest_start, end_times[p])
            
            # Check resource availability from earliest_start
            t = earliest_start
            max_search_time = project_end + max_dur * 3
            while t < max_search_time:
                can_start = True
                for r in range(n_res):
                    usage = 0.0
                    for j in range(n):
                        if scheduled[j] and start_times[j] <= t < end_times[j]:
                            usage += demands[j][r]
                    if usage + demands[best_activity][r] > capacities[r]:
                        can_start = False
                        break
                
                if can_start:
                    break
                
                # Find next event time
                next_event = None
                for j in range(n):
                    if scheduled[j] and start_times[j] < t < end_times[j]:
                        if next_event is None or end_times[j] < next_event:
                            next_event = end_times[j]
                
                if next_event is not None and next_event > t:
                    t = next_event
                else:
                    t += 1.0
            
            start_times[best_activity] = t
            end_times[best_activity] = t + durations[best_activity]
            scheduled[best_activity] = True
            current_time = t
        
        return start_times, end_times
    
    # Baseline rule: combine CPM criticality with eigenvector centrality
    def baseline_rule(i, st, et, sched):
        # Higher score = higher priority
        # Blend criticality (low float) with structural bottleneck importance
        return -total_float[i] + 0.5 * eig_centrality[i] * max_float + durations[i] * 0.3
    
    # Run simulation with baseline rule
    sim_start, sim_end = simulate_serial_schedule(baseline_rule)
    
    # Check deadline
    if time.time() - t_start > DEADLINE - SAFETY_MARGIN:
        # Fallback to static CPM-based priority with eigenvector
        priority_scores = (
            0.40 * (max_float - total_float) / max_float +
            0.25 * tail / max_tail +
            0.15 * eig_centrality +
            0.08 * durations / max_dur +
            0.07 * max_demand_ratio / max_ratio +
            0.05 * sum_demand_ratio / max_sum_ratio
        )
        return np.maximum(priority_scores, 0.0)
    
    # Use simulated end times as priority (earlier completion = higher priority)
    max_sim_end = np.max(sim_end) if np.max(sim_end) > 0 else 1.0
    sim_priority = 1.0 - sim_end / max_sim_end
    
    # Combine with CPM signals, eigenvector centrality, and resource features
    # Weight the dynamic simulation results, blend with static features and spectral signal
    w_sim = 0.50
    w_float = 0.20
    w_tail = 0.10
    w_eig = 0.10
    w_res_max = 0.05
    w_res_sum = 0.05
    
    priority_scores = (
        w_sim * sim_priority +
        w_float * (max_float - total_float) / max_float +
        w_tail * tail / max_tail +
        w_eig * eig_centrality +
        w_res_max * max_demand_ratio / max_ratio +
        w_res_sum * sum_demand_ratio / max_sum_ratio
    )
    
    # Ensure non-negative scores
    priority_scores = np.maximum(priority_scores, 0.0)
    
    return priority_scores
