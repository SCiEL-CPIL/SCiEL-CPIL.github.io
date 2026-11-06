"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict, deque

def priority(instance):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    # Parse instance structure
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    predecessors = instance.get("predecessors", None)
    
    # Build adjacency list for predecessors
    pred_list = [list() for _ in range(n)]
    if predecessors is not None:
        if isinstance(predecessors, dict):
            for act, preds in predecessors.items():
                act_idx = int(act)
                for p in preds:
                    pred_list[act_idx].append(int(p))
        elif isinstance(predecessors, (list, tuple, np.ndarray)):
            for i, preds in enumerate(predecessors):
                if preds is not None:
                    if isinstance(preds, (list, tuple, np.ndarray)):
                        pred_list[i] = [int(p) for p in preds]
                    elif isinstance(preds, (int, np.integer)):
                        pred_list[i] = [int(preds)]
    
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
        
    # Build Successor list
    succ_list = [list() for _ in range(n)]
    for i in range(n):
        for p in pred_list[i]:
            succ_list[p].append(i)
            
    # Compute Topological Order
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for p in pred_list[i]:
            in_degree[i] += 1
            
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
            
    # Backward Pass: Latest Start
    project_end = max(earliest_start[i] + durations[i] for i in range(n)) if n > 0 else 0
    latest_finish = np.full(n, project_end)
    for node in reversed(topo_order):
        if succ_list[node]:
            latest_finish[node] = min(earliest_start[s] for s in succ_list[node])
        else:
            latest_finish[node] = project_end
            
    latest_start = latest_finish - durations
    total_float = latest_start - earliest_start
    total_float = np.maximum(total_float, 0)
    
    # Compute Longest Path from node to end (Critical Path Length)
    longest_path = np.zeros(n)
    for node in reversed(topo_order):
        if succ_list[node]:
            longest_path[node] = durations[node] + max(longest_path[s] for s in succ_list[node])
        else:
            longest_path[node] = durations[node]
            
    # Helper: Simulate serial schedule given a priority array
    def simulate(priority_scores):
        # priority_scores: higher is better
        # Returns: start times, makespan, set of critical path nodes
        start_times = np.zeros(n)
        # resource availability over time: we will track end times per resource
        # Since n=30, we can use a simple list of (end_time, resource_idx) or just arrays
        
        # We maintain current resource usage by tracking end times.
        # For each time step (which will be the start of an activity), we check availability.
        
        # To be efficient, we can just keep track of end times for each resource.
        # resource_end_times[j] = list of end times? No, just the current "busy until" times.
        # Actually, since we schedule sequentially, we can just keep a list of (end_time, res_idx) for active jobs?
        # No, simpler: for each resource, keep a list of end times of jobs that have started.
        # When starting a job at time t, we can just check if demand <= capacity - sum(demands of jobs active at t).
        
        # Better:
        # Maintain `resource_usage` as a list of (end_time, demand_vector) for currently active jobs?
        # Or simply, for each resource j, maintain a list of end_times.
        # To check availability at time t:
        #   For each resource j, count how many jobs with end_time > t are active.
        #   Sum their demands.
        #   If sum + new_demand <= capacity, it's available.
        
        # With n=30, this is fast enough.
        
        # Let's use a more standard approach:
        # Keep track of `active_jobs` = list of (end_time, activity_idx)
        # When scheduling activity i at time t:
        #   1. Remove jobs from active_jobs where end_time <= t.
        #   2. Calculate current resource usage from remaining active_jobs.
        #   3. Check if i fits.
        #   4. If not, find the earliest time t' >= t where it fits.
        
        # Actually, finding the exact earliest time t' is tricky if we don't know future end times.
        # Standard Serial SGS:
        #   1. Select eligible activity with highest priority.
        #   2. Determine start time t:
        #      - t_min = max(ES[i], max(end_time of predecessors))
        #      - Find smallest t >= t_min such that resources are available.
        #      - To find t, we can look at the end times of currently active jobs.
        
        # Implementation:
        #   `active_jobs` is a list of (end_time, activity_idx).
        #   When starting i at t:
        #       - We know all jobs in `active_jobs` are running at t (since we process in chronological order of start times? No, we process in order of selection, but start times must be non-decreasing).
        #       - Wait, Serial SGS ensures start times are non-decreasing.
        #       - So, at any point, `active_jobs` contains jobs that have started but not finished.
        #       - The current time is the start time of the last scheduled job.
        
        # Let's refine:
        #   `scheduled` = set of scheduled activities.
        #   `start_times` = dict/array.
        #   `active_jobs` = list of (end_time, activity_idx).
        
        #   For each step:
        #       1. Find eligible activities (all predecessors scheduled).
        #       2. Select best by priority.
        #       3. Calculate earliest possible start `t_early` based on predecessors.
        #       4. Find the actual start time `t_start`:
        #          - Initialize `t_candidate` = t_early.
        #          - Loop:
        #             - Check resource availability at `t_candidate`.
        #             - To check availability, we need to know which jobs are active at `t_candidate`.
        #             - A job is active at `t` if `start_time[j] <= t < end_time[j]`.
        #             - We can pre-calculate end times for all scheduled jobs.
        #             - Since n=30, we can just iterate through all scheduled jobs to check availability.
        
        #   This O(n^2) or O(n^3) check is fine for n=30.
        
        start_times = np.zeros(n)
        scheduled = [False] * n
        # We need to track end times of scheduled jobs to check availability.
        end_times = np.zeros(n)
        
        # To speed up, we can keep a list of (end_time, activity_idx) for jobs that are "active" or "future"?
        # Actually, just checking all scheduled jobs is O(n) per check.
        # Finding the next available time might require checking multiple time points.
        
        # Let's optimize the time search:
        # The resource availability only changes at the start or end times of jobs.
        # So the candidate start times are:
        # 1. t_early
        # 2. End times of any currently active job that is consuming a resource needed by i.
        
        for _ in range(n):
            # 1. Find eligible
            eligible = []
            for i in range(n):
                if not scheduled[i]:
                    # Check predecessors
                    ok = True
                    for p in pred_list[i]:
                        if not scheduled[p]:
                            ok = False
                            break
                    if ok:
                        eligible.append(i)
                        
            if not eligible:
                # Should not happen if DAG is valid
                break
                
            # 2. Select best
            best_i = max(eligible, key=lambda x: priority_scores[x])
            
            # 3. Calculate t_early
            t_early = 0
            for p in pred_list[best_i]:
                t_early = max(t_early, start_times[p] + durations[p])
            
            # 4. Find t_start
            t_candidate = t_early
            
            # We will iterate through potential start times.
            # Potential times are t_early and end times of jobs active at t_early or starting later.
            # Actually, if resources are not available at t, the earliest time they might be is when some active job finishes.
            
            found = False
            while not found:
                # Check availability at t_candidate
                # Calculate resource usage at t_candidate
                usage = np.zeros(n_resources)
                for j in range(n):
                    if scheduled[j]:
                        if start_times[j] <= t_candidate < end_times[j]:
                            usage += demand_matrix[j]
                            
                if np.all(usage + demand_matrix[best_i] <= cap_vector):
                    found = True
                else:
                    # Find the earliest end time of any job active at t_candidate
                    # that is consuming a resource where we have a conflict.
                    # Actually, we just need to jump to the next event time (end time of some active job).
                    next_end = np.inf
                    for j in range(n):
                        if scheduled[j]:
                            if start_times[j] <= t_candidate < end_times[j]:
                                # Check if this job is causing a conflict
                                # If usage + demand > cap for any resource, and this job uses that resource
                                # We can just jump to the minimum end time of ANY active job.
                                # This might skip some valid times if a job finishes but doesn't free the specific resource,
                                # but it will eventually find the right time.
                                next_end = min(next_end, end_times[j])
                    
                    if next_end == np.inf:
                        # No active jobs? Then resources should be available.
                        # This case shouldn't happen if we checked availability and failed.
                        # Unless demand > capacity?
                        break 
                    
                    t_candidate = next_end
                    
            start_times[best_i] = t_candidate
            end_times[best_i] = t_candidate + durations[best_i]
            scheduled[best_i] = True
            
        makespan = np.max(end_times) if n > 0 else 0
        
        # Identify Critical Path in the schedule
        # A node is critical if it lies on a path from source to sink with length == makespan
        # We can find this by tracing back from the sink.
        # Sink: node with no successors (or all successors scheduled).
        # Actually, let's find all nodes that are part of a longest path in the *scheduled* graph.
        
        # Build scheduled graph
        # dist[i] = length of longest path from i to end, considering only scheduled edges?
        # No, the precedence graph is fixed. The *times* change.
        # The critical path in the schedule is the sequence of activities that determines the makespan.
        
        # Let's compute `slack` in the schedule.
        # Late finish LF[i] = makespan for sinks.
        # LF[i] = min(LF[s] - durations[s] for s in succ_list[i])? No, that's for CPM.
        # In a schedule, the "lateness" is determined by the specific start times.
        
        # Alternative:
        # A node i is on the critical path if:
        # ES[i] (in schedule) + durations[i] + (Longest path from i to end in precedence graph) == Makespan?
        # Not exactly, because resource constraints might delay successors.
        
        # Better approach:
        # Compute `backward_time[i]` = latest time i can finish without delaying the project.
        # Backward pass on the *scheduled* network.
        # For sinks: BW[i] = makespan
        # For others: BW[i] = min(BW[s] - durations[s] for s in succ_list[i] if scheduled[s])
        # Wait, this assumes successors start immediately after i finishes?
        # No, in the schedule, successor s starts at ST[s].
        # So the constraint is: ST[s] >= ET[i].
        # The "float" of i is min(ST[s] - ET[i]) for all successors s.
        # If float is 0, it's critical.
        
        # Let's compute `min_successor_start[i]` = min(ST[s] for s in succ_list[i])
        # If ET[i] == min_successor_start[i], then i is "tight" against its successors.
        # A critical path is a chain of tight activities from start to end.
        
        # Let's identify nodes with zero "schedule float".
        # Float[i] = min(ST[s] - ET[i]) for s in succ_list[i].
        # For sinks, Float[i] = makespan - ET[i].
        
        # Then find the path from a source to a sink where all nodes have 0 float?
        # Not necessarily, but the critical path consists of nodes where the delay propagates.
        
        # Simple heuristic for "critical set":
        # Nodes i such that ET[i] + LongestPath[i] >= Makespan - epsilon?
        # This is a good proxy.
        
        critical_set = set()
        for i in range(n):
            # ET[i] + LongestPath[i] is the earliest the project can finish if i starts at ST[i]
            # and successors start as early as possible.
            # If this is close to Makespan, i is critical.
            if ET[i] + longest_path[i] >= makespan - 1e-6:
                critical_set.add(i)
                
        return start_times, makespan, critical_set

    # Initial Priority: Standard CPM-based
    # High longest_path, low float
    def normalize(arr):
        mn = np.min(arr)
        mx = np.max(arr)
        if mx > mn:
            return (arr - mn) / (mx - mn)
        return np.zeros_like(arr)
        
    norm_longest_path = normalize(longest_path)
    norm_float_inv = 1.0 - normalize(total_float)
    
    current_priority = 0.5 * norm_longest_path + 0.5 * norm_float_inv
    
    best_makespan = np.inf
    best_priority = current_priority.copy()
    
    # Iterative Improvement Loop
    for iteration in range(50):
        if time.time() > deadline:
            break
            
        _, makespan, critical_set = simulate(current_priority)
        
        if makespan < best_makespan:
            best_makespan = makespan
            best_priority = current_priority.copy()
            
        # Check for stagnation
        if makespan >= best_makespan - 1e-6:
            # If we haven't improved, try a different perturbation or break
            # If we've done a few iterations without improvement, break
            if iteration > 5:
                break
                
        # Update Priority
        # Boost activities in the critical set
        # Also, prioritize activities that have many successors in the critical set?
        
        new_priority = current_priority.copy()
        
        # Boost critical activities
        for i in critical_set:
            new_priority[i] += 1.0
            
        # Secondary boost: activities that are predecessors to critical activities
        # This helps "unlock" the critical path
        for i in range(n):
            if i not in critical_set:
                # Check if any successor is critical
                has_critical_succ = any(s in critical_set for s in succ_list[i])
                if has_critical_succ:
                    new_priority[i] += 0.5
                    
        # Normalize to keep values in a reasonable range?
        # Or just use relative ordering.
        current_priority = new_priority
        
    # Final simulation to ensure best_priority is valid and corresponds to best_makespan
    # (It should, but just in case)
    _, final_makespan, _ = simulate(best_priority)
    
    return best_priority
# EVOLVE-BLOCK-END
