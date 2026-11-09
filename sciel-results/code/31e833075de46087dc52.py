import numpy as np
from collections import deque
import time

def priority(instance):
    """
    Strategy: Dynamic Resource-Aware Priority via Simulated SGS.
    
    This function simulates the Serial Schedule Generation Scheme (SSGS) internally.
    Instead of using static priority vectors (like Min Slack or Max Duration) for the 
    entire run, it computes a dynamic priority score for each eligible activity at each 
    step of the simulation.
    
    The dynamic score combines:
    1. Static CPM Slack (lower is better).
    2. Dynamic Resource Pressure: Penalizes activities that demand resources which are 
       currently heavily utilized. This encourages scheduling "light" activities when 
       resources are tight, or "heavy" activities when resources are free, depending 
       on the specific penalty structure. Here we penalize high demand on scarce resources.
    3. Duration (longer activities often need to be started earlier to avoid delaying successors, 
       but also occupy resources longer. We use a mild duration bonus to break ties).
    
    The final priority vector returned is derived from the *sequence of choices* made by 
    this dynamic simulation. Specifically, we assign higher priority to activities that 
    were scheduled earlier in the successful dynamic simulation. This effectively "locks in" 
    the good dynamic ordering as a static priority vector for the external evaluator.
    """
    start_time = time.time()
    DEADLINE = 105.0  # Safety margin before 120s limit (and 150s kill)
    
    # 1. Extract Data
    durations = np.asarray(instance["durations"], dtype=float)
    predecessors = instance["predecessors"]
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    
    n = len(durations)
    num_resources = len(capacities)
    
    # Build adjacency lists
    successors = [[] for _ in range(n)]
    for j in range(n):
        for i in predecessors[j]:
            successors[i].append(j)
            
    # Topological Sort
    in_degree = [len(predecessors[j]) for j in range(n)]
    queue = deque([j for j in range(n) if in_degree[j] == 0])
    topo_order = []
    in_degree_copy = in_degree[:]
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_degree_copy[v] -= 1
            if in_degree_copy[v] == 0:
                queue.append(v)
                
    if len(topo_order) != n:
        # Cycle or invalid, fallback to duration
        return durations.copy()

    # 2. CPM Calculation
    ES = [0.0] * n
    EF = [0.0] * n
    for j in topo_order:
        if predecessors[j]:
            ES[j] = max(EF[i] for i in predecessors[j])
        else:
            ES[j] = 0.0
        EF[j] = ES[j] + durations[j]
        
    makespan_cpm = max(EF)
    
    LF = [makespan_cpm] * n
    LS = [makespan_cpm] * n
    for j in reversed(topo_order):
        if successors[j]:
            LF[j] = min(LS[k] for k in successors[j])
        else:
            LF[j] = makespan_cpm
        LS[j] = LF[j] - durations[j]
        
    Slack = np.array([LS[j] - ES[j] for j in range(n)])
    
    # Normalize Slack to [0, 1] for combination
    min_slack = np.min(Slack)
    max_slack = np.max(Slack)
    if max_slack > min_slack:
        norm_slack = (Slack - min_slack) / (max_slack - min_slack)
    else:
        norm_slack = np.zeros(n)
        
    # Normalize Duration to [0, 1]
    min_dur = np.min(durations)
    max_dur = np.max(durations)
    if max_dur > min_dur:
        norm_dur = (durations - min_dur) / (max_dur - min_dur)
    else:
        norm_dur = np.zeros(n)

    # 3. Dynamic SGS Simulation
    # We simulate the SGS, but at each step, we compute a dynamic priority for eligible activities.
    # The goal is to find a good schedule. Then we convert that schedule's order into a static priority.
    
    def run_dynamic_sgs():
        """
        Runs SGS with dynamic priority calculation.
        Returns the order in which activities were scheduled (list of activity indices).
        """
        scheduled_count = 0
        scheduled_activities = [] # List of (start, end, demands)
        finish_times = np.full(n, -1.0)
        scheduled_order = []
        
        # Initial eligible
        eligible = deque()
        for j in range(n):
            if len(predecessors[j]) == 0:
                eligible.append(j)
                
        # Track resource usage over time? 
        # For n=30, we can just scan scheduled activities for resource availability.
        
        while scheduled_count < n:
            if not eligible:
                break
                
            # Compute dynamic priority for each eligible activity
            # Dynamic Score = w1 * (1 - NormSlack) + w2 * ResourcePenalty + w3 * NormDur
            # We want HIGH score to be selected.
            # Low Slack is good -> (1 - NormSlack) is high.
            # Low Resource Pressure is good -> We want to penalize high pressure.
            # So ResourcePenalty should be low for good candidates.
            # Let's define ResourcePressure_j = sum over resources r of (demand_jr / current_capacity_r) * (current_usage_r / capacity_r)
            # This penalizes activities that use resources that are already busy.
            
            best_act = None
            best_score = -np.inf
            
            # Current time for resource check? 
            # In SGS, we determine start time *after* selection. 
            # However, dynamic priority often uses the *earliest possible start time* or *current simulation time*.
            # Standard SGS: Select activity -> Find earliest start -> Schedule.
            # To make priority dynamic, we can estimate the "current" resource load at the 
            # *earliest possible start time* for each candidate, or simply at the 
            # *minimum* of all candidate earliest start times.
            
            # Let's calculate earliest start times for all eligible candidates first to establish a "time context".
            # Actually, the standard dynamic heuristic in literature (e.g., "Dynamic Slack") 
            # often re-evaluates slack or uses the current time.
            # Here, we will compute the earliest start time for each candidate *assuming* 
            # we pick it, then evaluate resource contention at that time.
            
            candidates_info = []
            for j in eligible:
                # 1. Predecessor constraint
                pred_finish = 0.0
                for i in predecessors[j]:
                    if finish_times[i] > pred_finish:
                        pred_finish = finish_times[i]
                
                # 2. Resource constraint (Earliest time)
                t = pred_finish
                while True:
                    can_schedule = True
                    for r in range(num_resources):
                        demand = demands[j, r]
                        if demand == 0:
                            continue
                        used = 0.0
                        for (s, e, d) in scheduled_activities:
                            if s <= t < e:
                                used += d[r]
                        if used + demand > capacities[r]:
                            can_schedule = False
                            break
                    if can_schedule:
                        break
                    next_t = np.inf
                    for (s, e, d) in scheduled_activities:
                        if s <= t < e:
                            if e < next_t:
                                next_t = e
                    if next_t == np.inf:
                        t += 1.0
                    else:
                        t = next_t
                
                # Now we have the earliest start time t for j.
                # Calculate Resource Pressure at time t
                pressure = 0.0
                for r in range(num_resources):
                    demand = demands[j, r]
                    if demand == 0:
                        continue
                    used = 0.0
                    for (s, e, d) in scheduled_activities:
                        if s <= t < e:
                            used += d[r]
                    # Pressure is high if used/capacity is high
                    if capacities[r] > 0:
                        pressure += (demand / capacities[r]) * (used / capacities[r])
                    else:
                        pressure += 0
                    
                # Dynamic Score
                # High score = Good
                # 1. Low Slack is good. norm_slack is 0 (low slack) to 1 (high slack).
                # So (1 - norm_slack[j]) is high for low slack.
                # 2. Low Pressure is good. So -pressure.
                # 3. Long duration is slightly good (to get it out of the way). norm_dur[j].
                
                score = 1.0 * (1.0 - norm_slack[j]) - 2.0 * pressure + 0.1 * norm_dur[j]
                
                candidates_info.append((j, score, t))
                
            # Select best
            max_score = -np.inf
            best_j = -1
            best_t = 0.0
            for j, score, t in candidates_info:
                if score > max_score:
                    max_score = score
                    best_j = j
                    best_t = t
            
            if best_j == -1:
                break
                
            # Schedule best_j at best_t
            end_t = best_t + durations[best_j]
            scheduled_activities.append((best_t, end_t, demands[best_j]))
            finish_times[best_j] = end_t
            scheduled_order.append(best_j)
            scheduled_count += 1
            
            # Update eligible
            eligible.remove(best_j)
            for k in successors[best_j]:
                is_ready = True
                for p in predecessors[k]:
                    if finish_times[p] == -1.0:
                        is_ready = False
                        break
                if is_ready:
                    eligible.append(k)
                    
        return scheduled_order

    # Run the dynamic SGS
    # We might run it multiple times with different weights to be robust?
    # Given time constraints, one run with well-tuned weights is safer.
    # Let's try a few variations of the weights to find the best makespan, 
    # then return the priority vector corresponding to the best run.
    
    best_makespan = np.inf
    best_order = None
    
    # Weight variations: (w_slack, w_pressure, w_dur)
    weight_configs = [
        (1.0, 2.0, 0.1), # Balanced
        (1.0, 3.0, 0.1), # Stronger resource penalty
        (1.0, 1.5, 0.1), # Weaker resource penalty
        (1.5, 2.0, 0.1), # Stronger slack
    ]
    
    # We need to modify run_dynamic_sgs to accept weights and return makespan
    def run_dynamic_sgs_with_weights(w_slack, w_pressure, w_dur):
        scheduled_count = 0
        scheduled_activities = []
        finish_times = np.full(n, -1.0)
        scheduled_order = []
        
        eligible = deque()
        for j in range(n):
            if len(predecessors[j]) == 0:
                eligible.append(j)
                
        while scheduled_count < n:
            if not eligible:
                break
                
            candidates_info = []
            for j in eligible:
                pred_finish = 0.0
                for i in predecessors[j]:
                    if finish_times[i] > pred_finish:
                        pred_finish = finish_times[i]
                
                t = pred_finish
                while True:
                    can_schedule = True
                    for r in range(num_resources):
                        demand = demands[j, r]
                        if demand == 0:
                            continue
                        used = 0.0
                        for (s, e, d) in scheduled_activities:
                            if s <= t < e:
                                used += d[r]
                        if used + demand > capacities[r]:
                            can_schedule = False
                            break
                    if can_schedule:
                        break
                    next_t = np.inf
                    for (s, e, d) in scheduled_activities:
                        if s <= t < e:
                            if e < next_t:
                                next_t = e
                    if next_t == np.inf:
                        t += 1.0
                    else:
                        t = next_t
                
                pressure = 0.0
                for r in range(num_resources):
                    demand = demands[j, r]
                    if demand == 0:
                        continue
                    used = 0.0
                    for (s, e, d) in scheduled_activities:
                        if s <= t < e:
                            used += d[r]
                    if capacities[r] > 0:
                        pressure += (demand / capacities[r]) * (used / capacities[r])
                    else:
                        pressure += 0
                    
                score = w_slack * (1.0 - norm_slack[j]) - w_pressure * pressure + w_dur * norm_dur[j]
                candidates_info.append((j, score, t))
                
            max_score = -np.inf
            best_j = -1
            best_t = 0.0
            for j, score, t in candidates_info:
                if score > max_score:
                    max_score = score
                    best_j = j
                    best_t = t
            
            if best_j == -1:
                break
                
            end_t = best_t + durations[best_j]
            scheduled_activities.append((best_t, end_t, demands[best_j]))
            finish_times[best_j] = end_t
            scheduled_order.append(best_j)
            scheduled_count += 1
            
            eligible.remove(best_j)
            for k in successors[best_j]:
                is_ready = True
                for p in predecessors[k]:
                    if finish_times[p] == -1.0:
                        is_ready = False
                        break
                if is_ready:
                    eligible.append(k)
                    
        makespan = max([e for (_, e, _) in scheduled_activities]) if scheduled_activities else 0.0
        return makespan, scheduled_order

    for w_s, w_p, w_d in weight_configs:
        if time.time() - start_time > DEADLINE:
            break
        makespan, order = run_dynamic_sgs_with_weights(w_s, w_p, w_d)
        if makespan < best_makespan:
            best_makespan = makespan
            best_order = order
            
    if best_order is None:
        # Fallback to Min Slack
        return -Slack

    # Construct Priority Vector from Best Order
    # Activities scheduled earlier should have higher priority.
    # Priority = n - index_in_order
    # This creates a strict total order based on the successful simulation.
    priority_vec = np.zeros(n)
    for idx, act in enumerate(best_order):
        priority_vec[act] = n - idx
        
    # If there are any activities not in best_order (shouldn't happen), assign low priority
    for j in range(n):
        if priority_vec[j] == 0.0:
            priority_vec[j] = 0.5 # Low priority
            
    return priority_vec
