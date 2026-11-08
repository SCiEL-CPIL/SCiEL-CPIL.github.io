import numpy as np
from collections import deque
import time

def priority(instance):
    """
    Calculates priority scores using a Resource-Adjusted Critical Path heuristic.
    
    Strategy:
    1. Compute static CPM Slack (ES/LS) as a baseline.
    2. Simulate the Serial Schedule Generation Scheme (SSGS) using the static Slack 
       as the priority rule to generate a feasible schedule.
    3. Analyze the simulated schedule to identify "bottleneck" activities.
       An activity is a bottleneck if:
       a) It has low static slack (critical path).
       b) It consumes a high fraction of a resource's capacity.
       c) It is scheduled close to the end of the makespan (high "reverse" slack in sim).
    4. Return scores that prioritize activities with high "bottleneck potential".
       Higher score = higher priority.
    """
    start_time = time.time()
    durations = np.asarray(instance["durations"], dtype=float)
    predecessors = instance["predecessors"]
    demands = instance["demands"]  # Shape (n, m)
    capacities = instance["capacities"] # Shape (m,)
    
    n = len(durations)
    m = len(capacities)
    
    # Build successor list
    successors = [[] for _ in range(n)]
    for j in range(n):
        for i in predecessors[j]:
            successors[i].append(j)
            
    # Topological Sort
    in_degree = [len(predecessors[j]) for j in range(n)]
    queue = deque([j for j in range(n) if in_degree[j] == 0])
    topo_order = []
    
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    if len(topo_order) != n:
        # Fallback for cycles (should not happen)
        return durations.copy()

    # 1. Static CPM Calculation
    ES = [0.0] * n
    EF = [0.0] * n
    for j in topo_order:
        if predecessors[j]:
            ES[j] = max(EF[i] for i in predecessors[j])
        else:
            ES[j] = 0.0
        EF[j] = ES[j] + durations[j]
        
    makespan_cpm = max(EF) if n > 0 else 0.0
    
    LF = [makespan_cpm] * n
    LS = [makespan_cpm] * n
    for j in reversed(topo_order):
        if successors[j]:
            LF[j] = min(LS[k] for k in successors[k] if k != j) # Wait, LS[k] is latest start of successor
            # Correction: LF[j] = min(LS[s] for s in successors[j])
            if successors[j]:
                LF[j] = min(LS[k] for k in successors[j])
            else:
                LF[j] = makespan_cpm
        else:
            LF[j] = makespan_cpm
        LS[j] = LF[j] - durations[j]
        
    static_slack = [LS[j] - ES[j] for j in range(n)]
    
    # 2. Simulate SSGS using Min-Static-Slack as priority
    # We need to simulate the exact process the evaluator will use.
    # The evaluator starts the highest priority eligible activity.
    # Here we simulate that to see which activities actually block others.
    
    # For simulation, we need a priority function. We use -static_slack.
    # This is a recursive definition if we want to use the new score, but we can't.
    # So we simulate with the *current* known good heuristic (Min Slack) to identify
    # which activities are "critical" in the resource-constrained sense.
    
    sim_start = [0.0] * n
    sim_finish = [0.0] * n
    remaining = set(range(n))
    scheduled = set()
    current_time = 0.0
    resource_usage = [0.0] * m # Usage at current_time? No, we need to track intervals.
    
    # A simpler simulation:
    # Maintain a list of (finish_time, activity_id) to know when resources are freed.
    # And a set of eligible activities.
    
    eligible = []
    for j in range(n):
        if not predecessors[j]:
            eligible.append(j)
            
    # We need a priority queue for eligible activities.
    # Priority: -static_slack[j] (higher is better).
    # We'll use a simple list and sort, or heapq. n=30 is small.
    
    while eligible:
        # Check deadline
        if time.time() - start_time > 140:
             # Fallback to static slack if simulation takes too long (unlikely for n=30)
             return np.array([-s for s in static_slack], dtype=float)

        # Find highest priority eligible activity
        # Eligible means all predecessors are scheduled.
        # In this loop, 'eligible' list contains nodes whose preds are scheduled.
        
        best_idx = -1
        best_score = -float('inf')
        for idx, act in enumerate(eligible):
            score = -static_slack[act]
            if score > best_score:
                best_score = score
                best_idx = idx
        
        if best_idx == -1:
            break
            
        act = eligible.pop(best_idx)
        
        # Determine earliest start time for 'act'
        # 1. Precedence constraint: max finish time of predecessors
        es_prec = 0.0
        for pred in predecessors[act]:
            es_prec = max(es_prec, sim_finish[pred])
            
        # 2. Resource constraint
        # We need to find the earliest time t >= es_prec where resources are available.
        # Since n is small, we can just scan forward.
        # However, we need to know resource availability over time.
        # Let's maintain a list of events: (time, activity_id, is_finish)
        # Actually, simpler: just check against all scheduled activities.
        
        t = es_prec
        while True:
            # Check if 'act' can start at t
            can_start = True
            for r in range(m):
                if demands[act][r] > 0:
                    # Sum demands of all scheduled activities that overlap with [t, t+dur]
                    current_load = 0.0
                    for k in scheduled:
                        if sim_start[k] < t + durations[act] and sim_finish[k] > t:
                            current_load += demands[k][r]
                    if current_load + demands[act][r] > capacities[r]:
                        can_start = False
                        break
            if can_start:
                break
            t += 1.0 # Integer time steps? Durations might be float.
            # If durations are float, this loop is risky.
            # Better approach: collect all relevant event times (start/finish of scheduled acts).
            # For n=30, O(n^2) scan is fine, but let's be safe with float durations.
            # Let's just jump to the next event time.
            
            # Find the earliest finish time of any overlapping activity that blocks us.
            next_event = float('inf')
            for k in scheduled:
                if sim_start[k] < t + durations[act] and sim_finish[k] > t:
                    if sim_finish[k] > t:
                        next_event = min(next_event, sim_finish[k])
            if next_event == float('inf'):
                 # Should not happen if can_start was false
                 break
            t = next_event
            
        sim_start[act] = t
        sim_finish[act] = t + durations[act]
        scheduled.add(act)
        
        # Add newly eligible successors
        for succ in successors[act]:
            if succ not in scheduled and succ not in eligible:
                # Check if all preds of succ are scheduled
                all_preds_done = True
                for p in predecessors[succ]:
                    if p not in scheduled:
                        all_preds_done = False
                        break
                if all_preds_done:
                    eligible.append(succ)

    sim_makespan = max(sim_finish) if n > 0 else 0.0
    
    # 3. Calculate Resource-Adjusted Priority
    # We want to prioritize activities that:
    # a) Have low static slack (critical path)
    # b) Are "late" in the simulation (high reverse slack in sim) -> These are on the critical path of the resource-constrained schedule.
    # c) Consume significant resources relative to capacity.
    
    # Calculate "Sim Slack" (Reverse Slack):
    # How much can an activity be delayed in the simulation without increasing makespan?
    # This is hard to compute exactly without backward pass on the resource-constrained graph.
    # Approximation: 
    # Priority = -static_slack + alpha * (sim_finish / sim_makespan) + beta * resource_pressure
    
    # Let's use a simpler, robust metric:
    # Score = -static_slack[j] + 0.5 * (sim_finish[j] / max(sim_makespan, 1.0))
    # This boosts activities that finish late (likely critical) and have low static slack.
    
    # Also, consider resource pressure:
    # pressure[j] = sum(demands[j][r] / capacities[r]) for r where cap > 0
    
    scores = np.zeros(n)
    for j in range(n):
        base = -static_slack[j]
        
        # Normalized finish time in simulation (0 to 1)
        norm_finish = sim_finish[j] / sim_makespan if sim_makespan > 0 else 0.0
        
        # Resource pressure
        pressure = 0.0
        for r in range(m):
            if capacities[r] > 0:
                pressure += demands[j][r] / capacities[r]
        
        # Combine:
        # Higher priority for: Low static slack, Late finish, High resource pressure.
        # Weights: Static slack is most important. Late finish indicates actual criticality.
        scores[j] = base + 0.5 * norm_finish + 0.1 * pressure
        
    return scores

