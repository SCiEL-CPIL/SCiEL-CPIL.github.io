"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque
import time

def priority(instance):
    """
    Greedy Lookahead (Min-Conflict) Priority Heuristic.
    
    Strategy:
    1. Compute static Bottom Level (BL) and Total Successor Work (TSW) for each activity.
    2. Simulate the serial schedule generation scheme (SGS) in a "dry run" to find the 
       earliest possible start time for each activity if it were scheduled immediately.
    3. For each activity i, calculate a 'Regret' score:
       - Estimate the delay imposed on its successors if i starts at its earliest possible time.
       - The regret is weighted by the criticality (BL) of the successors.
    4. Priority = Regret + Base_Priority.
       - Higher Regret means the activity is more "urgent" to schedule to prevent blocking critical successors.
       - Base_Priority ensures that critical path activities are preferred even if they don't have immediate successors.
    """
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin
    
    n = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    predecessors = instance.get("predecessors", [[] for _ in range(n)])
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    
    if demands is None or capacities is None:
        return _bottom_level(n, durations, predecessors)
    
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    n_resources = len(capacities)
    
    # Build graph structures
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)
            in_degree[i] += 1
    
    # Topological sort
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)
                
    if len(topo_order) != n:
        return durations.copy()

    # Static Metrics
    # Bottom Level (Longest path to end)
    bl = np.zeros(n)
    for node in reversed(topo_order):
        if not successors[node]:
            bl[node] = durations[node]
        else:
            bl[node] = durations[node] + max(bl[s] for s in successors[node])
            
    # Total Successor Work (Sum of durations of all successors, direct and indirect)
    # This helps identify activities that "block" a lot of work
    tsw = np.zeros(n)
    for node in reversed(topo_order):
        tsw[node] = durations[node]
        for s in successors[node]:
            tsw[node] += tsw[s]

    # Simulate SGS to find "Earliest Possible Start" (EPS) for each activity
    # We assume a greedy schedule where we pick the highest BL activity first.
    # This gives us a reference schedule to evaluate "regret".
    scheduled = [False] * n
    start_times = [0.0] * n
    # Resource usage: list of (end_time, demand) for each resource
    resource_usage = [[] for _ in range(n_resources)]
    
    # We need to simulate the actual scheduling process to know the context.
    # However, the priority function is called once. We must return static scores.
    # The "Lookahead" here is: 
    # If I schedule i now, at time T_i, how much do I delay my successors?
    # Since we don't know T_i exactly without running the full SGS, we estimate T_i 
    # using the resource availability at the earliest precedence start time.
    
    # Let's compute a "Resource Constraint Score" for each activity.
    # For each activity i, check if it can start at its precedence EST.
    # If not, how much delay?
    
    est_prec = np.zeros(n)
    for node in topo_order:
        if predecessors[node]:
            est_prec[node] = max(est_prec[p] + durations[p] for p in predecessors[node])
        else:
            est_prec[node] = 0.0

    # We will perform a light simulation to estimate the "actual" start time if we followed a BL-based schedule.
    # This simulation is O(n^2 * R). n=30, R=4. Very fast.
    sim_scheduled = [False] * n
    sim_starts = [0.0] * n
    sim_usage = [[] for _ in range(n_resources)]
    
    # Sort by BL descending for the simulation
    sim_order = sorted(range(n), key=lambda x: -bl[x])
    
    # We need to respect precedence in the simulation too.
    # Standard SGS:
    # 1. Maintain set of eligible activities (all predecessors scheduled).
    # 2. Pick highest priority (BL) from eligible.
    # 3. Schedule it at earliest possible time >= max(pred_end) and resource available.
    
    eligible = set(i for i in range(n) if in_degree[i] == 0)
    # Re-compute in_degree for simulation
    sim_in_deg = in_degree[:]
    
    while eligible:
        # Pick highest BL from eligible
        current = max(eligible, key=lambda x: bl[x])
        
        # Determine start time
        earliest_prec = 0.0
        if predecessors[current]:
            earliest_prec = max(sim_starts[p] + durations[p] for p in predecessors[current])
            
        # Check resources
        earliest_res = 0.0
        for r in range(n_resources):
            d = demands[current][r]
            if d > 0:
                # Find earliest time t >= earliest_prec where resource r is free
                # We have a list of (end_time, demand) intervals.
                # Since we schedule sequentially, we can just check the latest end time of overlapping jobs?
                # No, we need to find a gap.
                # Simple approach: iterate through sorted intervals.
                # But n is small.
                
                # Current usage is a list of [start, end, demand]? No, we just added them.
                # Let's keep it simple: for each resource, maintain a list of intervals.
                # To find start time:
                # Start with t = earliest_prec.
                # While there is a conflict at t, move t to the end of the conflicting interval.
                
                t = earliest_prec
                while True:
                    conflict_end = -1
                    for interval in sim_usage[r]:
                        int_start, int_end, int_demand = interval
                        # If intervals overlap: int_start < t + dur AND int_end > t
                        # Actually, we just need to know if resource is busy at t.
                        # Since we are checking if we can start at t:
                        # If int_start <= t < int_end, there is a conflict.
                        # The new t should be at least int_end.
                        if int_start <= t < int_end:
                            if int_end > conflict_end:
                                conflict_end = int_end
                    if conflict_end == -1:
                        break
                    t = conflict_end
                    # If t < earliest_prec, something is wrong, but t only increases.
                
                if t > earliest_res:
                    earliest_res = t
        
        actual_start = max(earliest_prec, earliest_res)
        sim_starts[current] = actual_start
        
        # Update resource usage
        for r in range(n_resources):
            d = demands[current][r]
            if d > 0:
                sim_usage[r].append((actual_start, actual_start + durations[current], d))
        
        sim_scheduled[current] = True
        
        # Update eligible set
        eligible.remove(current)
        for s in successors[current]:
            sim_in_deg[s] -= 1
            if sim_in_deg[s] == 0:
                eligible.add(s)
        
        if time.time() - start_time > DEADLINE:
            break

    # Now, for each activity i, calculate Regret.
    # Regret[i] = sum over successors s of: 
    #   max(0, (sim_starts[i] + durations[i]) - est_prec[s]) * bl[s]
    # This measures how much i delays the start of its successors relative to their precedence EST.
    # If sim_starts[i] is much later than est_prec[i], it might delay successors.
    # Actually, a better metric: 
    # If we schedule i, it finishes at F_i = sim_starts[i] + durations[i].
    # Successor s can start at max(F_i, est_prec[s]).
    # Delay imposed on s = max(0, F_i - est_prec[s]).
    # Weighted by bl[s].
    
    regret = np.zeros(n)
    for i in range(n):
        finish_i = sim_starts[i] + durations[i]
        for s in successors[i]:
            delay = max(0.0, finish_i - est_prec[s])
            regret[i] += delay * bl[s]
            
    # Normalize regret
    max_regret = np.max(regret) if n > 0 else 1.0
    if max_regret > 0:
        regret_norm = regret / max_regret
    else:
        regret_norm = regret

    # Combine Regret and Bottom Level
    # High Regret = High Priority (Schedule it to relieve congestion)
    # High BL = High Priority (It's on the critical path)
    
    # Let's try: Priority = 0.5 * BL_norm + 0.5 * Regret_norm
    # Or maybe Regret is more important for resource constrained problems.
    
    bl_norm = bl / np.max(bl) if np.max(bl) > 0 else bl
    
    priority_scores = 0.4 * bl_norm + 0.6 * regret_norm
    
    return priority_scores

def _bottom_level(n, durations, predecessors):
    successors = [[] for _ in range(n)]
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)
    
    in_degree = [0] * n
    for i in range(n):
        for pred in predecessors[i]:
            in_degree[i] += 1
    
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)
    
    if len(topo_order) != n:
        return durations.copy()
    
    bottom_level = np.zeros(n)
    for node in reversed(topo_order):
        if not successors[node]:
            bottom_level[node] = durations[node]
        else:
            bottom_level[node] = durations[node] + max(bottom_level[s] for s in successors[node])
    
    return bottom_level
# EVOLVE-BLOCK-END
