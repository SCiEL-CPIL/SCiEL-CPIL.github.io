import numpy as np
import time
from collections import deque
import math

def priority(instance):
    """
    Score each activity for the serial schedule-generation scheme.
    Higher priority (score) is scheduled first among eligible activities.
    
    Strategy:
    1. Compute Critical Path Length (CPL) from each activity to the project end.
    2. Compute Earliest Start (ES) times.
    3. Compute Total Slack (Float) for each activity.
    4. Priority Score = CPL / (1 + Slack).
       - High CPL indicates the activity is part of a long path.
       - Low Slack indicates the activity is critical (on or near the critical path).
       - Dividing by (1+Slack) amplifies the priority of critical activities 
         while maintaining the influence of path length.
    """
    start_time = time.time()
    DEADLINE = 140.0

    n = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    preds = instance["predecessors"]
    
    # Build successor list and in-degrees for topological sort
    successors = [[] for _ in range(n)]
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for p in preds[i]:
            successors[p].append(i)
            in_degree[i] += 1
    
    # Topological sort (Kahn's algorithm)
    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)
    
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
        
    if len(topo_order) != n:
        # Fallback if graph is cyclic (should not happen in RCPSP)
        return durations.copy()
    
    # Compute CPL (Critical Path Length) in reverse topological order
    cpl = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        if not successors[u]:
            cpl[u] = durations[u]
        else:
            # CPL is duration + max(CPL of successors)
            max_succ_cpl = 0.0
            for v in successors[u]:
                if cpl[v] > max_succ_cpl:
                    max_succ_cpl = cpl[v]
            cpl[u] = durations[u] + max_succ_cpl

    # Check deadline
    if time.time() - start_time > DEADLINE:
        return cpl.copy()

    # Compute Earliest Start (ES) times in topological order
    es = np.zeros(n, dtype=float)
    for u in topo_order:
        if not preds[u]:
            es[u] = 0.0
        else:
            max_pred_ef = 0.0
            for p in preds[u]:
                ef = es[p] + durations[p]
                if ef > max_pred_ef:
                    max_pred_ef = ef
            es[u] = max_pred_ef
    
    # Project length (CPM makespan)
    project_length = 0.0
    for i in range(n):
        ef = es[i] + durations[i]
        if ef > project_length:
            project_length = ef
    
    # Compute Slack
    # Slack = (ES + Duration + CPL) - Project_Length
    # Note: CPL here is length from i to end.
    # Path length through i = ES[i] + Dur[i] + (CPL[i] - Dur[i])? 
    # No, CPL[i] defined above includes Dur[i].
    # So path length = ES[i] + CPL[i].
    # Wait, CPL[i] = Dur[i] + max(CPL[succ]).
    # So ES[i] + CPL[i] is the length of the longest path starting at i and going to end.
    # The total path length through i is ES[i] + CPL[i].
    # Slack[i] = (ES[i] + CPL[i]) - project_length.
    
    slack = np.zeros(n, dtype=float)
    for i in range(n):
        slack[i] = (es[i] + cpl[i]) - project_length
        
    # Ensure slack is non-negative for safety
    slack = np.maximum(slack, 0.0)

    # Check deadline
    if time.time() - start_time > DEADLINE:
        return cpl.copy()

    # Compute resource contention component using log-sum-exp
    # For each activity, measure the "resource tail": the maximum resource
    # utilization pressure it imposes on downstream scheduling.
    # Use a soft-max surrogate to capture aggregate resource pressure.
    
    # For each resource, compute the sum of demands of activities that share
    # a critical path relationship with activity i.
    # Simplified: for each activity i, compute the total demand pressure
    # of activities that could overlap with i given the schedule.
    
    # Compute resource pressure: for each activity, sum demands weighted by
    # proximity to the critical path
    resource_pressure = np.zeros(n, dtype=float)
    for i in range(n):
        for r in range(len(capacities)):
            if capacities[r] > 0:
                # Sum of demands for resource r of activities that are
                # "close" to i in the precedence graph
                # Use a simple heuristic: activities with similar ES times
                # or that are successors/predecessors
                pressure = 0.0
                for j in range(n):
                    if i != j:
                        # Weight by similarity of ES and resource demand
                        es_diff = abs(es[i] - es[j])
                        # Activities closer in time are more likely to conflict
                        weight = math.exp(-es_diff / max(project_length, 1.0))
                        pressure += demands[j, r] * weight
                resource_pressure[i] += pressure / capacities[r]
    
    # Normalize resource pressure to [0, 1]
    max_rp = np.max(resource_pressure) if np.max(resource_pressure) > 0 else 1.0
    resource_pressure_norm = resource_pressure / max_rp
    
    # Combine CPL/slack with resource pressure
    # Higher resource pressure should increase priority to avoid bottlenecks
    # Use a weighted combination
    base_score = cpl / (1.0 + slack)
    
    # Scale base_score to comparable range
    max_base = np.max(base_score) if np.max(base_score) > 0 else 1.0
    base_score_norm = base_score / max_base
    
    # Final score: weighted combination emphasizing resource contention
    # Activities with high resource pressure get boosted priority
    scores = base_score_norm + 0.3 * resource_pressure_norm
    
    return scores
