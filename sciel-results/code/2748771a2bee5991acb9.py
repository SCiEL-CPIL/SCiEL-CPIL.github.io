"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

def priority(instance):
    """
    Compute a priority score for each activity in a resource-constrained project scheduling instance.
    
    Strategy: Enhanced Criticality with Path-Weighted Resource Bottleneck.
    Combines inverse total float (criticality) with resource pressure and 
    backward resource load (successor dependency weight) to identify bottlenecks.
    """
    start_time = time.time()
    DEADLINE = 115.0 # Safety margin for 120s limit
    
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    # Get predecessors and build successors
    predecessors = instance.get("predecessors", None)
    if predecessors is None:
        predecessors = [[] for _ in range(n)]
    
    # Ensure predecessors are lists of ints
    preds = []
    for i in range(n):
        p = predecessors[i]
        if p is None:
            preds.append([])
        else:
            preds.append(list(p))
            
    successors = [[] for _ in range(n)]
    for j in range(n):
        for i in preds[j]:
            successors[i].append(j)
    
    # Get resource demands and capacities
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    
    if demands is None:
        demands = np.ones((n, 1))
    else:
        demands = np.asarray(demands, dtype=float)
        if demands.ndim == 1:
            demands = demands.reshape(-1, 1)
    
    if capacities is None:
        capacities = np.ones(demands.shape[1])
    else:
        capacities = np.asarray(capacities, dtype=float)
        if capacities.ndim == 0:
            capacities = np.array([capacities])

    # Handle edge case: no resources or zero capacities
    if demands.shape[1] == 0 or capacities.size == 0 or np.any(capacities == 0):
        # Fallback to duration-based if resources are invalid
        return durations / (np.max(durations) + 1e-9)

    # Topological Sort
    in_degree = np.array([len(preds[i]) for i in range(n)])
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)
                
    # If cycle detected (should not happen in valid RCPSP), fallback
    if len(topo_order) < n:
        topo_order = list(range(n))

    # Forward Pass: Earliest Start (ES)
    ES = np.zeros(n, dtype=float)
    for i in topo_order:
        for pred in preds[i]:
            if ES[pred] + durations[pred] > ES[i]:
                ES[i] = ES[pred] + durations[pred]
        
    # Project End Time (Makespan of CPM)
    makespan_cpm = np.max(ES + durations)
    
    # Backward Pass: Latest Finish (LF)
    LF = np.full(n, makespan_cpm, dtype=float)
    for i in reversed(topo_order):
        for succ in successors[i]:
            if LF[succ] - durations[succ] < LF[i]:
                LF[i] = LF[succ] - durations[succ]
                
    # Total Float (Slack)
    TF = LF - ES - durations
    TF = np.maximum(TF, 0.0)
    
    # Check time budget
    if time.time() - start_time > DEADLINE:
        return 1.0 / (TF + 1e-9)

    # Resource Scarcity Metric
    # Pressure = sum over resources of (demand / capacity)^2
    safe_caps = np.where(capacities > 0, capacities, 1.0)
    ratio = demands / safe_caps
    resource_pressure = np.sum(ratio**2, axis=1)
    
    # Normalize resource pressure
    max_pressure = np.max(resource_pressure)
    if max_pressure > 0:
        resource_pressure = resource_pressure / max_pressure
    else:
        resource_pressure = np.ones(n)

    # Criticality Score (Normalized Inverse Float)
    min_tf = np.min(TF)
    max_tf = np.max(TF)
    if max_tf > min_tf:
        # Map TF [min_tf, max_tf] to [1, 0]
        crit_norm = 1.0 - (TF - min_tf) / (max_tf - min_tf)
    else:
        crit_norm = np.ones(n)

    # Path-Weighted Downstream Work:
    # For each activity, compute:
    #   1. total_downstream_duration: sum of durations of all activities in the transitive closure
    #      (including itself), weighted by criticality (inverse float) of each downstream activity
    #   2. path_diversity: number of distinct "critical paths" through the subgraph below this node
    #      (approximated by counting minimal paths to sinks with low float)
    # Priority signal: total_downstream_duration / (1 + path_diversity)
    # This prioritizes activities that complete large amounts of remaining work
    # while penalizing nodes with many parallel alternatives (less bottleneck-like).
    
    # First, compute for each node the "critical downstream work":
    # For each node j, define its critical weight as (1 / (TF[j] + 1)) * durations[j]
    # Then for each node i, downstream_work[i] = sum over all j in closure(i) of critical_weight[j]
    # But we want to discount by parallelism: divide by (1 + number of independent paths)
    
    # Compute critical weight per node
    crit_weight = durations / (TF + 1.0)  # Activities with zero float get highest weight
    
    # Count number of "critical successor paths" for each node
    # A successor path is "critical" if it goes through activities with low float
    # Approximation: for each node, count how many immediate successors have TF below median
    median_tf = np.median(TF)
    
    # For each node, compute:
    #   path_count[i] = number of immediate successors with TF <= median_tf (bottleneck successors)
    #   If no successors, path_count = 0 (it's a sink)
    path_count = np.zeros(n, dtype=float)
    for i in range(n):
        succs = successors[i]
        if not succs:
            path_count[i] = 0.0
        else:
            # Count successors that are on critical paths (low float)
            low_float_succs = sum(1 for s in succs if TF[s] <= median_tf)
            # At least 1 if there are any successors (they represent alternative paths)
            path_count[i] = max(1, low_float_succs) if low_float_succs > 0 else 1.0
    
    # Compute downstream work in reverse topological order
    # downstream_work[i] = crit_weight[i] + sum over successors s of (downstream_work[s] / path_count[i])
    # The division by path_count accounts for parallelism: if there are k parallel paths,
    # each contributes 1/k of the downstream work to this node's priority
    
    downstream_work = np.zeros(n, dtype=float)
    for i in reversed(topo_order):
        own_work = crit_weight[i]
        succ_work = 0.0
        for s in successors[i]:
            succ_work += downstream_work[s]
        # Divide by (1 + path_count) to account for parallel alternatives
        # path_count >= 1 if there are successors, so divisor is at least 2
        if successors[i]:
            downstream_work[i] = own_work + succ_work / (1.0 + path_count[i])
        else:
            downstream_work[i] = own_work
    
    # Normalize downstream work
    max_dw = np.max(downstream_work)
    if max_dw > 0:
        bwd_load_norm = downstream_work / max_dw
    else:
        bwd_load_norm = np.ones(n)
    
    # Combine:
    # Base: Criticality (from Insp 1 which was better)
    # Boost by Resource Pressure and Path-Weighted Downstream Work
    # Priority = CritNorm * (1 + w1 * ResourcePressure + w2 * BwdLoadNorm)
    
    # Weights: Resource pressure is immediate constraint. Downstream work is structural importance.
    # Since RP and BLD are 0..1, (1 + 0.5*RP + 0.5*BLD) ranges from 1.0 to 2.0.
    
    priority_scores = crit_norm * (1.0 + 0.5 * resource_pressure + 0.5 * bwd_load_norm)
    
    # Final check on time
    if time.time() - start_time > DEADLINE:
         return priority_scores

    # Ensure no NaN or Inf values
    priority_scores = np.nan_to_num(priority_scores, nan=0.0, posinf=1e6, neginf=0.0)
    
    return priority_scores
# EVOLVE-BLOCK-END
