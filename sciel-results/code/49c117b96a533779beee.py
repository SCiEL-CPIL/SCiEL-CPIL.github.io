"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict, deque

def priority(instance):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit
    
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
    else:
        # If no predecessors provided, assume no dependencies
        pass

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

    # 2. Topological Sort & Path Calculations
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
        # Handle cycles (shouldn't happen in valid RCPSP, but safe guard)
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
    
    # Longest Path from node to end (Critical Path Length)
    longest_path = np.zeros(n)
    for node in reversed(topo_order):
        if succ_list[node]:
            longest_path[node] = durations[node] + max(longest_path[s] for s in succ_list[node])
        else:
            longest_path[node] = durations[node]

    # 3. Calculate Resource Bottleneck Centrality
    # Strategy: 
    # "Flow" perspective: An activity's priority is determined by how much "bottlenecked flow" it gates.
    # Bottlenecked flow = Downstream demand on scarce resources.
    
    # Step A: Calculate downstream demand aggregation
    # down_demand[i] is a vector of size n_resources representing the total demand
    # of all activities in the subgraph rooted at i (including i).
    # Actually, we want the demand of the *dependents* to see what i is gating.
    # Let's compute "Total Downstream Demand" for each activity.
    
    # We traverse in reverse topological order.
    # downstream_demand[i] = demand[i] + sum(downstream_demand[s] for s in succ_list[i])
    # This gives the total resource consumption of the entire subtree.
    
    downstream_demand = np.zeros((n, n_resources))
    for node in reversed(topo_order):
        downstream_demand[node] = demand_matrix[node].copy()
        for s in succ_list[node]:
            downstream_demand[node] += downstream_demand[s]
            
    # Step B: Calculate Resource Scarcity Weight
    # A resource is "scarce" if capacity is low relative to total demand or individual demands.
    # Here we use inverse capacity as a proxy for bottleneck severity.
    # inv_cap[j] = 1 / cap[j]
    
    if n_resources > 0 and np.any(cap_vector > 0):
        inv_cap = np.zeros(n_resources)
        for j in range(n_resources):
            if cap_vector[j] > 0:
                inv_cap[j] = 1.0 / cap_vector[j]
            else:
                inv_cap[j] = 0.0 # Avoid division by zero, treat as no bottleneck
    else:
        inv_cap = np.zeros(n_resources)
        
    # Step C: Calculate Bottleneck Centrality Score
    # For each activity i, score is the dot product of its downstream demand and inverse capacities.
    # This highlights activities that unlock large amounts of demand on tight resources.
    
    bottleneck_centrality = np.zeros(n)
    if n_resources > 0:
        # Vectorized dot product: downstream_demand @ inv_cap
        bottleneck_centrality = downstream_demand @ inv_cap
        
    # 4. Combine Signals
    # We want to prioritize:
    # 1. High Bottleneck Centrality (New Structural Signal)
    # 2. High Longest Path (Critical Path)
    # 3. Low Float (Tightness)
    
    def normalize(arr):
        mn = np.min(arr)
        mx = np.max(arr)
        if mx > mn:
            return (arr - mn) / (mx - mn)
        return np.zeros_like(arr)
        
    norm_bottleneck = normalize(bottleneck_centrality)
    norm_longest_path = normalize(longest_path)
    norm_float_inv = 1.0 - normalize(total_float)
    norm_duration = normalize(durations)
    
    # Weights:
    # The new strategy bets on Bottleneck Centrality being a stronger discriminator in resource-constrained environments.
    # We give it significant weight, but keep CP as a strong secondary signal.
    
    base_priority = (
        0.40 * norm_bottleneck +
        0.30 * norm_longest_path +
        0.20 * norm_float_inv +
        0.10 * norm_duration
    )
    
    # 5. Final Adjustments
    # Boost activities that are both on the critical path (low float) AND have high bottleneck centrality
    # This targets the "gates" of the critical path.
    
    adjusted_priority = base_priority.copy()
    
    max_float = np.max(total_float) if n > 0 else 0
    if max_float > 0:
        critical_mask = total_float < (0.1 * max_float)
        # Additive boost for critical bottlenecks
        boost = 0.15 * norm_bottleneck * critical_mask
        adjusted_priority += boost

    return adjusted_priority
# EVOLVE-BLOCK-END
