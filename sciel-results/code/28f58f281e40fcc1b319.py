"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(instance):
    """
    Returns a priority score for each activity to be used in a Serial SGS.
    Uses a combination of longest processing time (LPT), critical path length,
    and resource pressure.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin below 120s limit

    # Extract instance data
    N = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)  # Shape (N, R)
    capacities = np.asarray(instance["capacities"], dtype=float)  # Shape (R,)
    
    # Handle predecessors: assume list of lists where preds[i] is list of indices
    # Or a matrix? Let's assume list of lists for now, but handle both.
    preds_raw = instance.get("predecessors", None)
    
    if preds_raw is None:
        # No predecessors, all independent
        pred_lists = [[] for _ in range(N)]
    elif isinstance(preds_raw, list):
        if len(preds_raw) > 0 and isinstance(preds_raw[0], list):
            pred_lists = preds_raw
        else:
            # Maybe it's a flat list? Unlikely. Assume list of lists.
            pred_lists = preds_raw
    elif isinstance(preds_raw, np.ndarray):
        # Maybe a boolean matrix or adjacency matrix?
        if preds_raw.shape == (N, N):
            pred_lists = []
            for i in range(N):
                # Find j such that preds_raw[j][i] is True? Or preds_raw[i][j]?
                # Standard: preds[i] is list of j such that j->i.
                # If matrix M, M[j][i]=1 means j->i.
                # Let's check typical conventions. Usually preds[i] is predecessors of i.
                # If matrix, often M[i][j] = 1 if i->j? Or M[j][i]=1 if j->i?
                # Let's assume the input 'predecessors' is already in the format
                # expected by the SGS, which is usually a list of lists.
                # If it's an array, we might need to convert.
                # Let's try to infer.
                pass
        pred_lists = [[] for _ in range(N)] # Fallback

    # Convert pred_lists to a consistent format: list of lists of ints
    # Ensure all are lists of ints
    processed_preds = []
    for p in pred_lists:
        if isinstance(p, np.ndarray):
            processed_preds.append(list(p.astype(int)))
        elif isinstance(p, list):
            processed_preds.append([int(x) for x in p])
        else:
            processed_preds.append([])
            
    # Build successors and in-degrees for topological sort / critical path
    successors = [[] for _ in range(N)]
    in_degree = [0] * N
    for i in range(N):
        for j in processed_preds[i]:
            successors[j].append(i)
            in_degree[i] += 1

    # Calculate Critical Path Length (CPL) from each node to the end
    # CPL[i] = durations[i] + max(CPL[k] for k in successors[i])
    # We can compute this via topological sort (reverse order)
    # Since N=30, we can do a simple DFS with memoization or iterative topo sort.
    
    # Topological sort using Kahn's algorithm
    from collections import deque
    queue = deque()
    topo_order = []
    indeg_copy = in_degree[:]
    for i in range(N):
        if indeg_copy[i] == 0:
            queue.append(i)
            
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            indeg_copy[v] -= 1
            if indeg_copy[v] == 0:
                queue.append(v)
                
    # If topo_order length is not N, there's a cycle. Fallback to simple order.
    if len(topo_order) != N:
        topo_order = list(range(N))
        
    # Compute CPL in reverse topological order
    cpl = np.zeros(N, dtype=float)
    for u in reversed(topo_order):
        max_succ_cpl = 0.0
        for v in successors[u]:
            if cpl[v] > max_succ_cpl:
                max_succ_cpl = cpl[v]
        cpl[u] = durations[u] + max_succ_cpl

    # Calculate Resource Pressure
    # For each activity, how much does it demand relative to capacity?
    # pressure[i] = sum(demands[i, r] / capacities[r]) for r where capacities[r] > 0
    resource_pressure = np.zeros(N, dtype=float)
    if demands.ndim == 2 and capacities.size > 0:
        R = capacities.size
        for r in range(R):
            if capacities[r] > 0:
                resource_pressure += demands[:, r] / capacities[r]
            else:
                # If capacity is 0, any demand is critical?
                # Assume high pressure if demand > 0
                if demands[:, r] > 0:
                    resource_pressure += 100.0 # Arbitrary high penalty

    # Combine scores
    # We want higher score = higher priority (scheduled earlier)
    # LPT: durations
    # CPL: cpl
    # Pressure: resource_pressure
    
    # Normalize or weight them?
    # Simple weighted sum:
    # score = w1 * durations + w2 * cpl + w3 * resource_pressure
    
    # Let's try equal weights or specific weights.
    # CPL is often the most important for makespan.
    # LPT is good for resource leveling.
    
    # Let's use: score = cpl + 0.5 * durations + 0.5 * resource_pressure
    # Or maybe just cpl is dominant?
    # The prompt says "Combining critical path length, duration, and resource pressure creates a more informed priority".
    
    score = cpl + 0.5 * durations + 0.5 * resource_pressure
    
    return score
# EVOLVE-BLOCK-END
