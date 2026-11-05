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
    # Or a matrix? Let's assume standard format.
    if "predecessors" in instance:
        preds_data = instance["predecessors"]
        # If it's a list of lists
        if isinstance(preds_data, list) and (not preds_data or isinstance(preds_data[0], (list, tuple, np.ndarray))):
            preds = [list(p) if p else [] for p in preds_data]
        else:
            # Maybe it's a matrix or something else, try to handle gracefully
            try:
                preds_matrix = np.asarray(preds_data)
                if preds_matrix.ndim == 2:
                    preds = [list(np.where(preds_matrix[i] != 0)[0]) for i in range(N)]
                else:
                    preds = [[] for _ in range(N)]
            except:
                preds = [[] for _ in range(N)]
    else:
        preds = [[] for _ in range(N)]
    
    # 1. Compute Critical Path Length (CPL) using backward pass
    # CPL[i] = duration[i] + max(CPL[j] for j in successors of i)
    # We need successors
    successors = [[] for _ in range(N)]
    for i in range(N):
        for p in preds[i]:
            successors[p].append(i)
    
    # Topological order for computation
    in_degree = [0] * N
    for i in range(N):
        for p in preds[i]:
            in_degree[i] += 1
            
    from collections import deque
    queue = deque([i for i in range(N) if in_degree[i] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)
    
    # If topo_order is incomplete, there's a cycle, handle gracefully
    if len(topo_order) < N:
        # Fallback to simple LPT
        return durations.copy()
    
    # Reverse topological order for backward pass
    cpl = np.zeros(N)
    for i in reversed(topo_order):
        if successors[i]:
            max_succ_cpl = max(cpl[s] for s in successors[i])
        else:
            max_succ_cpl = 0.0
        cpl[i] = durations[i] + max_succ_cpl
    
    # 2. Compute Resource Pressure
    # For each activity, compute the ratio of its demand to capacity for each resource
    # and take the maximum (bottleneck resource)
    if demands.ndim == 2:
        resource_pressure = np.max(demands / capacities, axis=1)
    else:
        # If demands is 1D, assume single resource
        resource_pressure = demands / capacities
    
    # 3. Combine scores
    # Normalize each component to [0, 1]
    if np.max(cpl) > 0:
        cpl_norm = cpl / np.max(cpl)
    else:
        cpl_norm = np.zeros(N)
    
    if np.max(durations) > 0:
        dur_norm = durations / np.max(durations)
    else:
        dur_norm = np.zeros(N)
    
    if np.max(resource_pressure) > 0:
        rp_norm = resource_pressure / np.max(resource_pressure)
    else:
        rp_norm = np.zeros(N)
    
    # Weighted combination
    # Higher weight for critical path and duration
    alpha = 0.5  # Weight for CPL
    beta = 0.3   # Weight for Duration
    gamma = 0.2  # Weight for Resource Pressure
    
    scores = alpha * cpl_norm + beta * dur_norm + gamma * rp_norm
    
    # Check time
    if time.time() > deadline:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
