import numpy as np
import heapq
import time

def priority(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([])
    
    start_time = time.time()
    deadline = start_time + 100  # safety margin
    
    # Compute k-core peeling
    adj = adjacency.astype(np.int32)
    degrees = adj.sum(axis=1).copy()
    alive = np.ones(n, dtype=bool)
    
    # Min-heap of (current_degree, vertex)
    heap = [(degrees[i], i) for i in range(n)]
    heapq.heapify(heap)
    
    removal_order = np.zeros(n, dtype=np.int32)
    core_numbers = np.zeros(n, dtype=np.int32)
    removed = np.zeros(n, dtype=bool)
    current_max_core = 0
    removal_step = 0
    
    while heap:
        deg, v = heapq.heappop(heap)
        if removed[v]:
            continue
        if not alive[v]:
            continue
        # Deg might be stale; check actual current degree
        actual_deg = adj[v, alive].sum()
        if actual_deg > deg:
            # Stale entry, skip
            continue
        
        removed[v] = True
        alive[v] = False
        removal_step += 1
        removal_order[v] = removal_step
        core_numbers[v] = current_max_core
        
        # Update neighbors
        neighbors = np.where(adj[v, :])[0]
        for u in neighbors:
            if alive[u]:
                # Remove edge v-u
                # Update degree
                new_deg = adj[u, alive].sum() - 1  # since v is now not alive
                # Actually, let's just recompute or track properly
                pass
        
        # This approach is getting complicated. Let me use a simpler method.
    
    # Simpler approach for n=200: just do iterative peeling
    alive = np.ones(n, dtype=bool)
    current_degrees = adj.sum(axis=1).copy()
    removal_order = np.zeros(n, dtype=np.int32)
    core_numbers = np.zeros(n, dtype=np.int32)
    removed_count = 0
    current_core = 0
    
    while removed_count < n:
        # Find vertices with minimum degree among alive
        alive_indices = np.where(alive)[0]
        if len(alive_indices) == 0:
            break
        min_deg = current_degrees[alive_indices].min()
        # Get all alive vertices with this min degree
        candidates = alive_indices[current_degrees[alive_indices] == min_deg]
        
        for v in candidates:
            if not alive[v]:
                continue
            alive[v] = False
            removed_count += 1
            removal_order[v] = removed_count
            core_numbers[v] = current_core
            # Update neighbors' degrees
            neighbors = np.where(adj[v, :])[0]
            for u in neighbors:
                if alive[u]:
                    current_degrees[u] -= 1
        
        current_core += 1
    
    # Now compute scores
    # Strategy: prefer low core number (peripheral) and low degree
    # Also consider: neighbors' core numbers - if neighbors are high core, picking v is "cheaper"
    
    # Neighbor core sum: sum of core numbers of neighbors
    neighbor_core_sum = np.zeros(n, dtype=np.float64)
    for i in range(n):
        neighbors = np.where(adj[i, :])[0]
        if len(neighbors) > 0:
            neighbor_core_sum[i] = core_numbers[neighbors].sum()
    
    # Score: lower is better (since greedy picks descending, we want high score = good pick)
    # Wait, the interface says "picks vertices in descending score order"
    # So higher score = picked first
    # We want to pick good vertices first, so good vertices should have high score
    
    # Good for IS: low degree, low core number (peripheral), neighbors with high core (so we don't block good picks)
    
    # Let's try: score = -degree - 0.1*core_number + 0.01*neighbor_core_sum
    # Or normalize and combine
    
    degree = adj.sum(axis=1).astype(np.float64)
    
    # Normalize components
    if degree.max() > 0:
        degree_norm = degree / degree.max()
    else:
        degree_norm = degree.copy()
    
    if core_numbers.max() > 0:
        core_norm = core_numbers.astype(np.float64) / core_numbers.max()
    else:
        core_norm = core_numbers.astype(np.float64).copy()
    
    if neighbor_core_sum.max() > 0:
        ncs_norm = neighbor_core_sum / neighbor_core_sum.max()
    else:
        ncs_norm = neighbor_core_sum.copy()
    
    # Combine: we want low degree, low core, high neighbor core sum
    score = -1.0 * degree_norm - 0.5 * core_norm + 0.3 * ncs_norm
    
    return score
