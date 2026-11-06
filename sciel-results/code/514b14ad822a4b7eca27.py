import random
import time
import math

def construct():
    start_time = time.time()
    time_limit = 115.0
    
    n = 16
    full_mask = (1 << 28) - 1
    
    # 1. Precompute Q3 masks in K8
    # Q3 edges in 0..7:
    # 0-1, 0-2, 0-4, 1-3, 1-5, 2-3, 2-6, 3-7, 4-5, 4-6, 5-7, 6-7
    q3_edges_local = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    
    # Map local edge (u, v) to bit index in 28-bit mask
    # u < v, bit = u*7 - u*(u-1)//2 + v - u - 1 ?
    # Simpler:
    # bit_index = 0
    # for i in range(8):
    #   for j in range(i+1, 8):
    #     bit_index_map[(i,j)] = bit_index
    #     bit_index += 1
    
    bit_index_map = {}
    idx = 0
    for i in range(8):
        for j in range(i+1, 8):
            bit_index_map[(i, j)] = idx
            idx += 1
            
    # Generate all 840 Q3 masks
    # Permutations of 0..7
    q3_masks = set()
    import itertools
    for perm in itertools.permutations(range(8)):
        # perm[i] is the vertex in K8 corresponding to Q3 vertex i
        mask = 0
        for u, v in q3_edges_local:
            pu = perm[u]
            pv = perm[v]
            if pu > pv:
                pu, pv = pv, pu
            mask |= (1 << bit_index_map[(pu, pv)])
        q3_masks.add(mask)
        
    q3_masks_list = list(q3_masks)
    
    # 2. Precompute Lookup Table
    # lookup[mask] = 1 if mask contains any Q3 mask
    lookup = bytearray(1 << 28)
    
    # Optimization: Only compute if time permits? 
    # 55M ops might be ~5-10s.
    # We have 115s.
    
    for qm in q3_masks_list:
        Z = full_mask ^ qm
        s = 0
        while True:
            lookup[qm | s] = 1
            if s == Z:
                break
            s = (s - Z) & Z
            
    if time.time() - start_time > time_limit:
        return []

    # 3. Precompute Subsets and Edge Positions
    subsets = []
    subsets_containing_edge = [[[] for _ in range(n)] for _ in range(n)]
    
    # Map global vertices to local 0..7 for each subset
    # For each subset, we need to know which bit in the 28-bit mask corresponds to each global edge (u,v) in the subset
    
    # Precompute for each subset:
    # - list of vertices
    # - for each pair (u,v) in subset, the bit index in the 28-bit mask
    
    subset_info = [] # list of dicts: {(u,v): bit_idx}
    
    import itertools
    for combo in itertools.combinations(range(n), 8):
        subsets.append(combo)
        # Map combo vertices to 0..7
        # combo is sorted
        # local index of combo[i] is i
        # For each pair in combo, find global (u,v) and local bit
        info = {}
        for i in range(8):
            for j in range(i+1, 8):
                u = combo[i]
                v = combo[j]
                # Local bit index for (i,j) in K8
                b_idx = bit_index_map[(i, j)]
                info[(u, v)] = b_idx
                info[(v, u)] = b_idx # just in case, though we use u<v
                # Register this subset for edge (u,v)
                subsets_containing_edge[u][v].append(len(subsets) - 1)
                subsets_containing_edge[v][u].append(len(subsets) - 1)
        subset_info.append(info)
        
    if time.time() - start_time > time_limit:
        return []

    # 4. Initialize Graph
    # Start with a random graph, density 0.9
    adj = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.9:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # Compute initial induced masks
    induced_masks = [0] * len(subsets)
    for idx, combo in enumerate(subsets):
        mask = 0
        info = subset_info[idx]
        for i in range(8):
            u = combo[i]
            for j in range(i+1, 8):
                v = combo[j]
                if (adj[u] >> v) & 1:
                    b_idx = info[(u, v)]
                    mask |= (1 << b_idx)
        induced_masks[idx] = mask
        
    # Compute initial violation count
    violation_count = 0
    for mask in induced_masks:
        if lookup[mask]:
            violation_count += 1
            
    best_violation = violation_count
    best_adj = adj[:]
    best_edge_count = sum(bin(adj[i]).count('1') for i in range(n)) // 2
    
    # 5. Simulated Annealing
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    # Objective: minimize violations, then maximize edges
    # Score = violation_count * 10000 - edge_count
    
    while time.time() - start_time < time_limit:
        if T < min_T:
            T = min_T
            # Restart or break?
            # If we are stuck, we might break.
            # But let's continue cooling slowly or restart?
            # For now, just keep min_T.
            
        # Pick random edge
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
            
        # Check if edge exists? 
        # We can flip it regardless.
        # If it exists, we remove it. If not, we add it.
        
        affected_indices = subsets_containing_edge[u][v]
        
        delta_violation = 0
        new_masks = []
        
        for idx in affected_indices:
            info = subset_info[idx]
            b_idx = info[(u, v)]
            bit = 1 << b_idx
            old_mask = induced_masks[idx]
            new_mask = old_mask ^ bit
            new_masks.append((idx, new_mask))
            
            old_v = lookup[old_mask]
            new_v = lookup[new_mask]
            delta_violation += (new_v - old_v)
            
        # Calculate edge count change
        edge_exists = (adj[u] >> v) & 1
        delta_edges = -1 if edge_exists else 1
        
        # Score change
        # We want to minimize violations.
        # If delta_violation < 0, it's good.
        # If delta_violation == 0, we prefer adding edges (delta_edges > 0).
        
        # Acceptance probability
        if delta_violation < 0:
            accept = True
        elif delta_violation == 0:
            if delta_edges > 0:
                accept = True
            else:
                # Removing an edge with no violation change is bad
                # But might be needed to escape local minima?
                # Usually we want to maximize edges, so removing is bad.
                # Accept with low probability?
                accept = random.random() < 0.1
        else:
            # delta_violation > 0
            # Penalty for violations
            # Cost = delta_violation * 10000 - delta_edges
            cost = delta_violation * 10000 - delta_edges
            if cost < 0:
                accept = True
            else:
                accept = random.random() < math.exp(-cost / T)
                
        if accept:
            # Apply changes
            for idx, new_mask in new_masks:
                induced_masks[idx] = new_mask
            violation_count += delta_violation
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            
            current_edge_count = sum(bin(adj[i]).count('1') for i in range(n)) // 2
            # Update best
            if violation_count < best_violation or (violation_count == best_violation and current_edge_count > best_edge_count):
                best_violation = violation_count
                best_adj = adj[:]
                best_edge_count = current_edge_count
                
        T *= cooling
        
    # Return best graph
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
                
    return edges
