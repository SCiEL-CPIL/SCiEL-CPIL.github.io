import random
import time
import math
import itertools

def construct():
    start_time = time.time()
    time_limit = 115.0
    
    n = 16
    full_mask = (1 << 28) - 1
    
    # 1. Precompute Q3 masks in K8
    q3_edges_local = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    
    bit_index_map = {}
    idx = 0
    for i in range(8):
        for j in range(i+1, 8):
            bit_index_map[(i, j)] = idx
            idx += 1
            
    q3_masks = set()
    for perm in itertools.permutations(range(8)):
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
    lookup = bytearray(1 << 28)
    
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
    subset_info = []
    
    for combo in itertools.combinations(range(n), 8):
        subsets.append(combo)
        info = {}
        for i in range(8):
            for j in range(i+1, 8):
                u = combo[i]
                v = combo[j]
                b_idx = bit_index_map[(i, j)]
                info[(u, v)] = b_idx
                info[(v, u)] = b_idx
                subsets_containing_edge[u][v].append(len(subsets) - 1)
                subsets_containing_edge[v][u].append(len(subsets) - 1)
        subset_info.append(info)
        
    if time.time() - start_time > time_limit:
        return []

    # 4. Initialize Graph - start with complete graph and remove edges to fix violations
    adj = [0] * n
    for i in range(n):
        for j in range(i+1, n):
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
    current_edge_count = best_edge_count
    
    # 5. Greedy phase: remove edges to eliminate violations
    # For each violating subset, find edges to remove
    greedy_time_limit = start_time + 15.0
    
    while time.time() < greedy_time_limit and violation_count > 0:
        # Find a violating subset
        violating_idx = -1
        for i, mask in enumerate(induced_masks):
            if lookup[mask]:
                violating_idx = i
                break
        
        if violating_idx == -1:
            break
            
        combo = subsets[violating_idx]
        info = subset_info[violating_idx]
        
        # Find edges in this subset that are present
        # Remove the edge whose removal causes the most violation decrease
        best_edge = None
        best_delta = 0
        
        for i in range(8):
            u = combo[i]
            for j in range(i+1, 8):
                v = combo[j]
                if (adj[u] >> v) & 1:
                    # Check if removing this edge helps
                    b_idx = info[(u, v)]
                    bit = 1 << b_idx
                    affected = subsets_containing_edge[u][v]
                    
                    delta = 0
                    for idx2 in affected:
                        old_mask = induced_masks[idx2]
                        if old_mask & bit:
                            new_mask = old_mask ^ bit
                            delta += (lookup[new_mask] - lookup[old_mask])
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_edge = (u, v)
        
        if best_edge is None:
            break
            
        u, v = best_edge
        # Remove the edge
        adj[u] ^= (1 << v)
        adj[v] ^= (1 << u)
        current_edge_count -= 1
        
        # Update induced masks
        affected = subsets_containing_edge[u][v]
        for idx2 in affected:
            info2 = subset_info[idx2]
            b_idx = info2[(u, v)]
            bit = 1 << b_idx
            induced_masks[idx2] ^= bit
            
        # Update violation count
        for idx2 in affected:
            # We need to recompute, but we already did the delta above
            pass
        
        # Recompute violation count for affected subsets only
        old_violation = 0
        for idx2 in affected:
            if lookup[induced_masks[idx2]]:
                old_violation += 1
        
        # Actually, let's just recompute all violations since it's not too bad
        violation_count = 0
        for mask in induced_masks:
            if lookup[mask]:
                violation_count += 1
        
        if violation_count < best_violation or (violation_count == best_violation and current_edge_count > best_edge_count):
            best_violation = violation_count
            best_adj = adj[:]
            best_edge_count = current_edge_count
    
    if time.time() - start_time > time_limit:
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (best_adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    # 6. Simulated Annealing
    T = 1.0
    cooling = 0.9995
    min_T = 0.001
    
    while time.time() - start_time < time_limit - 5.0:
        if T < min_T:
            T = min_T
            
        # Pick random edge
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
            
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
            
        edge_exists = (adj[u] >> v) & 1
        delta_edges = -1 if edge_exists else 1
        
        # Acceptance criteria
        if delta_violation < 0:
            accept = True
        elif delta_violation == 0:
            if delta_edges > 0:
                accept = True
            else:
                accept = random.random() < 0.05
        else:
            # Penalize violations heavily
            cost = delta_violation * 100 - delta_edges
            if cost < 0:
                accept = True
            else:
                accept = random.random() < math.exp(-cost / T)
                
        if accept:
            for idx, new_mask in new_masks:
                induced_masks[idx] = new_mask
            violation_count += delta_violation
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            current_edge_count += delta_edges
            
            if violation_count < best_violation or (violation_count == best_violation and current_edge_count > best_edge_count):
                best_violation = violation_count
                best_adj = adj[:]
                best_edge_count = current_edge_count
                
        T *= cooling
        
    # 7. Final polish: try to add edges greedily
    if time.time() - start_time < time_limit - 2.0:
        # Try adding each missing edge if it doesn't create violations
        for i in range(n):
            for j in range(i+1, n):
                if not (best_adj[i] >> j) & 1:
                    # Check if adding this edge would create violations
                    # Use the current adj state (which is best_adj)
                    # We need to check against best_adj
                    temp_adj = best_adj[:]
                    temp_adj[i] |= (1 << j)
                    temp_adj[j] |= (1 << i)
                    
                    # Check all affected subsets
                    can_add = True
                    affected = subsets_containing_edge[i][j]
                    for idx in affected:
                        combo = subsets[idx]
                        info = subset_info[idx]
                        mask = 0
                        for a in range(8):
                            u2 = combo[a]
                            for b in range(a+1, 8):
                                v2 = combo[b]
                                if (temp_adj[u2] >> v2) & 1:
                                    b_idx = info[(u2, v2)]
                                    mask |= (1 << b_idx)
                        if lookup[mask]:
                            can_add = False
                            break
                    
                    if can_add:
                        best_adj[i] |= (1 << j)
                        best_adj[j] |= (1 << i)
                        best_edge_count += 1
    
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
                
    return edges
