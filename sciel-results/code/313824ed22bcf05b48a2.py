import time
import random
import itertools

def construct():
    start_time = time.time()
    time_limit = 115.0
    
    n = 16
    full_mask = (1 << 28) - 1
    
    # 1. Precompute Q3 masks
    q3_edges_local = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    
    bit_index_map = {}
    idx = 0
    for i in range(8):
        for j in range(i+1, 8):
            bit_index_map[(i, j)] = idx
            idx += 1
            
    # Generate all 840 Q3 masks
    q3_masks_set = set()
    for perm in itertools.permutations(range(8)):
        mask = 0
        for u, v in q3_edges_local:
            pu = perm[u]
            pv = perm[v]
            if pu > pv:
                pu, pv = pv, pu
            mask |= (1 << bit_index_map[(pu, pv)])
        q3_masks_set.add(mask)
        
    q3_masks_list = list(q3_masks_set)
    
    # 2. Precompute Subsets and Edge Positions
    subsets = list(itertools.combinations(range(n), 8))
    num_subsets = len(subsets)
    
    # For each edge (u,v), store list of (subset_idx, bit_idx)
    edge_to_subsets = [[[] for _ in range(n)] for _ in range(n)]
    
    # For each subset, store edge mapping
    subset_edges = []
    
    for s_idx, combo in enumerate(subsets):
        edge_dict = {}
        for i in range(8):
            for j in range(i+1, 8):
                u = combo[i]
                v = combo[j]
                b_idx = bit_index_map[(i, j)]
                edge_dict[(u, v)] = b_idx
                edge_to_subsets[u][v].append((s_idx, b_idx))
                edge_to_subsets[v][u].append((s_idx, b_idx))
        subset_edges.append(edge_dict)
        
    # Lazy Cache for Q3 check
    q3_cache = {}
    
    def has_q3(mask):
        if mask in q3_cache:
            return q3_cache[mask]
        # Quick check: Q3 has 12 edges
        if mask.bit_count() < 12:
            q3_cache[mask] = 0
            return 0
        # Check against all 840 masks
        for q in q3_masks_list:
            if (mask & q) == q:
                q3_cache[mask] = 1
                return 1
        q3_cache[mask] = 0
        return 0

    # 3. Greedy Initialization
    adj = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
            
    induced_masks = [0] * num_subsets
    for s_idx in range(num_subsets):
        mask = 0
        edge_dict = subset_edges[s_idx]
        for (u, v), b_idx in edge_dict.items():
            if (adj[u] >> v) & 1:
                mask |= (1 << b_idx)
        induced_masks[s_idx] = mask
        
    violation_count = 0
    for mask in induced_masks:
        if has_q3(mask):
            violation_count += 1
            
    # Greedy edge removal
    max_greedy_steps = 500
    for _ in range(max_greedy_steps):
        if violation_count == 0:
            break
        if time.time() - start_time > time_limit - 30:
            break
            
        viol_subset = -1
        for s_idx in range(num_subsets):
            if has_q3(induced_masks[s_idx]):
                viol_subset = s_idx
                break
                
        if viol_subset == -1:
            break
            
        combo = subsets[viol_subset]
        edge_dict = subset_edges[viol_subset]
        best_edge = None
        best_delta = 0
        
        for (u, v), b_idx in edge_dict.items():
            if not ((adj[u] >> v) & 1):
                continue
            
            bit = 1 << b_idx
            old_mask = induced_masks[viol_subset]
            new_mask = old_mask ^ bit
            
            delta = -has_q3(old_mask) + has_q3(new_mask)
            for (s_idx2, b_idx2) in edge_to_subsets[u][v]:
                if s_idx2 == viol_subset:
                    continue
                old_m2 = induced_masks[s_idx2]
                new_m2 = old_m2 ^ (1 << b_idx2)
                delta += -has_q3(old_m2) + has_q3(new_m2)
                
            if delta < best_delta:
                best_delta = delta
                best_edge = (u, v, b_idx, s_idx)
                
        if best_edge is None:
            # Fallback: remove random edge from subset
            u = combo[random.randint(0, 7)]
            v = combo[random.randint(0, 7)]
            if u == v: continue
            if u > v: u, v = v, u
            if (adj[u] >> v) & 1:
                b_idx = edge_dict.get((u, v))
                if b_idx is not None:
                    bit = 1 << b_idx
                    for (s_idx2, b_idx2) in edge_to_subsets[u][v]:
                        old_m = induced_masks[s_idx2]
                        new_m = old_m ^ (1 << b_idx2)
                        violation_count += -has_q3(old_m) + has_q3(new_m)
                        induced_masks[s_idx2] = new_m
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
        else:
            u, v, b_idx, s_idx = best_edge
            bit = 1 << b_idx
            for (s_idx2, b_idx2) in edge_to_subsets[u][v]:
                old_m = induced_masks[s_idx2]
                new_m = old_m ^ (1 << b_idx2)
                violation_count += -has_q3(old_m) + has_q3(new_m)
                induced_masks[s_idx2] = new_m
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            
    best_violation = violation_count
    best_adj = adj[:]
    best_edge_count = sum(adj[i].bit_count() for i in range(n)) // 2
    
    # 4. Local Search
    # Try to add edges or swap edges
    while time.time() - start_time < time_limit - 3:
        # Pick a random non-edge
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
            
        if (adj[u] >> v) & 1:
            continue # Already an edge
            
        # Calculate delta_violation for adding (u,v)
        delta_violation = 0
        changes = []
        for (s_idx, b_idx) in edge_to_subsets[u][v]:
            bit = 1 << b_idx
            old_mask = induced_masks[s_idx]
            new_mask = old_mask ^ bit
            changes.append((s_idx, new_mask, old_mask))
            delta_violation += has_q3(new_mask) - has_q3(old_mask)
            
        if delta_violation == 0:
            # Accept addition
            for (s_idx, new_mask, _) in changes:
                induced_masks[s_idx] = new_mask
            violation_count += delta_violation
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            best_edge_count += 1
            best_adj = adj[:]
            best_violation = violation_count
        else:
            # Try to find a swap: remove (x,y) to fix violations
            # Only consider edges in the subsets that became violating
            # Identify newly violating subsets
            newly_violating = []
            for (s_idx, new_mask, old_mask) in changes:
                if has_q3(new_mask) and not has_q3(old_mask):
                    newly_violating.append(s_idx)
            
            if not newly_violating:
                continue
                
            # Try to remove an edge from one of the newly violating subsets
            found_swap = False
            for s_idx in newly_violating:
                combo = subsets[s_idx]
                edge_dict = subset_edges[s_idx]
                # Try removing each edge in this subset
                for (x, y), b_idx in edge_dict.items():
                    if not ((adj[x] >> y) & 1):
                        continue
                    # Calculate delta for removing (x,y)
                    delta_remove = 0
                    # We need to check if removing (x,y) fixes all violations
                    # This is expensive. Approximate: check if it reduces violation count
                    # For now, just try a few random edges from the subset
                    pass
            
            # Simpler Swap Strategy:
            # Pick a random edge (x,y) from the graph that is in one of the newly violating subsets
            # and check if adding (u,v) and removing (x,y) results in 0 violations.
            
            # To keep it fast, limit the search
            candidates = []
            for s_idx in newly_violating:
                combo = subsets[s_idx]
                edge_dict = subset_edges[s_idx]
                for (x, y), b_idx in edge_dict.items():
                    if (adj[x] >> y) & 1:
                        candidates.append((x, y))
                        
            if candidates:
                # Shuffle to avoid bias
                random.shuffle(candidates)
                for (x, y) in candidates[:10]: # Limit to 10 candidates
                    if x == u and y == v:
                        continue
                    # Check swap: Add (u,v), Remove (x,y)
                    # We need to calculate the net delta
                    # This requires checking all subsets affected by (u,v) and (x,y)
                    # This is complex. 
                    # Alternative: Just try the swap and check validity?
                    # No, we need to update state.
                    
                    # Let's just reject for now if no simple addition works.
                    # Or, use a simpler heuristic: if delta_violation is small, try to fix it.
                    pass
                    
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
                
    return edges
