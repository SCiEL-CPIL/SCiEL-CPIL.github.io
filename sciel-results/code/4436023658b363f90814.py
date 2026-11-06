import time
import math
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 118.0  # Safety margin for 120s limit (150s hard kill)
    
    n = 16
    full_mask = (1 << 28) - 1
    
    # 1. Precompute Q3 masks in K8
    # Q3 edges in 0..7:
    q3_edges_local = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    
    # Map local edge (u, v) to bit index in 28-bit mask
    bit_index_map = {}
    idx = 0
    for i in range(8):
        for j in range(i+1, 8):
            bit_index_map[(i, j)] = idx
            idx += 1
            
    # Generate all 840 Q3 masks
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
    # lookup[mask] = 1 if mask contains any Q3 mask
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
    
    subset_info = [] # list of dicts: {(u,v): bit_idx}
    
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

    # 4. Greedy Construction: Start with empty graph, add edges greedily if no new Q3
    adj = [0] * n
    induced_masks = [0] * len(subsets)
    edge_count = 0
    edge_order = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(edge_order)
    
    for u, v in edge_order:
        # Check if adding (u,v) creates any new Q3
        affected_indices = subsets_containing_edge[u][v]
        would_violate = False
        for idx in affected_indices:
            info = subset_info[idx]
            b_idx = info[(u, v)]
            bit = 1 << b_idx
            old_mask = induced_masks[idx]
            new_mask = old_mask | bit
            if lookup[new_mask]:
                would_violate = True
                break
        
        if not would_violate:
            # Add edge
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edge_count += 1
            for idx in affected_indices:
                info = subset_info[idx]
                b_idx = info[(u, v)]
                bit = 1 << b_idx
                induced_masks[idx] |= bit

    # 5. Simulated Annealing Polish
    best_edge_count = edge_count
    best_adj = adj[:]
    T = 1.0
    cooling = 0.9999
    min_T = 0.001
    
    while time.time() - start_time < time_limit:
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
        
        # Check if edge exists
        edge_exists = (adj[u] >> v) & 1
        if edge_exists:
            # Removing edge
            delta_edges = -1
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
            
            # Only accept removal if it reduces violations (shouldn't happen in SA start)
            # or with very low probability to escape local minima
            if delta_violation < 0:
                accept = True
            else:
                accept = random.random() < 0.05
        else:
            # Adding edge
            delta_edges = 1
            delta_violation = 0
            new_masks = []
            for idx in affected_indices:
                info = subset_info[idx]
                b_idx = info[(u, v)]
                bit = 1 << b_idx
                old_mask = induced_masks[idx]
                new_mask = old_mask | bit
                new_masks.append((idx, new_mask))
                old_v = lookup[old_mask]
                new_v = lookup[new_mask]
                delta_violation += (new_v - old_v)
            
            # Accept if no new violations, or with probability exp(-cost/T)
            if delta_violation == 0:
                accept = True
            else:
                # Cost is number of new violations
                accept = random.random() < math.exp(-delta_violation * 50 / T)
                
        if accept:
            # Apply changes
            for idx, new_mask in new_masks:
                induced_masks[idx] = new_mask
            if edge_exists:
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                edge_count -= 1
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_count += 1
                
            # Update best
            if edge_count > best_edge_count:
                best_edge_count = edge_count
                best_adj = adj[:]
                
        T *= cooling
        
    # Return best graph
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
                
    return edges
