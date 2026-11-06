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
        
    if time.time() - start_time > time_limit:
        return []

    def compute_induced_masks(adj):
        """Compute induced masks for all subsets."""
        masks = [0] * num_subsets
        for s_idx, combo in enumerate(subsets):
            mask = 0
            edge_dict = subset_edges[s_idx]
            for (u, v), b_idx in edge_dict.items():
                if (adj[u] >> v) & 1:
                    mask |= (1 << b_idx)
            masks[s_idx] = mask
        return masks

    def count_violations(masks):
        """Count number of violating subsets."""
        return sum(lookup[m] for m in masks)

    def edge_count(adj):
        """Count edges in adjacency list."""
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2

    # 4. Multi-restart greedy initialization
    best_adj = None
    best_edge_count = -1
    best_violation = float('inf')
    best_induced_masks = None
    
    num_greedy_restarts = 10
    
    for restart in range(num_greedy_restarts):
        if time.time() - start_time > time_limit - 60:
            break
            
        # Start with K16
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
        # Add some random edge removals for diversity
        if restart > 0:
            num_removals = random.randint(5, 15)
            for _ in range(num_removals):
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u != v:
                    if u > v:
                        u, v = v, u
                    if (adj[u] >> v) & 1:
                        adj[u] ^= (1 << v)
                        adj[v] ^= (1 << u)
                        
        # Compute initial induced masks
        induced_masks = compute_induced_masks(adj)
        violation_count = count_violations(induced_masks)
        cur_edge_count = edge_count(adj)
        
        # Greedy edge removal to eliminate violations
        for _ in range(200):
            if violation_count == 0:
                break
            if time.time() - start_time > time_limit - 60:
                break
                
            # Find a violating subset
            viol_subset = -1
            for s_idx in range(num_subsets):
                if lookup[induced_masks[s_idx]]:
                    viol_subset = s_idx
                    break
                    
            if viol_subset == -1:
                break
                
            # Try removing each edge in this subset
            combo = subsets[viol_subset]
            edge_dict = subset_edges[viol_subset]
            best_edge = None
            best_delta = 0
            
            for (u, v), b_idx in edge_dict.items():
                if not ((adj[u] >> v) & 1):
                    continue
                    
                # Try removing this edge
                bit = 1 << b_idx
                old_mask = induced_masks[viol_subset]
                new_mask = old_mask ^ bit
                
                # Count how many violations this would fix
                delta = -lookup[old_mask] + lookup[new_mask]
                for (s_idx2, b_idx2) in edge_to_subsets[u][v]:
                    if s_idx2 == viol_subset:
                        continue
                    old_m2 = induced_masks[s_idx2]
                    new_m2 = old_m2 ^ (1 << b_idx2)
                    delta += -lookup[old_m2] + lookup[new_m2]
                    
                if delta < best_delta:
                    best_delta = delta
                    best_edge = (u, v, b_idx, s_idx)
                    
            if best_edge is None:
                u = combo[random.randint(0, 7)]
                v = combo[random.randint(0, 7)]
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                if (adj[u] >> v) & 1:
                    b_idx = edge_dict.get((u, v))
                    if b_idx is not None:
                        bit = 1 << b_idx
                        for (s_idx2, b_idx2) in edge_to_subsets[u][v]:
                            old_m = induced_masks[s_idx2]
                            new_m = old_m ^ (1 << b_idx2)
                            violation_count += -lookup[old_m] + lookup[new_m]
                            induced_masks[s_idx2] = new_m
                        adj[u] ^= (1 << v)
                        adj[v] ^= (1 << u)
                        cur_edge_count -= 1
            else:
                u, v, b_idx, s_idx = best_edge
                bit = 1 << b_idx
                for (s_idx2, b_idx2) in edge_to_subsets[u][v]:
                    old_m = induced_masks[s_idx2]
                    new_m = old_m ^ (1 << b_idx2)
                    violation_count += -lookup[old_m] + lookup[new_m]
                    induced_masks[s_idx2] = new_m
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                cur_edge_count -= 1
                
        # Track best
        if violation_count < best_violation or (violation_count == best_violation and cur_edge_count > best_edge_count):
            best_violation = violation_count
            best_adj = adj[:]
            best_edge_count = cur_edge_count
            best_induced_masks = induced_masks[:]

    # 5. Simulated Annealing from best greedy result
    adj = best_adj[:]
    induced_masks = best_induced_masks[:]
    violation_count = best_violation
    cur_edge_count = best_edge_count
    
    T = 0.5
    cooling = 0.9998
    min_T = 0.01
    no_improve_count = 0
    restart_threshold = 3000
    sa_start_time = time.time()
    
    while time.time() - start_time < time_limit - 10:
        if T < min_T:
            T = min_T
            
        # Pick random edge
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
            
        affected = edge_to_subsets[u][v]
        
        delta_violation = 0
        changes = []
        
        for (s_idx, b_idx) in affected:
            bit = 1 << b_idx
            old_mask = induced_masks[s_idx]
            new_mask = old_mask ^ bit
            changes.append((s_idx, new_mask))
            
            old_v = lookup[old_mask]
            new_v = lookup[new_mask]
            delta_violation += (new_v - old_v)
            
        # Calculate edge count change
        edge_exists = (adj[u] >> v) & 1
        delta_edges = -1 if edge_exists else 1
        
        # Acceptance: prioritize reducing violations, then increasing edges
        if delta_violation < 0:
            accept = True
        elif delta_violation == 0:
            if delta_edges > 0:
                accept = True
            else:
                accept = random.random() < 0.03
        else:
            # delta_violation > 0: only accept if we gain enough edges
            cost = delta_violation * 50 - delta_edges
            if cost < 0:
                accept = True
            else:
                accept = random.random() < math.exp(-cost / T)
                
        if accept:
            for (s_idx, new_mask) in changes:
                induced_masks[s_idx] = new_mask
            violation_count += delta_violation
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            cur_edge_count += delta_edges
            
            if violation_count < best_violation or (violation_count == best_violation and cur_edge_count > best_edge_count):
                best_violation = violation_count
                best_adj = adj[:]
                best_edge_count = cur_edge_count
                best_induced_masks = induced_masks[:]
                no_improve_count = 0
            else:
                no_improve_count += 1
                
        T *= cooling
        
        # Periodic restart if stuck
        if no_improve_count > restart_threshold:
            adj = best_adj[:]
            induced_masks = best_induced_masks[:]
            violation_count = best_violation
            cur_edge_count = best_edge_count
            T = 0.3
            no_improve_count = 0
            restart_threshold = max(1500, int(restart_threshold * 0.85))

    # 6. Final local search polish: try all single edge additions
    if time.time() - start_time < time_limit - 5:
        adj = best_adj[:]
        induced_masks = best_induced_masks[:]
        violation_count = best_violation
        cur_edge_count = best_edge_count
        
        # Try adding each missing edge
        improved = True
        while improved and time.time() - start_time < time_limit - 5:
            improved = False
            for i in range(n):
                for j in range(i+1, n):
                    if (adj[i] >> j) & 1:
                        continue  # Edge exists
                    
                    # Check if adding this edge creates violations
                    delta_v = 0
                    changes = []
                    for (s_idx, b_idx) in edge_to_subsets[i][j]:
                        bit = 1 << b_idx
                        old_mask = induced_masks[s_idx]
                        new_mask = old_mask | bit  # Adding edge
                        changes.append((s_idx, new_mask))
                        delta_v += lookup[new_mask] - lookup[old_mask]
                    
                    if delta_v <= 0:
                        # Apply the change
                        for (s_idx, new_mask) in changes:
                            induced_masks[s_idx] = new_mask
                        violation_count += delta_v
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                        cur_edge_count += 1
                        improved = True
                        
                        if violation_count < best_violation or (violation_count == best_violation and cur_edge_count > best_edge_count):
                            best_violation = violation_count
                            best_adj = adj[:]
                            best_edge_count = cur_edge_count
                            best_induced_masks = induced_masks[:]
        
        # Also try removing edges that don't help
        if time.time() - start_time < time_limit - 5:
            adj = best_adj[:]
            induced_masks = best_induced_masks[:]
            violation_count = best_violation
            cur_edge_count = best_edge_count
            
            improved = True
            while improved and time.time() - start_time < time_limit - 5:
                improved = False
                for i in range(n):
                    for j in range(i+1, n):
                        if not ((adj[i] >> j) & 1):
                            continue  # Edge doesn't exist
                        
                        # Check if removing this edge helps
                        delta_v = 0
                        changes = []
                        for (s_idx, b_idx) in edge_to_subsets[i][j]:
                            bit = 1 << b_idx
                            old_mask = induced_masks[s_idx]
                            new_mask = old_mask ^ bit  # Removing edge
                            changes.append((s_idx, new_mask))
                            delta_v += lookup[new_mask] - lookup[old_mask]
                        
                        if delta_v < 0:
                            # Apply the change
                            for (s_idx, new_mask) in changes:
                                induced_masks[s_idx] = new_mask
                            violation_count += delta_v
                            adj[i] ^= (1 << j)
                            adj[j] ^= (1 << i)
                            cur_edge_count -= 1
                            improved = True
                            
                            if violation_count < best_violation or (violation_count == best_violation and cur_edge_count > best_edge_count):
                                best_violation = violation_count
                                best_adj = adj[:]
                                best_edge_count = cur_edge_count
                                best_induced_masks = induced_masks[:]
    
    # Return best graph
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
                
    return edges
