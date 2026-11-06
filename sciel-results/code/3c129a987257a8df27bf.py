import random
import time
import math
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

    # Helper: compute violation count and edge count from adj
    def compute_violations_and_edges(adj):
        violation_count = 0
        edge_count = 0
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
                        edge_count += 1
            induced_masks[idx] = mask
            if lookup[mask]:
                violation_count += 1
        # edge_count was double-counted (each edge appears in multiple subsets)
        # Recalculate properly
        edge_count = sum(bin(adj[i]).count('1') for i in range(n)) // 2
        return violation_count, edge_count, induced_masks

    # Helper: final greedy improvement
    def final_greedy(adj, edge_count):
        for i in range(n):
            for j in range(i+1, n):
                if not ((adj[i] >> j) & 1):
                    # Check if adding this edge creates a Q3
                    affected = subsets_containing_edge[i][j]
                    creates_q3 = False
                    for idx in affected:
                        info = subset_info[idx]
                        b_idx = info[(i, j)]
                        bit = 1 << b_idx
                        # Get current mask for this subset from adj
                        mask = 0
                        combo = subsets[idx]
                        for a in range(8):
                            u = combo[a]
                            for b in range(a+1, 8):
                                v = combo[b]
                                if (adj[u] >> v) & 1:
                                    bb_idx = info[(u, v)]
                                    mask |= (1 << bb_idx)
                        new_mask = mask | bit
                        if lookup[new_mask]:
                            creates_q3 = True
                            break
                    if not creates_q3:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                        edge_count += 1
        return adj, edge_count

    # Multiple restarts with diverse initializations
    best_adj_global = None
    best_edge_count_global = 0
    
    # Initial densities for diversity
    densities = [0.6, 0.7, 0.8, 0.9, 0.95]
    
    for density in densities:
        if time.time() - start_time > time_limit - 10:
            break
            
        # Initialize with given density
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < density:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        violation_count, edge_count, induced_masks = compute_violations_and_edges(adj)
        
        # Penalty weight for Q3 violations
        penalty_weight = 50
        tabu_list = {}
        tabu_tenure = 7
        current_step = 0
        current_edge_count = edge_count
        
        # Simulated annealing parameters
        T = 8.0
        cooling = 0.9995
        min_T = 0.005
        
        # Track best solution for this restart
        best_score = -violation_count * penalty_weight + edge_count
        best_adj = adj[:]
        best_edge_count = edge_count
        
        # Non-improving step counter for temperature restart
        non_improving_steps = 0
        max_non_improving = 5000
        
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
            
            # QUBO score change
            score_change = -penalty_weight * delta_violation + delta_edges
            
            # Tabu check
            edge_key = (u, v)
            is_tabu = edge_key in tabu_list and tabu_list[edge_key] > current_step
            
            # Acceptance criteria
            if score_change > 0:
                accept = True
            elif not is_tabu and score_change > -2 * penalty_weight:
                # Smooth acceptance with temperature
                accept = random.random() < math.exp(score_change / T)
            else:
                accept = False
                
            if accept:
                # Apply changes
                for idx, new_mask in new_masks:
                    induced_masks[idx] = new_mask
                violation_count += delta_violation
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                current_edge_count += delta_edges
                current_step += 1
                
                # Add to tabu list
                tabu_list[edge_key] = current_step + tabu_tenure
                
                # Clean up old tabu entries
                if len(tabu_list) > 100:
                    old_keys = [k for k, v in tabu_list.items() if v <= current_step]
                    for k in old_keys:
                        del tabu_list[k]
                
                # Update best
                current_score = -violation_count * penalty_weight + current_edge_count
                if current_score > best_score:
                    best_score = current_score
                    best_adj = adj[:]
                    best_edge_count = current_edge_count
                    non_improving_steps = 0
                else:
                    non_improving_steps += 1
                    
                # Temperature restart if stuck
                if non_improving_steps > max_non_improving:
                    T = 8.0
                    non_improving_steps = 0
            else:
                non_improving_steps += 1
                if non_improving_steps > max_non_improving:
                    T = 8.0
                    non_improving_steps = 0
                    
            T *= cooling
            
        # Final greedy improvement
        best_adj, best_edge_count = final_greedy(best_adj, best_edge_count)
        
        # Update global best
        if best_edge_count > best_edge_count_global:
            best_edge_count_global = best_edge_count
            best_adj_global = best_adj[:]
            
    # If no valid result found, return empty
    if best_adj_global is None:
        return []
    
    # Return best graph
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj_global[i] >> j) & 1:
                edges.append([i, j])
                
    return edges
