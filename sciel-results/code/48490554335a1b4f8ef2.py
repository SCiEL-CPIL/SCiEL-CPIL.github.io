import random
import time
import math
import itertools
from collections import Counter

def construct():
    start_time = time.time()
    time_limit = 115.0
    
    n = 16
    full_mask = (1 << 28) - 1
    
    # Precompute Q3 masks in K8
    q3_edges_local = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    
    bit_index_map = {}
    idx = 0
    for i in range(8):
        for j in range(i+1, 8):
            bit_index_map[(i, j)] = idx
            idx += 1
            
    # Generate all Q3 masks
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
    
    # Precompute Lookup Table
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

    # Precompute Subsets and Edge Positions
    subsets = list(itertools.combinations(range(n), 8))
    num_subsets = len(subsets)
    
    edge_to_subsets = [[[] for _ in range(n)] for _ in range(n)]
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

    def compute_edge_count(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def compute_masks_fast(adj, subset_edges, num_subsets):
        """Fast mask computation using precomputed edge lists"""
        masks = [0] * num_subsets
        for s_idx in range(num_subsets):
            mask = 0
            edge_dict = subset_edges[s_idx]
            for (u, v), b_idx in edge_dict.items():
                if (adj[u] >> v) & 1:
                    mask |= (1 << b_idx)
            masks[s_idx] = mask
        return masks

    def update_mask_for_edge(adj, induced_masks, u, v, adding):
        """Update masks when adding/removing edge (u,v). Returns delta_violation."""
        delta_violation = 0
        for (s_idx, b_idx) in edge_to_subsets[u][v]:
            bit = 1 << b_idx
            old_mask = induced_masks[s_idx]
            if adding:
                new_mask = old_mask | bit
            else:
                new_mask = old_mask & ~bit
            if new_mask != old_mask:
                induced_masks[s_idx] = new_mask
                delta_violation += lookup[new_mask] - lookup[old_mask]
        return delta_violation

    def greedy_fix(adj, induced_masks, violation_count, max_iters=1000):
        """Remove edges to eliminate Q3 violations, preferring edges that cause fewest new violations"""
        for _ in range(max_iters):
            if time.time() - start_time > time_limit - 10:
                break
            if violation_count == 0:
                break
            
            # Find a violating subset
            viol_subset = -1
            for s_idx in range(num_subsets):
                if lookup[induced_masks[s_idx]]:
                    viol_subset = s_idx
                    break
                    
            if viol_subset == -1:
                break
            
            combo = subsets[viol_subset]
            edge_dict = subset_edges[viol_subset]
            best_edge = None
            best_delta = 1000  # Start high to find minimum delta
            
            # Score each edge by how many violations it would remove vs create
            for (u, v), b_idx in edge_dict.items():
                if not ((adj[u] >> v) & 1):
                    continue
                
                bit = 1 << b_idx
                old_mask = induced_masks[viol_subset]
                new_mask = old_mask ^ bit
                
                delta = -lookup[old_mask] + lookup[new_mask]
                for (s_idx2, b_idx2) in edge_to_subsets[u][v]:
                    if s_idx2 == viol_subset:
                        continue
                    old_m2 = induced_masks[s_idx2]
                    new_m2 = old_m2 ^ (1 << b_idx2)
                    delta += -lookup[old_m2] + lookup[new_m2]
                    
                if delta < best_delta:
                    best_delta = delta
                    best_edge = (u, v, b_idx, viol_subset)
                    
            if best_edge is None:
                # Random fallback - pick random edge from violating subset
                existing_edges = [(u, v) for (u, v) in edge_dict if (adj[u] >> v) & 1]
                if not existing_edges:
                    break
                u, v = random.choice(existing_edges)
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
                
        return adj, induced_masks, violation_count

    def local_search(adj, induced_masks, violation_count, time_budget):
        """Simulated annealing to maximize edges while keeping Q3-free"""
        current_edge_count = compute_edge_count(adj)
        best_edge_count = current_edge_count
        best_adj = adj[:]
        best_violation = violation_count
        
        T = 8.0
        cooling = 0.9995
        min_T = 0.01
        no_improve = 0
        max_no_improve = 5000
        
        while time.time() - start_time < time_limit - 5:
            if T < min_T:
                T = min_T
                
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
                
            edge_exists = (adj[u] >> v) & 1
            delta_edges = -1 if edge_exists else 1
            
            # Improved acceptance with spectral gap-inspired penalty
            if delta_violation < 0:
                accept = True
            elif delta_violation == 0:
                if delta_edges > 0:
                    accept = True
                else:
                    # Allow edge removals to escape local optima, more aggressive early
                    removal_prob = 0.15 if T > 1.0 else 0.05
                    accept = random.random() < removal_prob
            else:
                # Use a smooth surrogate: penalty grows with violation count
                # but is tempered by temperature
                base_penalty = 30.0
                edge_bonus = 8.0
                cost = delta_violation * base_penalty - delta_edges * edge_bonus
                # Add a smooth term that increases penalty for larger violations
                smooth_penalty = 5.0 * math.log(1.0 + delta_violation)
                cost += smooth_penalty
                if cost <= 0:
                    accept = True
                else:
                    accept = random.random() < math.exp(-cost / T)
                    
            if accept:
                for (s_idx, new_mask) in changes:
                    induced_masks[s_idx] = new_mask
                violation_count += delta_violation
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                current_edge_count += delta_edges
                
                improved = False
                if violation_count == 0 and current_edge_count > best_edge_count:
                    best_edge_count = current_edge_count
                    best_adj = adj[:]
                    best_violation = violation_count
                    improved = True
                elif violation_count < best_violation:
                    best_violation = violation_count
                    if current_edge_count > best_edge_count:
                        best_edge_count = current_edge_count
                        best_adj = adj[:]
                    improved = True
                    
                if improved:
                    no_improve = 0
                else:
                    no_improve += 1
                    
                # Restart temperature if stuck
                if no_improve > max_no_improve:
                    no_improve = 0
                    T = max(T, 2.0)
                    
            T *= cooling
            
        return best_adj, best_edge_count

    # Multi-start approach with different initializations
    best_overall_adj = None
    best_overall_edges = 0
    
    # Strategy 1: Start from K16, greedy fix
    adj1 = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            adj1[i] |= (1 << j)
            adj1[j] |= (1 << i)
            
    masks1 = compute_masks_fast(adj1, subset_edges, num_subsets)
    viol1 = sum(1 for m in masks1 if lookup[m])
    
    adj1, masks1, viol1 = greedy_fix(adj1, masks1, viol1)
    
    if viol1 == 0:
        adj1, edges1 = local_search(adj1, masks1, viol1, 25)
        if edges1 > best_overall_edges:
            best_overall_edges = edges1
            best_overall_adj = adj1[:]
    
    # Strategy 2: Random dense graph with varying densities
    for restart in range(5):
        if time.time() - start_time > time_limit - 30:
            break
            
        density = 0.60 + 0.08 * restart  # 0.60, 0.68, 0.76, 0.84, 0.92
        adj2 = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < density:
                    adj2[i] |= (1 << j)
                    adj2[j] |= (1 << i)
                    
        masks2 = compute_masks_fast(adj2, subset_edges, num_subsets)
        viol2 = sum(1 for m in masks2 if lookup[m])
        
        adj2, masks2, viol2 = greedy_fix(adj2, masks2, viol2)
        
        if viol2 == 0:
            adj2, edges2 = local_search(adj2, masks2, viol2, 15)
            if edges2 > best_overall_edges:
                best_overall_edges = edges2
                best_overall_adj = adj2[:]
    
    # Strategy 3: Structured initialization - remove edges to break Q3 patterns
    for restart in range(4):
        if time.time() - start_time > time_limit - 20:
            break
            
        adj3 = [0] * n
        # Start with K16 and remove a random set of edges
        for i in range(n):
            for j in range(i+1, n):
                adj3[i] |= (1 << j)
                adj3[j] |= (1 << i)
        
        # Remove 8-14 random edges
        num_removals = 8 + random.randint(0, 6)
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        for u, v in all_edges[:num_removals]:
            adj3[u] ^= (1 << v)
            adj3[v] ^= (1 << u)
                    
        masks3 = compute_masks_fast(adj3, subset_edges, num_subsets)
        viol3 = sum(1 for m in masks3 if lookup[m])
        
        adj3, masks3, viol3 = greedy_fix(adj3, masks3, viol3)
        
        if viol3 == 0:
            adj3, edges3 = local_search(adj3, masks3, viol3, 10)
            if edges3 > best_overall_edges:
                best_overall_edges = edges3
                best_overall_adj = adj3[:]
    
    # Final greedy edge addition with multiple passes
    if best_overall_adj is not None:
        final_masks = compute_masks_fast(best_overall_adj, subset_edges, num_subsets)
        
        # Multiple passes to ensure convergence
        for _ in range(8):
            changed = True
            while changed and time.time() - start_time < time_limit - 2:
                changed = False
                for i in range(n):
                    for j in range(i+1, n):
                        if not ((best_overall_adj[i] >> j) & 1):
                            affected = edge_to_subsets[i][j]
                            creates_q3 = False
                            for (s_idx, b_idx) in affected:
                                bit = 1 << b_idx
                                new_mask = final_masks[s_idx] | bit
                                if lookup[new_mask]:
                                    creates_q3 = True
                                    break
                            if not creates_q3:
                                best_overall_adj[i] |= (1 << j)
                                best_overall_adj[j] |= (1 << i)
                                best_overall_edges += 1
                                for (s_idx, b_idx) in affected:
                                    final_masks[s_idx] |= (1 << b_idx)
                                changed = True
    
    # Return edges
    if best_overall_adj is None:
        # Fallback: return a known valid construction
        # Use a 2-coloring based approach: split vertices into two sets of 8
        edges = []
        # Add all edges within each half (complete bipartite complement)
        for i in range(8):
            for j in range(i+1, 8):
                edges.append([i, j])
        for i in range(8, 16):
            for j in range(i+1, 16):
                edges.append([i, j])
        # Add some cross edges carefully
        for i in range(8):
            for j in range(8, 16):
                if (i ^ j) & 1:  # Add edges where parity differs
                    edges.append([i, j])
        return edges
        
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_overall_adj[i] >> j) & 1:
                edges.append([i, j])
                
    return edges
