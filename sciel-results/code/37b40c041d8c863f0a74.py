import random
import itertools
import time
import math

# Precompute Q3 structure
q3_adj = [0]*8
for i in range(8):
    for j in range(i+1, 8):
        if bin(i ^ j).count('1') == 1:
            q3_adj[i] |= (1 << j)
            q3_adj[j] |= (1 << i)

def check_q3_in_subset(subset, adj_bits):
    """
    Check if the induced subgraph on 'subset' (list of 8 global vertex ids)
    contains Q3 as a subgraph.
    """
    local_adj = [0]*8
    for i in range(8):
        vi = subset[i]
        for j in range(8):
            if i != j and (adj_bits[vi] >> subset[j]) & 1:
                local_adj[i] |= (1 << j)
    
    for i in range(8):
        if bin(local_adj[i]).count('1') < 3:
            return False
    
    mapping = [-1]*8
    
    def backtrack(q3_v):
        if q3_v == 8:
            return True
        
        used_mask = 0
        for u in range(q3_v):
            used_mask |= (1 << mapping[u])
            
        for s_idx in range(8):
            if (used_mask >> s_idx) & 1:
                continue
            
            valid = True
            for u in range(q3_v):
                if (q3_adj[u] >> q3_v) & 1:
                    if not ((local_adj[mapping[u]] >> s_idx) & 1):
                        valid = False
                        break
            
            if valid:
                mapping[q3_v] = s_idx
                if backtrack(q3_v + 1):
                    return True
                mapping[q3_v] = -1
                
        return False

    for first in range(8):
        mapping[0] = first
        if backtrack(1):
            return True
        mapping[0] = -1
        
    return False

def construct():
    n = 16
    time_limit = 115
    start_time = time.time()
    
    # Precompute all 8-subsets
    all_subsets = list(itertools.combinations(range(n), 8))
    num_subsets = len(all_subsets)
    
    # Map edge (u,v) to list of subset indices containing both u and v
    edge_to_subsets = {}
    for idx, subset in enumerate(all_subsets):
        for i in range(8):
            for j in range(i+1, 8):
                u = subset[i]
                v = subset[j]
                key = (min(u,v), max(u,v))
                if key not in edge_to_subsets:
                    edge_to_subsets[key] = []
                edge_to_subsets[key].append(idx)
    
    # Initialize graph: Random dense graph
    adj_bits = [0]*n
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.8:
                adj_bits[i] |= (1 << j)
                adj_bits[j] |= (1 << i)
                
    # Initial state evaluation
    subset_has_q3 = [False] * num_subsets
    num_q3s = 0
    for idx, subset in enumerate(all_subsets):
        if check_q3_in_subset(subset, adj_bits):
            subset_has_q3[idx] = True
            num_q3s += 1
            
    num_edges = sum(bin(adj_bits[i]).count('1') for i in range(n)) // 2
    
    best_num_edges = num_edges
    best_adj_bits = adj_bits[:]
    
    # Simulated Annealing
    T = 1.0
    cooling = 0.99
    lambda_val = 1000.0
    
    max_iter = 50000
    current_iter = 0
    
    while time.time() - start_time < time_limit - 5 and current_iter < max_iter:
        current_iter += 1
        
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
            
        key = (u, v)
        is_present = (adj_bits[u] >> v) & 1
        affected_subsets = edge_to_subsets.get(key, [])
        
        delta_q3 = 0
        changes = [] # (idx, old_val)
        
        # Apply move temporarily
        if is_present:
            adj_bits[u] &= ~(1 << v)
            adj_bits[v] &= ~(1 << u)
            for idx in affected_subsets:
                if subset_has_q3[idx]:
                    changes.append((idx, True))
                    if not check_q3_in_subset(all_subsets[idx], adj_bits):
                        delta_q3 -= 1
                        subset_has_q3[idx] = False
            delta_edges = -1
        else:
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
            for idx in affected_subsets:
                if not subset_has_q3[idx]:
                    changes.append((idx, False))
                    if check_q3_in_subset(all_subsets[idx], adj_bits):
                        delta_q3 += 1
                        subset_has_q3[idx] = True
            delta_edges = 1
            
        new_num_q3s = num_q3s + delta_q3
        new_num_edges = num_edges + delta_edges
        
        accept = False
        if num_q3s == 0:
            # Greedy: only accept if still valid and more edges
            if new_num_q3s == 0 and new_num_edges > num_edges:
                accept = True
        else:
            old_energy = num_q3s - lambda_val * num_edges
            new_energy = new_num_q3s - lambda_val * new_num_edges
            delta_energy = new_energy - old_energy
            
            if delta_energy < 0:
                accept = True
            else:
                if T > 0:
                    prob = math.exp(-delta_energy / T)
                    if random.random() < prob:
                        accept = True
        
        if accept:
            num_q3s = new_num_q3s
            num_edges = new_num_edges
            
            if num_q3s == 0 and num_edges > best_num_edges:
                best_num_edges = num_edges
                best_adj_bits = adj_bits[:]
        else:
            # Revert
            if is_present:
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
            else:
                adj_bits[u] &= ~(1 << v)
                adj_bits[v] &= ~(1 << u)
            
            # Revert subset_has_q3
            for idx, old_val in changes:
                subset_has_q3[idx] = old_val
                
        T *= cooling

    # Final verification and cleanup
    # Ensure best_adj_bits is Q3 free
    # If best_adj_bits has Q3s, we must fix it.
    # But best_adj_bits is only updated when num_q3s == 0, so it should be valid.
    # However, if we never found a valid solution, best_adj_bits is the initial one.
    
    # Check validity of best_adj_bits
    valid = True
    for idx, subset in enumerate(all_subsets):
        if check_q3_in_subset(subset, best_adj_bits):
            valid = False
            break
            
    if not valid:
        # Force validity
        adj_bits = best_adj_bits[:]
        while time.time() - start_time < time_limit - 1:
             found = False
             for idx, subset in enumerate(all_subsets):
                 if check_q3_in_subset(subset, adj_bits):
                     edges_in_sub = []
                     for i in range(8):
                         for j in range(i+1, 8):
                             if (adj_bits[subset[i]] >> subset[j]) & 1:
                                 edges_in_sub.append((subset[i], subset[j]))
                     if edges_in_sub:
                         u, v = random.choice(edges_in_sub)
                         adj_bits[u] &= ~(1 << v)
                         adj_bits[v] &= ~(1 << u)
                         found = True
                         break
                 if found:
                     break
             if not found:
                 break
        best_adj_bits = adj_bits

    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj_bits[i] >> j) & 1:
                edges.append([i, j])
                
    return edges

# EVOLVE-BLOCK-END
