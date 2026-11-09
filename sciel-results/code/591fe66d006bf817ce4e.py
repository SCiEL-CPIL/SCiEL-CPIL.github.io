import random
import itertools
import time
import math

# Precompute Q3 properties
q3_adj = [0]*8
for i in range(8):
    adj = 0
    for j in range(8):
        if i != j and bin(i ^ j).count('1') == 1:
            adj |= (1 << j)
    q3_adj[i] = adj

n = 16
all_subsets = list(itertools.combinations(range(n), 8))
num_subsets = len(all_subsets)

# Precompute edge to subsets mapping
edge_to_subsets = {}
for i in range(n):
    for j in range(i+1, n):
        edge_to_subsets[(i,j)] = []

for idx, subset in enumerate(all_subsets):
    for i in range(8):
        for j in range(i+1, 8):
            u, v = subset[i], subset[j]
            if u > v: u, v = v, u
            edge_to_subsets[(u,v)].append(idx)

def contains_q3_fast(subset, adj_bits):
    local_adj = [0]*8
    for i in range(8):
        vi = subset[i]
        neighbors = adj_bits[vi]
        for j in range(8):
            if i != j:
                vj = subset[j]
                if (neighbors >> vj) & 1:
                    local_adj[i] |= (1 << j)
    
    for i in range(8):
        if bin(local_adj[i]).count('1') < 3:
            return False
    
    total_deg = sum(bin(local_adj[i]).count('1') for i in range(8))
    if total_deg < 24:
        return False

    mapping = [-1]*8
    
    def backtrack(q3_v):
        if q3_v == 8:
            return True
        used_mask = 0
        for u in range(q3_v):
            if mapping[u] != -1:
                used_mask |= (1 << mapping[u])
        
        candidates = []
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
                candidates.append(s_idx)
                
        for s_idx in candidates:
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
    start_time = time.time()
    time_limit = 115
    
    adj_bits = [0]*n
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.6:
                adj_bits[i] |= (1 << j)
                adj_bits[j] |= (1 << i)
                
    is_bad = [False] * num_subsets
    violations = 0
    for idx, subset in enumerate(all_subsets):
        if contains_q3_fast(subset, adj_bits):
            is_bad[idx] = True
            violations += 1
            
    current_edges = sum(bin(adj_bits[i]).count('1') for i in range(n)) // 2
    
    best_edges = current_edges
    best_adj_bits = adj_bits[:]
    
    T = 1.0
    cooling = 0.999
    
    while time.time() - start_time < time_limit - 5:
        T *= cooling
        if T < 0.01: T = 0.01
        
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v: continue
        if u > v: u, v = v, u
        
        is_present = (adj_bits[u] >> v) & 1
        
        # Store state to revert
        old_is_bad_changes = []
        delta_violations = 0
        
        if is_present:
            # Remove edge
            for idx in edge_to_subsets[(u,v)]:
                if is_bad[idx]:
                    old_is_bad_changes.append((idx, True))
                    adj_bits[u] &= ~(1 << v)
                    adj_bits[v] &= ~(1 << u)
                    still_bad = contains_q3_fast(all_subsets[idx], adj_bits)
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
                    if not still_bad:
                        is_bad[idx] = False
                        delta_violations -= 1
            # Apply
            adj_bits[u] &= ~(1 << v)
            adj_bits[v] &= ~(1 << u)
            current_edges -= 1
            violations += delta_violations
        else:
            # Add edge
            for idx in edge_to_subsets[(u,v)]:
                if not is_bad[idx]:
                    old_is_bad_changes.append((idx, False))
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
                    now_bad = contains_q3_fast(all_subsets[idx], adj_bits)
                    adj_bits[u] &= ~(1 << v)
                    adj_bits[v] &= ~(1 << u)
                    if now_bad:
                        is_bad[idx] = True
                        delta_violations += 1
            # Apply
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
            current_edges += 1
            violations += delta_violations
            
        if is_present:
            delta_cost = delta_violations + 100
        else:
            delta_cost = delta_violations - 100
            
        accept = False
        if delta_cost < 0:
            accept = True
        else:
            if random.random() < math.exp(-delta_cost / T):
                accept = True
                
        if not accept:
            # Revert
            if is_present:
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                current_edges += 1
            else:
                adj_bits[u] &= ~(1 << v)
                adj_bits[v] &= ~(1 << u)
                current_edges -= 1
                
            violations -= delta_violations
            for idx, was_bad in old_is_bad_changes:
                is_bad[idx] = was_bad
                
        if violations == 0 and current_edges > best_edges:
            best_edges = current_edges
            best_adj_bits = adj_bits[:]

    # Repair best_adj_bits if it has violations
    final_adj = best_adj_bits[:]
    # Recalculate violations for final_adj
    final_bad = [False]*num_subsets
    final_violations = 0
    for idx, subset in enumerate(all_subsets):
        if contains_q3_fast(subset, final_adj):
            final_bad[idx] = True
            final_violations += 1
            
    if final_violations > 0:
        for _ in range(1000):
            bad_idx = -1
            for i in range(num_subsets):
                if final_bad[i]:
                    bad_idx = i
                    break
            if bad_idx == -1: break
            subset = all_subsets[bad_idx]
            edges_in_sub = []
            for i in range(8):
                for j in range(i+1, 8):
                    u, v = subset[i], subset[j]
                    if u > v: u, v = v, u
                    if (final_adj[u] >> v) & 1:
                        edges_in_sub.append((u,v))
            if not edges_in_sub: break
            u, v = random.choice(edges_in_sub)
            final_adj[u] &= ~(1 << v)
            final_adj[v] &= ~(1 << u)
            for idx in edge_to_subsets[(u,v)]:
                if final_bad[idx]:
                    if not contains_q3_fast(all_subsets[idx], final_adj):
                        final_bad[idx] = False
                        final_violations -= 1

    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (final_adj[i] >> j) & 1:
                edges.append([i, j])
    return edges
