# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Initial fallback: Paley graph on 17 vertices
    squares = {x * x % 17 for x in range(1, 17)}
    best_n = 17
    best_edges = [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]
    
    def has_k4(adj):
        n = len(adj)
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    common = adj[i] & adj[j]
                    k = common & -common
                    while k:
                        k_idx = k.bit_length() - 1
                        mask = common & ~((1 << (k_idx + 1)) - 1)
                        if adj[k_idx] & mask:
                            return True
                        k = common & (common - 1)
        return False

    def find_k4_edge(adj):
        n = len(adj)
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    common = adj[i] & adj[j]
                    k = common & -common
                    while k:
                        k_idx = k.bit_length() - 1
                        mask = common & ~((1 << (k_idx + 1)) - 1)
                        if adj[k_idx] & mask:
                            return (i, j)
                        k = common & (common - 1)
        return None

    def find_ind6_edge(adj, n):
        # Sample random 6-subsets
        for _ in range(300):
            subset = random.sample(range(n), 6)
            is_ind = True
            for u in range(6):
                for v in range(u + 1, 6):
                    u_idx = subset[u]
                    v_idx = subset[v]
                    if (adj[u_idx] >> v_idx) & 1:
                        is_ind = False
                        break
                if not is_ind:
                    break
            if is_ind:
                return (subset[0], subset[1])
        return None

    def verify_valid(adj, n):
        if has_k4(adj):
            return False
        
        # Check for Independent Set 6 (K6 in complement)
        comp_adj = [0] * n
        full_mask = (1 << n) - 1
        for i in range(n):
            comp_adj[i] = full_mask ^ (1 << i) ^ adj[i]
        
        # Check K6 in comp_adj using combinations
        for comb in combinations(range(n), 6):
            is_clique = True
            for u in range(6):
                for v in range(u + 1, 6):
                    u_idx = comb[u]
                    v_idx = comb[v]
                    if not ((comp_adj[u_idx] >> v_idx) & 1):
                        is_clique = False
                        break
                if not is_clique:
                    break
            if is_clique:
                return False
        return True

    for n in range(35, 17, -1):
        if time.time() > deadline - 5.0:
            break
        
        found = False
        for attempt in range(10): # 10 restarts per n
            if time.time() > deadline - 5.0:
                break
            
            # Random init
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Local search
            for step in range(3000):
                if time.time() > deadline - 5.0:
                    break
                
                # Check K4
                edge = find_k4_edge(adj)
                if edge:
                    u, v = edge
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                    continue
                
                # Check Ind 6 (sampling)
                edge = find_ind6_edge(adj, n)
                if edge:
                    u, v = edge
                    if not ((adj[u] >> v) & 1):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                    continue
                
                # If no violations found by sampling, verify fully
                if verify_valid(adj, n):
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if (adj[i] >> j) & 1:
                                edges.append([i, j])
                    best_n = n
                    best_edges = edges
                    found = True
                    break
            
            if found:
                break
        
        if found:
            break # Found max n, stop

    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
