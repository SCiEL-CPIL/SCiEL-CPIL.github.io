import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 135
    n = 20
    vertices = list(range(n))
    triples = list(combinations(vertices, 3))
    num_triples = len(triples)
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_pairs = len(all_pairs)
    
    # Precompute triple membership for faster checks
    vertex_to_triples = [[] for _ in range(n)]
    for idx, (a, b, c) in enumerate(triples):
        vertex_to_triples[a].append(idx)
        vertex_to_triples[b].append(idx)
        vertex_to_triples[c].append(idx)
    
    triples_arr = np.array(triples)  # shape (num_triples, 3)
    
    def popcount(x):
        return x.bit_count()
    
    def is_k33_free(adj_mask):
        for t in triples:
            u, v, w = t
            common = adj_mask[u] & adj_mask[v] & adj_mask[w]
            if popcount(common) > 2:
                return False
        return True
    
    def edge_count(adj_mask):
        return sum(m.bit_count() for m in adj_mask) // 2
    
    def mask_to_edges(adj_mask):
        edges = []
        for i in range(n):
            m = adj_mask[i]
            for j in range(i + 1, n):
                if m & (1 << j):
                    edges.append([i, j])
        return edges
    
    def can_add(u, v, adj_mask):
        bit_v = 1 << v
        bit_u = 1 << u
        new_u = adj_mask[u] | bit_v
        new_v = adj_mask[v] | bit_u
        
        for t_idx in vertex_to_triples[u]:
            a, b, c = triples[t_idx]
            ma = adj_mask[a]
            mb = adj_mask[b]
            mc = adj_mask[c]
            
            if a == u: ma = new_u
            elif a == v: ma = new_v
            if b == u: mb = new_u
            elif b == v: mb = new_v
            if c == u: mc = new_u
            elif c == v: mc = new_v
            
            if popcount(ma & mb & mc) > 2:
                return False
        
        for t_idx in vertex_to_triples[v]:
            a, b, c = triples[t_idx]
            if u in (a, b, c):
                continue
            
            ma = adj_mask[a]
            mb = adj_mask[b]
            mc = adj_mask[c]
            
            if a == v: ma = new_v
            if b == v: mb = new_v
            if c == v: mc = new_v
            
            if popcount(ma & mb & mc) > 2:
                return False
        
        return True
    
    def greedy_add(adj_mask):
        changed = True
        while changed:
            changed = False
            for i in range(n):
                for j in range(i + 1, n):
                    if not (adj_mask[i] & (1 << j)):
                        if can_add(i, j, adj_mask):
                            adj_mask[i] |= (1 << j)
                            adj_mask[j] |= (1 << i)
                            changed = True
        return adj_mask
    
    def find_violation(adj_mask):
        for u, v, w in triples:
            common = adj_mask[u] & adj_mask[v] & adj_mask[w]
            if popcount(common) > 2:
                c_list = [x for x in range(n) if common & (1 << x)]
                if c_list:
                    return (u, v, w, c_list[0], common)
        return None
    
    def repair(adj_mask):
        for _ in range(200):
            if time.time() > deadline:
                break
            if is_k33_free(adj_mask):
                break
            viol = find_violation(adj_mask)
            if viol is None:
                break
            u, v, w, c, common = viol
            adj_mask[u] &= ~(1 << c)
            adj_mask[c] &= ~(1 << u)
    
    def local_search(adj_mask, max_iters=300):
        repair(adj_mask)
        adj_mask = greedy_add(adj_mask)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            missing = [(i, j) for i in range(n) for j in range(i + 1, n) if not (adj_mask[i] & (1 << j))]
            if not missing:
                break
            
            random.shuffle(missing)
            improved = False
            for u, v in missing:
                if time.time() > deadline:
                    break
                if can_add(u, v, adj_mask):
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
                    improved = True
                    break
            
            if improved:
                continue
            
            all_edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adj_mask[i] & (1 << j)]
            if not all_edges:
                break
            
            random.shuffle(all_edges)
            random.shuffle(missing)
            
            # Try 1-for-1 swaps
            for i in range(min(30, len(all_edges), len(missing))):
                if time.time() > deadline:
                    break
                r_u, r_v = all_edges[i]
                a_u, a_v = missing[i]
                
                adj_mask[r_u] &= ~(1 << r_v)
                adj_mask[r_v] &= ~(1 << r_u)
                adj_mask[a_u] |= (1 << a_v)
                adj_mask[a_v] |= (1 << a_u)
                
                if is_k33_free(adj_mask):
                    adj_mask = greedy_add(adj_mask)
                    improved = True
                    break
                else:
                    adj_mask[r_u] |= (1 << r_v)
                    adj_mask[r_v] |= (1 << r_u)
                    adj_mask[a_u] &= ~(1 << a_v)
                    adj_mask[a_v] &= ~(1 << a_u)
            
            if improved:
                continue
            
            # Try 2-for-1 swaps: remove 2 edges, add 1
            for i in range(min(20, len(all_edges) - 1)):
                if time.time() > deadline:
                    break
                r1_u, r1_v = all_edges[i]
                r2_u, r2_v = all_edges[i + 1]
                
                # Skip if edges share a vertex (would be weird)
                if r1_u == r2_u or r1_u == r2_v or r1_v == r2_u or r1_v == r2_v:
                    continue
                    
                a_u, a_v = missing[i % len(missing)]
                
                # Remove both edges
                adj_mask[r1_u] &= ~(1 << r1_v)
                adj_mask[r1_v] &= ~(1 << r1_u)
                adj_mask[r2_u] &= ~(1 << r2_v)
                adj_mask[r2_v] &= ~(1 << r2_u)
                # Add new edge
                adj_mask[a_u] |= (1 << a_v)
                adj_mask[a_v] |= (1 << a_u)
                
                if is_k33_free(adj_mask):
                    adj_mask = greedy_add(adj_mask)
                    improved = True
                    break
                else:
                    # Revert
                    adj_mask[r1_u] |= (1 << r1_v)
                    adj_mask[r1_v] |= (1 << r1_u)
                    adj_mask[r2_u] |= (1 << r2_v)
                    adj_mask[r2_v] |= (1 << r2_u)
                    adj_mask[a_u] &= ~(1 << a_v)
                    adj_mask[a_v] &= ~(1 << a_u)
            
            if improved:
                continue
            
            # Try removing a random edge to escape local optimum
            if all_edges:
                r_u, r_v = all_edges[0]
                adj_mask[r_u] &= ~(1 << r_v)
                adj_mask[r_v] &= ~(1 << r_u)
                adj_mask = greedy_add(adj_mask)
        
        return adj_mask
    
    def polish(adj_mask, max_iters=100):
        """Final high-precision polish pass."""
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Try adding any missing edge
            added = False
            for i in range(n):
                for j in range(i + 1, n):
                    if not (adj_mask[i] & (1 << j)):
                        if can_add(i, j, adj_mask):
                            adj_mask[i] |= (1 << j)
                            adj_mask[j] |= (1 << i)
                            added = True
            
            if not added:
                break
            
            # After each addition, try more
            for i in range(n):
                for j in range(i + 1, n):
                    if not (adj_mask[i] & (1 << j)):
                        if can_add(i, j, adj_mask):
                            adj_mask[i] |= (1 << j)
                            adj_mask[j] |= (1 << i)
        
        return adj_mask
    
    best_adj = None
    best_edges = 0
    rng = random.Random(42)
    
    # Start with a structured initialization: complete graph minus a matching
    # or use random dense starts
    
    # First try a structured start: complete bipartite-like
    # Split into two sets of 10, connect all across
    adj_struct = [0] * n
    for i in range(10):
        for j in range(10, 20):
            adj_struct[i] |= (1 << j)
            adj_struct[j] |= (1 << i)
    adj_struct = local_search(adj_struct, max_iters=200)
    cnt = edge_count(adj_struct)
    if cnt > best_edges:
        best_edges = cnt
        best_adj = list(adj_struct)
    
    # Multiple random restarts
    for restart in range(150):
        if time.time() > deadline - 5:
            break
        
        density = 0.4 + rng.random() * 0.35
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < density:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        adj = local_search(adj, max_iters=150)
        
        cnt = edge_count(adj)
        if cnt > best_edges:
            best_edges = cnt
            best_adj = list(adj)
    
    # Final polish on the best solution
    if best_adj is not None:
        best_adj = polish(best_adj, max_iters=50)
        best_edges = edge_count(best_adj)
    
    if best_adj is None:
        best_adj = [0] * n
        best_edges = 0

    edges = mask_to_edges(best_adj)
    return edges
