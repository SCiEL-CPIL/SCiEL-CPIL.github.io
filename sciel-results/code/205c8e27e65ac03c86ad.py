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
    
    # Precompute for each triple the set of vertex indices it contains (for fast lookup)
    triple_vertices = [set(t) for t in triples]
    
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
        
        # Check triples containing u
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
        
        # Check triples containing v but not u
        for t_idx in vertex_to_triples[v]:
            a, b, c = triples[t_idx]
            if u == a or u == b or u == c:
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
        for _ in range(100):
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
    
    def try_swap_1for1(adj_mask, missing, all_edges, rng):
        """Try 1-for-1 swaps: remove one edge, add one missing edge."""
        if not all_edges or not missing:
            return False, adj_mask
        
        # Shuffle and try a limited number of swaps
        shuffled_edges = all_edges[:]
        shuffled_missing = missing[:]
        rng.shuffle(shuffled_edges)
        rng.shuffle(shuffled_missing)
        
        max_tries = min(30, len(shuffled_edges), len(shuffled_missing))
        for i in range(max_tries):
            if time.time() > deadline:
                break
            r_u, r_v = shuffled_edges[i]
            a_u, a_v = shuffled_missing[i]
            
            # Remove r_u-r_v
            adj_mask[r_u] &= ~(1 << r_v)
            adj_mask[r_v] &= ~(1 << r_u)
            # Add a_u-a_v
            adj_mask[a_u] |= (1 << a_v)
            adj_mask[a_v] |= (1 << a_u)
            
            if is_k33_free(adj_mask):
                adj_mask = greedy_add(adj_mask)
                return True, adj_mask
            else:
                # Revert
                adj_mask[r_u] |= (1 << r_v)
                adj_mask[r_v] |= (1 << r_u)
                adj_mask[a_u] &= ~(1 << a_v)
                adj_mask[a_v] &= ~(1 << a_u)
        
        return False, adj_mask
    
    def try_swap_2for1(adj_mask, missing, all_edges, rng):
        """Try 2-for-1 swaps: remove two edges, add one missing edge."""
        if not all_edges or not missing or len(all_edges) < 2:
            return False, adj_mask
        
        shuffled_edges = all_edges[:]
        shuffled_missing = missing[:]
        rng.shuffle(shuffled_edges)
        rng.shuffle(shuffled_missing)
        
        max_tries = min(15, len(shuffled_missing))
        for i in range(max_tries):
            if time.time() > deadline:
                break
            a_u, a_v = shuffled_missing[i]
            
            # Pick two random edges to remove
            e1_idx = rng.randrange(len(shuffled_edges))
            e2_idx = rng.randrange(len(shuffled_edges) - 1)
            if e2_idx >= e1_idx:
                e2_idx += 1
            
            r1_u, r1_v = shuffled_edges[e1_idx]
            r2_u, r2_v = shuffled_edges[e2_idx]
            
            # Remove both edges
            adj_mask[r1_u] &= ~(1 << r1_v)
            adj_mask[r1_v] &= ~(1 << r1_u)
            adj_mask[r2_u] &= ~(1 << r2_v)
            adj_mask[r2_v] &= ~(1 << r2_u)
            # Add a_u-a_v
            adj_mask[a_u] |= (1 << a_v)
            adj_mask[a_v] |= (1 << a_u)
            
            if is_k33_free(adj_mask):
                adj_mask = greedy_add(adj_mask)
                return True, adj_mask
            else:
                # Revert
                adj_mask[r1_u] |= (1 << r1_v)
                adj_mask[r1_v] |= (1 << r1_u)
                adj_mask[r2_u] |= (1 << r2_v)
                adj_mask[r2_v] |= (1 << r2_u)
                adj_mask[a_u] &= ~(1 << a_v)
                adj_mask[a_v] &= ~(1 << a_u)
        
        return False, adj_mask
    
    def local_search(adj_mask, max_iters=300, rng=None):
        if rng is None:
            rng = random.Random()
        
        # Phase 1: Repair
        repair(adj_mask)
        
        # Phase 2: Greedy Add
        adj_mask = greedy_add(adj_mask)
        
        # Phase 3: Local Search
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Try to add missing edges
            missing = [(i, j) for i in range(n) for j in range(i + 1, n) if not (adj_mask[i] & (1 << j))]
            if not missing:
                break
            
            rng.shuffle(missing)
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
            
            # If no direct addition works, try swaps
            all_edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adj_mask[i] & (1 << j)]
            if not all_edges:
                break
            
            # Try 1-for-1 swap first
            improved, adj_mask = try_swap_1for1(adj_mask, missing, all_edges, rng)
            if improved:
                continue
            
            # Try 2-for-1 swap
            improved, adj_mask = try_swap_2for1(adj_mask, missing, all_edges, rng)
            if improved:
                continue
            
            # Escape local optimum: remove a random edge
            if all_edges:
                r_u, r_v = all_edges[rng.randrange(len(all_edges))]
                adj_mask[r_u] &= ~(1 << r_v)
                adj_mask[r_v] &= ~(1 << r_u)
                adj_mask = greedy_add(adj_mask)
        
        return adj_mask
    
    def final_polish(adj_mask, rng):
        """Dedicated high-iteration polish pass on the best solution."""
        # Do several rounds of greedy add + local search
        for round_num in range(5):
            if time.time() > deadline - 5:
                break
            adj_mask = greedy_add(adj_mask)
            adj_mask = local_search(adj_mask, max_iters=100, rng=rng)
        
        # Final greedy add
        adj_mask = greedy_add(adj_mask)
        return adj_mask
    
    best_adj = None
    best_edges = 0
    rng = random.Random(42)
    
    # Multiple restarts with varied strategies
    for restart in range(150):
        if time.time() > deadline - 10:
            break
        
        # Vary initialization strategy
        strategy = restart % 3
        if strategy == 0:
            # Random dense
            density = 0.4 + rng.random() * 0.3
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if rng.random() < density:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
        elif strategy == 1:
            # Near-complete with random removals
            density = 0.7 + rng.random() * 0.2
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if rng.random() < density:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
        else:
            # Block structure: split into groups, dense within, sparse between
            group_size = 5
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    gi, gj = i // group_size, j // group_size
                    if gi == gj:
                        p = 0.8
                    else:
                        p = 0.3
                    if rng.random() < p:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
        
        # Repair and Optimize
        adj = local_search(adj, max_iters=150, rng=rng)
        
        cnt = edge_count(adj)
        if cnt > best_edges:
            best_edges = cnt
            best_adj = list(adj)
    
    if best_adj is None:
        best_adj = [0] * n
        best_edges = 0
    
    # Final polish on the best solution
    if time.time() < deadline - 5:
        best_adj = final_polish(best_adj, rng)
        best_edges = edge_count(best_adj)
    
    edges = mask_to_edges(best_adj)
    return edges
