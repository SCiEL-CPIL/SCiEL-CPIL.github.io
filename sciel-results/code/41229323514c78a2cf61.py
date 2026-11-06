import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Helper: Check if graph is C4-free using bitmasks
    def is_c4_free(adj_mask, n):
        for u in range(n):
            for v in range(u + 1, n):
                common = adj_mask[u] & adj_mask[v]
                if common & (common - 1):
                    return False
        return True

    # Helper: Check if there is an independent set of size >= target
    def has_independent_set(adj_mask, n, target=11):
        sys.setrecursionlimit(10000)
        
        def dfs(candidates, size):
            if size >= target:
                return True
            
            if candidates == 0:
                return False
            
            # Pruning: if even taking all candidates doesn't reach target
            if size + candidates.bit_count() < target:
                return False
                
            # Pick a vertex from candidates with highest degree in induced subgraph
            c = candidates
            best_v = -1
            max_deg = -1
            
            while c:
                lsb = c & (-c)
                v = lsb.bit_length() - 1
                c ^= lsb
                
                deg = (adj_mask[v] & candidates).bit_count()
                
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
            
            if best_v == -1:
                return False
                
            # Branch 1: Include best_v
            new_candidates = candidates & ~(adj_mask[best_v] | (1 << best_v))
            if dfs(new_candidates, size + 1):
                return True
                
            # Branch 2: Exclude best_v
            new_candidates = candidates & ~(1 << best_v)
            if dfs(new_candidates, size):
                return True
                
            return False

        all_candidates = (1 << n) - 1
        return dfs(all_candidates, 0)

    # Check if adding edge (u,v) creates a C4
    def can_add_edge(adj_mask, u, v, n):
        nu = adj_mask[u]
        nv = adj_mask[v]
        
        # Check if any neighbor of u is adjacent to any neighbor of v
        if nu.bit_count() < nv.bit_count():
            c = nu
            while c:
                lsb = c & (-c)
                x = lsb.bit_length() - 1
                c ^= lsb
                if adj_mask[x] & nv:
                    return False
        else:
            c = nv
            while c:
                lsb = c & (-c)
                x = lsb.bit_length() - 1
                c ^= lsb
                if adj_mask[x] & nu:
                    return False
        
        return True

    # Local search: try to add edges to reduce independence number
    def local_search_reduce_alpha(adj_mask, n, time_limit):
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Get all non-edges
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj_mask[u] & (1 << v)):
                        if can_add_edge(adj_mask, u, v, n):
                            # Try adding this edge
                            old_u = adj_mask[u]
                            old_v = adj_mask[v]
                            adj_mask[u] |= (1 << v)
                            adj_mask[v] |= (1 << u)
                            
                            # Check if this reduces alpha (or at least doesn't increase it)
                            if not has_independent_set(adj_mask, n, 11):
                                improved = True
                                break
                            else:
                                # Revert
                                adj_mask[u] = old_u
                                adj_mask[v] = old_v
                
                if improved:
                    break
        
        return adj_mask

    # Generate initial C4-free graph using random greedy
    def generate_greedy_c4_free(n, seed):
        random.seed(seed)
        adj_mask = [0] * n
        
        edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(edges)
        
        for u, v in edges:
            if can_add_edge(adj_mask, u, v, n):
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                
        return adj_mask

    # Try to find a graph with n vertices
    for n in range(43, 29, -1):
        max_seeds = 100
        
        for seed in range(max_seeds):
            if time.time() > deadline - 15:
                break
                
            adj_mask = generate_greedy_c4_free(n, seed)
            
            # Verify C4-free
            if not is_c4_free(adj_mask, n):
                continue
                
            # Check independence number
            if not has_independent_set(adj_mask, n, 11):
                edges = []
                for u in range(n):
                    m = adj_mask[u]
                    while m:
                        lsb = m & (-m)
                        v = lsb.bit_length() - 1
                        if u < v:
                            edges.append([u, v])
                        m ^= lsb
                return {'n': n, 'edges': edges}
        
        if time.time() > deadline - 30:
            break
            
    # Fallback: 10 disjoint triangles (n=30, alpha=10)
    fallback_edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        fallback_edges += [[a, b], [a, c], [b, c]]
    
    return {'n': 30, 'edges': fallback_edges}
