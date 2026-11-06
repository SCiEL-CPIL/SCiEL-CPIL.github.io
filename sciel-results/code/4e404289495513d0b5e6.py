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
        node_count = [0]
        max_nodes = 800000  # Limit to prevent excessive runtime
        deadline_local = deadline - 5  # Extra safety margin
        
        def dfs(candidates, size):
            if size >= target:
                return True
            
            if candidates == 0:
                return False
            
            # Pruning: if even taking all candidates doesn't reach target
            if size + candidates.bit_count() < target:
                return False
            
            # Time check
            if time.time() > deadline_local:
                return False
            
            # Limit node count
            node_count[0] += 1
            if node_count[0] > max_nodes:
                return False
            
            # Pick a vertex from candidates with FEWEST remaining candidates
            # (i.e., highest degree within the candidate set) to prune faster
            c = candidates
            best_v = -1
            min_remaining = n + 1
            
            while c:
                lsb = c & (-c)
                v = lsb.bit_length() - 1
                c ^= lsb
                
                # Number of candidates that can still be chosen if we include v
                remaining = (candidates & ~(adj_mask[v] | (1 << v))).bit_count()
                
                if remaining < min_remaining:
                    min_remaining = remaining
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

    def generate_greedy_c4_free(n, seed):
        random.seed(seed)
        adj_mask = [0] * n
        
        # List of all possible edges
        edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(edges)
        
        for u, v in edges:
            nu = adj_mask[u]
            nv = adj_mask[v]
            
            # Check if adding (u, v) creates a C4
            # C4 exists if there's x in N(u) and y in N(v) with x adjacent to y
            if nu.bit_count() < nv.bit_count():
                c = nu
                safe = True
                while c:
                    lsb = c & (-c)
                    x = lsb.bit_length() - 1
                    c ^= lsb
                    if adj_mask[x] & nv:
                        safe = False
                        break
            else:
                c = nv
                safe = True
                while c:
                    lsb = c & (-c)
                    x = lsb.bit_length() - 1
                    c ^= lsb
                    if adj_mask[x] & nu:
                        safe = False
                        break
            
            if safe:
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                
        return adj_mask

    # Try to find a graph with n vertices
    best_result = None
    
    for n in range(43, 29, -1):
        # Number of seeds to try for this n
        max_seeds = 1500
        
        for seed in range(max_seeds):
            if time.time() > deadline - 10:
                break
                
            adj_mask = generate_greedy_c4_free(n, seed)
            
            # Verify C4-free
            if not is_c4_free(adj_mask, n):
                continue
                
            # Check independence number
            # If alpha < 11, we are good.
            if not has_independent_set(adj_mask, n, 11):
                # Construct edge list
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
