import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

    # Helper: Check if graph is C4-free using bitmasks
    # adj_mask[i] is a bitmask of neighbors of vertex i
    def is_c4_free(adj_mask, n):
        for u in range(n):
            for v in range(u + 1, n):
                # Common neighbors of u and v
                common = adj_mask[u] & adj_mask[v]
                # If there are more than 1 common neighbor, a C4 exists
                # Check if common has more than 1 bit set
                if common & (common - 1):
                    return False
        return True

    # Helper: Check if there is an independent set of size >= target
    def has_independent_set(adj_mask, n, target=11):
        sys.setrecursionlimit(10000)
        node_count = [0]
        max_nodes = 2000000  # Limit to prevent infinite loops or excessive time per check
        
        def dfs(candidates, size):
            if size >= target:
                return True
            
            if candidates == 0:
                return False
            
            # Pruning: if even taking all candidates doesn't reach target
            if size + candidates.bit_count() < target:
                return False
            
            # Limit node count
            node_count[0] += 1
            if node_count[0] > max_nodes:
                return False
            
            # Pick a vertex from candidates
            # Strategy: pick vertex with highest degree in induced subgraph
            # Early stop if we find a vertex with very high degree for better pruning
            c = candidates
            best_v = -1
            max_deg = -1
            
            while c:
                lsb = c & (-c)
                v = lsb.bit_length() - 1
                c ^= lsb
                
                # Degree of v within the current candidate set
                deg = (adj_mask[v] & candidates).bit_count()
                
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
                    # Early stop heuristic: if degree is high, it's a good branching point
                    if deg >= 10:
                        break
            
            if best_v == -1:
                return False
            
            # Branch 1: Include best_v in independent set
            # Remove best_v and all its neighbors from candidates
            new_candidates = candidates & ~(adj_mask[best_v] | (1 << best_v))
            if dfs(new_candidates, size + 1):
                return True
                
            # Branch 2: Exclude best_v from independent set
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
            # A C4 is created if there exists x in N(u) and y in N(v) such that x is adjacent to y.
            # Actually, the condition for no C4 in the graph *after* adding edge (u,v) is:
            # For all x in N(u) \ {v} and y in N(v) \ {u}, x and y are not adjacent.
            # More simply: The sets N(u) and N(v) must not have any edges between them.
            # Since we are adding (u,v), u is not in N(u) and v is not in N(v).
            # We need to check if any neighbor of u is adjacent to any neighbor of v.
            
            # Efficient check:
            # Iterate over the smaller neighbor set.
            if nu.bit_count() < nv.bit_count():
                c = nu
                safe = True
                while c:
                    lsb = c & (-c)
                    x = lsb.bit_length() - 1
                    c ^= lsb
                    # If x is adjacent to any node in nv, we have a C4 (u-x-...-v)
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
                    # If x is adjacent to any node in nu, we have a C4 (v-x-...-u)
                    if adj_mask[x] & nu:
                        safe = False
                        break
            
            if safe:
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                
        return adj_mask

    # Try to find a graph with n vertices, starting from large n
    # We iterate down from 43 to 31
    for n in range(43, 30, -1):
        # Number of seeds to try for this n
        # Larger n is harder, so maybe fewer seeds? Or more?
        # Let's try a reasonable number.
        max_seeds = 5000
        
        for seed in range(max_seeds):
            if time.time() > deadline - 15:
                break
                
            adj_mask = generate_greedy_c4_free(n, seed)
            
            # Verify C4-free
            if not is_c4_free(adj_mask, n):
                continue
                
            # Check independence number
            # We want alpha < 11, so has_independent_set(adj, n, 11) should return False
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
    # This graph is C4-free (no cycles of length 4) and has alpha = 10 (pick 1 from each triangle)
    fallback_edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        fallback_edges += [[a, b], [a, c], [b, c]]
    
    return {'n': 30, 'edges': fallback_edges}
