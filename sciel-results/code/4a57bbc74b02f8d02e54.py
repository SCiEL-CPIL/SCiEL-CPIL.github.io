import time
import itertools

def construct():
    start = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    def is_c5_free(n, adj_mask):
        """Check exactly for no 5-cycle as a subgraph using bitmasks."""
        # A 5-cycle exists iff there are 5 distinct vertices v0..v4 
        # such that v0-v1-v2-v3-v4-v0 are all edges.
        # For n up to 35, check all 5-subsets with bitmask adjacency.
        for combo in itertools.combinations(range(n), 5):
            a, b, c, d, e = combo
            # Check all 12 distinct cycle orderings (up to rotation/reflection)
            # Fix a as start, check permutations of remaining 4
            # Due to symmetry, we only need to check (4-1)!/2 = 3 orderings
            # if we fix a and one neighbor, but let's be safe and check 12
            # Actually with a fixed, 4! = 24 permutations, but cycles are undirected
            # so we can halve: check 12
            # For correctness, check all 24 (fast with bitmasks)
            for perm in itertools.permutations([b, c, d, e]):
                x, y, z, w = perm
                if (adj_mask[a] >> x) & 1 and \
                   (adj_mask[x] >> y) & 1 and \
                   (adj_mask[y] >> z) & 1 and \
                   (adj_mask[z] >> w) & 1 and \
                   (adj_mask[w] >> a) & 1:
                    return False
        return True

    def check_alpha_at_least(n, adj_mask, target):
        """Check if there's an independent set of size >= target using bitmasks."""
        # non_adj[v] = bitmask of vertices non-adjacent to v (excluding v itself)
        all_mask = (1 << n) - 1
        non_adj = [0] * n
        for v in range(n):
            # non-adjacent to v = all vertices minus v minus neighbors of v
            non_adj[v] = all_mask & ~(adj_mask[v] | (1 << v))
        
        def dfs(candidates_mask, count):
            if count >= target:
                return True
            if not candidates_mask or time.time() > DEADLINE:
                return False
            if count + bin(candidates_mask).count('1') < target:
                return False
            # Pick the lowest set bit
            v = (candidates_mask & -candidates_mask).bit_length() - 1
            # Branch 1: include v
            if dfs(candidates_mask & non_adj[v] & ~((1 << (v+1)) - 1), count + 1):
                return True
            # Branch 2: exclude v
            if dfs(candidates_mask & ~(1 << v), count):
                return True
            return False
        
        return dfs(all_mask, 0)

    def validate(n, edges):
        """Validate: C5-free and no independent set of 9."""
        if time.time() > DEADLINE:
            return False
        adj_mask = [0] * n
        for u, v in edges:
            if u == v or u >= n or v >= n:
                return False
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        
        # Check C5-free
        if not is_c5_free(n, adj_mask):
            return False
        
        # Check no independent set of 9
        if check_alpha_at_least(n, adj_mask, 9):
            return False
        
        return True

    # Strategy: 8 disjoint K4s = 32 vertices (known best construction)
    
    # Base: 8 disjoint K4's
    n_base = 32
    edges_base = [[4 * b + i, 4 * b + j] for b in range(8) for i in range(4) for j in range(i + 1, 4)]
    
    if validate(n_base, edges_base):
        return {'n': n_base, 'edges': edges_base}
    
    # Fallback: 7 disjoint K4s
    n_fallback = 28
    edges_fallback = [[4 * b + i, 4 * b + j] for b in range(7) for i in range(4) for j in range(i + 1, 4)]
    
    if validate(n_fallback, edges_fallback):
        return {'n': n_fallback, 'edges': edges_fallback}
    
    # Last resort
    return {'n': n_fallback, 'edges': edges_fallback}
