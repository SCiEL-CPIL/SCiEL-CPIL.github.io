import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin
    
    best_n = 39
    best_edges = []
    
    def is_c4_free(n, edges):
        """Check if graph has no C4 using adjacency sets."""
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        # Check for C4: for each pair of vertices u,v, check if they have >= 2 common neighbors
        # A C4 exists iff there exist u,v with |N(u) ∩ N(v)| >= 2 and u not adjacent to v
        # Actually no: a C4 is u-a-v-b-u where a,b are distinct, u~a, a~v, v~b, b~u
        # So for the C4 to exist, we need u and v to have at least 2 common neighbors a,b
        # But u and v don't need to be adjacent.
        # Wait, in a C4 u-a-v-b-u, u and v are NOT adjacent (they're opposite vertices).
        # So a C4 exists iff there exist non-adjacent u,v with |N(u) ∩ N(v)| >= 2.
        # But actually even if u~v, if they have 2 common neighbors, that forms a triangle + extra, not necessarily C4.
        # Let me think again. C4: vertices p,q,r,s with edges p-q, q-r, r-s, s-p and no other edges among them.
        # So p and r are opposite (not adjacent), q and s are opposite (not adjacent).
        # A C4 exists iff there exist p,r (not adjacent) with |N(p) ∩ N(r)| >= 2.
        for u in range(n):
            for v in range(u+1, n):
                if v not in adj[u]:  # u and v not adjacent
                    common = adj[u] & adj[v]
                    if len(common) >= 2:
                        return False
        return True
    
    def min_degree(n, edges):
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        return min(deg) if n > 0 else 0
    
    def greedy_build(n, seed, target_min_deg):
        """Greedily add edges to try to achieve target_min_deg while staying C4-free."""
        random.seed(seed)
        adj = [set() for _ in range(n)]
        edges = []
        
        # Try all possible edges in random order
        all_pairs = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(all_pairs)
        
        for u, v in all_pairs:
            # Check if adding (u,v) would create a C4
            # Adding (u,v) creates a C4 iff there exist distinct a,b in N(u) ∩ N(v)
            # such that the cycle u-a-v-b-u is valid. Since a~u, a~v, b~u, b~v,
            # and we need a≠b, and the four vertices u,a,v,b form a 4-cycle.
            # The condition is simply |N(u) ∩ N(v)| >= 2.
            common = adj[u] & adj[v]
            if len(common) <= 1:
                adj[u].add(v)
                adj[v].add(u)
                edges.append([u, v])
        
        return edges
    
    def try_n(n, seed_base, time_budget):
        """Try to build a C4-free graph on n vertices with min degree >= n-39."""
        required_deg = n - 39
        if required_deg < 0:
            required_deg = 0
        
        for attempt in range(100):
            if time.time() > time_budget:
                break
            seed = seed_base + attempt
            edges = greedy_build(n, seed, required_deg)
            md = min_degree(n, edges)
            if md >= required_deg:
                # Verify no C4
                if is_c4_free(n, edges):
                    return n, edges
        return None, None
    
    # Try n from 47 down to 39
    for n in range(47, 38, -1):
        if time.time() > deadline:
            break
        result_n, result_edges = try_n(n, seed_base=1000*n, time_budget=deadline)
        if result_n is not None:
            best_n = result_n
            best_edges = result_edges
            break
    
    return {'n': best_n, 'edges': best_edges}
