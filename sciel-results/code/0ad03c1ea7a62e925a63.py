import time
import sys

def construct():
    start_time = time.time()
    deadline = 150.0
    
    def random_graph(n, p, seed):
        import random
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve_instance(n, edges):
        # Build adjacency bitmasks for the complement graph
        # adj_mask[i] has bit j set if (i, j) is an edge in the COMPLEMENT graph
        # i.e., if (i, j) is NOT an edge in the original graph.
        # Note: We need to handle self-loops? No, simple graphs.
        
        # Initialize complement adjacency
        comp_adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                # If (i,j) is NOT in original edges, it IS in complement
                # Checking membership in edges list is O(E). 
                # Better to build original adj first.
                pass
        
        # Build original adjacency bitmasks
        orig_adj = [0] * n
        for u, v in edges:
            orig_adj[u] |= (1 << v)
            orig_adj[v] |= (1 << u)
        
        # Build complement adjacency bitmasks
        # For each i, comp_adj[i] should have bits for j != i where orig_adj[i] does NOT have bit j.
        for i in range(n):
            # All bits except i
            all_bits = (1 << n) - 1
            # Remove i itself
            all_bits ^= (1 << i)
            # Remove neighbors in original graph
            comp_adj[i] = all_bits & (~orig_adj[i])

        best_clique = []
        best_size = 0

        # Bron-Kerbosch with pivoting to find MAXIMAL cliques
        # We want the maximum one.
        # R: current clique
        # P: candidates that can be added
        # X: candidates already processed (to avoid duplicates)
        
        def bron_kerbosch(R, P, X):
            nonlocal best_clique, best_size
            
            if not P and not X:
                if len(R) > best_size:
                    best_size = len(R)
                    best_clique = list(R)
                return
            
            # Pivoting: choose a pivot u from P union X with max degree in P
            # This reduces the number of recursive calls
            if P | X:
                # Find pivot
                # We can just pick an arbitrary vertex from P|X for simplicity, 
                # but max degree is better. For n=40, even arbitrary might be fine, 
                # but let's do a simple heuristic: pick u in P|X that has most neighbors in P.
                u = -1
                max_deg = -1
                # Iterate through P|X
                # To iterate bits, we can use a helper
                mask = P | X
                while mask:
                    # Get lowest set bit
                    lsb = mask & -mask
                    idx = lsb.bit_length() - 1
                    # Count neighbors of idx in P
                    # deg = popcount(comp_adj[idx] & P)
                    # Since we are in complement graph, comp_adj[idx] are neighbors in complement.
                    # We need neighbors in P.
                    deg = (comp_adj[idx] & P).bit_count()
                    if deg > max_deg:
                        max_deg = deg
                        u = idx
                    mask ^= lsb
                
                if u != -1:
                    # Recurse on P \ N(u)
                    # N(u) in complement is comp_adj[u]
                    P_minus_Nu = P & ~comp_adj[u]
                    mask = P_minus_Nu
                    while mask:
                        lsb = mask & -mask
                        v = lsb.bit_length() - 1
                        # R' = R U {v}
                        R_new = R | lsb
                        # P' = P ∩ N(v)
                        P_new = P & comp_adj[v]
                        # X' = X ∩ N(v)
                        X_new = X & comp_adj[v]
                        
                        bron_kerbosch(R_new, P_new, X_new)
                        
                        P &= ~lsb
                        X |= lsb
                        mask ^= lsb
                else:
                    # Fallback if no pivot found (should not happen if P|X not empty)
                    mask = P
                    while mask:
                        lsb = mask & -mask
                        v = lsb.bit_length() - 1
                        R_new = R | lsb
                        P_new = P & comp_adj[v]
                        X_new = X & comp_adj[v]
                        bron_kerbosch(R_new, P_new, X_new)
                        P &= ~lsb
                        X |= lsb
                        mask ^= lsb

        # Initial call
        # R = 0
        # P = all vertices
        # X = 0
        all_vertices = (1 << n) - 1
        bron_kerbosch(0, all_vertices, 0)

        # best_clique is the Maximum Independent Set of the original graph
        # Minimum Vertex Cover is V \ MIS
        mis_set = set()
        mask = 0
        # Convert best_clique bits to set
        # Wait, best_clique was stored as list of ints? No, R is an int bitmask.
        # I need to convert the bitmask of best_clique to a set of vertices.
        
        # Re-convert best_clique bitmask to set
        # Actually, inside bron_kerbosch, R is an int. 
        # When we update best_clique, we should store the int or convert it.
        # Let's convert the final bitmask.
        
        # The variable `best_clique` holds the list? 
        # In my draft above, I said `best_clique = list(R)`, but R is an int.
        # I need to fix that. I will store the bitmask and convert at the end.
        
        # Let's re-verify the code structure.
        # I will store `best_mask`.
        
        return solve_instance_with_mask(n, edges)

    def solve_instance_with_mask(n, edges):
        orig_adj = [0] * n
        for u, v in edges:
            orig_adj[u] |= (1 << v)
            orig_adj[v] |= (1 << u)
        
        comp_adj = [0] * n
        for i in range(n):
            all_bits = (1 << n) - 1
            all_bits ^= (1 << i)
            comp_adj[i] = all_bits & (~orig_adj[i])

        best_mask = 0
        best_size = 0

        def bron_kerbosch(R, P, X):
            nonlocal best_mask, best_size
            
            if not P and not X:
                r_size = R.bit_count()
                if r_size > best_size:
                    best_size = r_size
                    best_mask = R
                return
            
            if P | X:
                # Find pivot u in P|X with max degree in P
                u = -1
                max_deg = -1
                mask = P | X
                while mask:
                    lsb = mask & -mask
                    idx = lsb.bit_length() - 1
                    deg = (comp_adj[idx] & P).bit_count()
                    if deg > max_deg:
                        max_deg = deg
                        u = idx
                    mask ^= lsb
                
                if u != -1:
                    P_minus_Nu = P & ~comp_adj[u]
                    mask = P_minus_Nu
                    while mask:
                        lsb = mask & -mask
                        v = lsb.bit_length() - 1
                        R_new = R | lsb
                        P_new = P & comp_adj[v]
                        X_new = X & comp_adj[v]
                        
                        bron_kerbosch(R_new, P_new, X_new)
                        
                        P &= ~lsb
                        X |= lsb
                        mask ^= lsb
                else:
                    mask = P
                    while mask:
                        lsb = mask & -mask
                        v = lsb.bit_length() - 1
                        R_new = R | lsb
                        P_new = P & comp_adj[v]
                        X_new = X & comp_adj[v]
                        bron_kerbosch(R_new, P_new, X_new)
                        P &= ~lsb
                        X |= lsb
                        mask ^= lsb

        all_vertices = (1 << n) - 1
        bron_kerbosch(0, all_vertices, 0)

        # Convert best_mask to set of vertices
        mis_vertices = set()
        m = best_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            mis_vertices.add(idx)
            m ^= lsb
            
        all_vertices_set = set(range(n))
        min_vc = all_vertices_set - mis_vertices
        return sorted(min_vc)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline - 5:
            # Fallback: return empty or heuristic if running out of time
            # But these should be fast.
            pass
        edges = random_graph(n, p, seed)
        vc = solve_instance_with_mask(n, edges)
        results.append(vc)
        
    return results
