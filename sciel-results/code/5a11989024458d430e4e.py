import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def has_independent_set_10(adj, n):
        """
        Returns True if the graph has an independent set of size 10.
        Uses branch and bound with bitsets.
        """
        mask_all = (1 << n) - 1
        
        # Precompute degrees for initial ordering? 
        # For n<=41, simple B&B is fast enough if pruned well.
        
        def bb(candidates, size):
            if size >= 10:
                return True
            if candidates == 0:
                return False
            
            # Upper bound: if remaining vertices + current size < 10, fail
            if size + candidates.bit_count() < 10:
                return False
            
            # Heuristic: Pick vertex with max degree in the induced subgraph of candidates
            # Actually, for independent set, we want to pick a vertex that "covers" many others?
            # No, for finding an independent set, we branch on including or excluding a vertex.
            # Standard heuristic: pick vertex with max degree in the subgraph to reduce search space quickly?
            # Or min degree? 
            # Let's use a simple heuristic: pick the vertex with the highest degree in the candidate set.
            
            best_v = -1
            max_deg = -1
            temp = candidates
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                temp ^= lsb
                # Degree of v in the subgraph induced by candidates
                deg = (adj[v] & candidates).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
            
            if best_v == -1:
                return False
            
            # Branch 1: Include best_v in independent set
            # Remove best_v and its neighbors from candidates
            new_cands = candidates & ~(1 << best_v) & ~adj[best_v]
            if bb(new_cands, size + 1):
                return True
            
            # Branch 2: Exclude best_v from independent set
            new_cands = candidates & ~(1 << best_v)
            if bb(new_cands, size):
                return True
            
            return False
        
        return bb(mask_all, 0)

    def is_triangle_free(adj, n):
        for u in range(n):
            neighbors = adj[u]
            temp = neighbors
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                temp ^= lsb
                if v > u:
                    if adj[u] & adj[v]:
                        return False
        return True

    def build_edge_list(adj, n):
        edges = []
        for i in range(n):
            temp = adj[i]
            while temp:
                lsb = temp & -temp
                j = lsb.bit_length() - 1
                if j > i:
                    edges.append([i, j])
                temp ^= lsb
        return edges

    def make_andrasfai(k):
        """
        Construct Andrásfai graph A_k.
        Vertices 0..3k-1.
        Edge (i, j) if (j-i) mod 3k is in {1, k, 2k-1, 2k, 3k-1} ? 
        Actually, standard definition:
        A_k is a 3-partite graph with parts V_0, V_1, V_2 of size k.
        Edges connect V_i to V_{i+1} and V_{i-1} in a specific way.
        Specifically, vertex v in V_i is connected to vertices in V_{i+1} and V_{i-1} 
        based on a cyclic shift.
        
        A simpler construction for A_k (k>=3):
        Vertices 0, ..., 3k-1.
        (i, j) is an edge iff (j - i) mod 3k is in {1, k, 2k-1, 2k, 3k-1}.
        Let's verify for k=3 (n=9, C5 blowup? No, A_3 is C_9? No).
        For k=3, n=9.
        Let's use the known property: A_k is triangle-free and alpha(A_k) = k.
        So A_9 has n=27, alpha=9.
        """
        n = 3 * k
        adj = [0] * n
        # Connect i to i+1, i+k, i+2k-1, i+2k, i+3k-1 (mod n)
        # Wait, the standard Andrásfai graph A_k has edges (i, j) if j-i mod 3k is 1, k, 2k-1, 2k, 3k-1.
        # Let's check if this is triangle free.
        # For k=3: n=9.
        # Diffs: 1, 3, 5, 6, 8.
        # This forms a specific graph.
        
        for i in range(n):
            for d in [1, k, 2*k-1, 2*k, 3*k-1]:
                j = (i + d) % n
                if j != i:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj, n

    # Start with A_9 (n=27, alpha=9)
    adj, n = make_andrasfai(9)
    
    # Verify A_9 is valid
    if not is_triangle_free(adj, n) or has_independent_set_10(adj, n):
        # Fallback to a smaller known valid graph if A_9 construction is wrong
        # A_6 (n=18, alpha=6)
        adj, n = make_andrasfai(6)
        if not is_triangle_free(adj, n) or has_independent_set_10(adj, n):
             # A_3 (n=9, alpha=3)
             adj, n = make_andrasfai(3)
             if not is_triangle_free(adj, n) or has_independent_set_10(adj, n):
                 # C5 (n=5, alpha=2)
                 n = 5
                 adj = [0]*5
                 for i in range(5):
                     adj[i] |= (1 << ((i+1)%5)) | (1 << ((i+4)%5))
                     adj[(i+1)%5] |= (1 << i)
                     adj[(i+4)%5] |= (1 << i)

    best_adj = adj[:]
    best_n = n

    # Try to extend to larger n using local search / random restarts
    # We want to find a triangle-free graph with alpha < 10.
    # We can try to add vertices or modify edges.
    
    # Strategy: Random triangle-free graph generation with alpha check
    # Since n can be up to 40, random generation is hard.
    # However, we can try to "blow up" the structure of A_9 slightly?
    # Or just try random graphs for n in [28, 40] and check.
    
    # Given the time limit, let's try a few random triangle-free graphs for n > best_n
    # and see if any have alpha < 10.
    
    for target_n in range(best_n + 1, 41):
        if time.time() > deadline:
            break
        
        found = False
        attempts = 0
        max_attempts = 1000 # Limit attempts per n
        
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            
            # Generate random triangle-free graph
            adj_try = [0] * target_n
            # Random edge order
            edges = [(i, j) for i in range(target_n) for j in range(i + 1, target_n)]
            random.shuffle(edges)
            
            for u, v in edges:
                # Check if adding (u,v) creates a triangle
                # Common neighbors of u and v must be 0
                if adj_try[u] & adj_try[v]:
                    continue
                # Add edge
                adj_try[u] |= (1 << v)
                adj_try[v] |= (1 << u)
            
            # Check independence
            if not has_independent_set_10(adj_try, target_n):
                # Valid graph found!
                best_adj = adj_try
                best_n = target_n
                found = True
                break # Move to next target_n
        
        if found:
            # We found a larger graph, continue to see if we can go even larger
            # But we already updated best_n, so the loop condition handles it?
            # The loop is `for target_n in range(best_n + 1, 41)`.
            # If we update best_n inside, the range is fixed at start.
            # So we should break and restart or just continue?
            # Since we want max n, we can just break out of inner loop and continue outer.
            # But the outer loop range is fixed. 
            # Let's just break out of the inner loop and let the outer loop continue.
            # However, if we found n=30, we should try n=31 next.
            # The current loop structure `for target_n in range(...)` iterates sequentially.
            # So if we find n=28, we break inner, outer goes to 29.
            # This is correct.
            pass

    # If we didn't find anything better than A_9, we return A_9.
    # We can also try to optimize A_9 with local search to see if we can add vertices?
    # Adding vertices to a maximal triangle-free graph is tricky.
    
    # Let's do a quick local search on the best found graph to see if we can improve?
    # No, local search usually optimizes within a fixed n.
    
    return {'n': best_n, 'edges': build_edge_list(best_adj, best_n)}
