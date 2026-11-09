import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin

    # Fallback: 10 disjoint triangles (n=30, alpha=10)
    best_n = 30
    best_edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        best_edges += [[a, b], [a, c], [b, c]]

    def get_edges_from_adj_bitmask(adj_list, n):
        edges = []
        for u in range(n):
            mask = adj_list[u]
            # Iterate over set bits in mask that are > u
            m = mask
            while m:
                # Get lowest set bit
                lsb = m & -m
                v = lsb.bit_length() - 1
                if v > u:
                    edges.append([u, v])
                m ^= lsb
        return edges

    def is_c4_free_bitmask(adj_list, n):
        # C4-free iff for all u,v, |N(u) \cap N(v)| <= 1
        # Note: N(u) does not include u.
        for u in range(n):
            for v in range(u + 1, n):
                common = adj_list[u] & adj_list[v]
                if common.bit_count() > 1:
                    return False
        return True

    def has_independent_set_exact(adj_list, n, k):
        """
        Returns True if there is an independent set of size k in G.
        Equivalent to finding a clique of size k in the complement graph.
        Uses Branch and Bound with Greedy Coloring.
        """
        # Build complement adjacency bitmasks
        # comp_adj[i] has bits j set if i != j and NOT (i,j) in G
        all_mask = (1 << n) - 1
        comp_adj = []
        for i in range(n):
            # Remove self and existing neighbors
            comp_mask = all_mask ^ (1 << i) ^ adj_list[i]
            comp_adj.append(comp_mask)
        
        # Precompute degrees in complement for ordering? 
        # We will use a recursive function.
        
        def greedy_color_bound(candidates_mask):
            """
            Returns an upper bound on the clique size in the subgraph induced by candidates_mask.
            Uses greedy coloring.
            """
            if candidates_mask == 0:
                return 0
            
            colors = 0
            remaining = candidates_mask
            color_class = 0
            color_mask = 0
            
            while remaining:
                # Pick a vertex from remaining
                # Heuristic: pick vertex with lowest degree in remaining? 
                # Or just first one for speed.
                # For coloring bound, order matters. 
                # A simple order is fine for a bound, but better order gives tighter bound.
                # Let's just pick the lowest index vertex in remaining for simplicity and speed.
                v_bit = remaining & -remaining
                v = v_bit.bit_length() - 1
                
                # Remove v from remaining
                remaining ^= v_bit
                
                # Check if v can be added to current color class
                # v is compatible with color class if it has no edges to any vertex in color_class
                # In complement graph, "edge" means "adjacent in complement".
                # So v can join color class if comp_adj[v] & color_mask == 0
                if (comp_adj[v] & color_mask) == 0:
                    color_mask |= v_bit
                else:
                    # Start a new color class
                    colors += 1
                    color_class = v_bit
                    color_mask = v_bit
            
            colors += 1 # Don't forget the last color class
            return colors

        def backtrack(candidates_mask, clique_size):
            if clique_size >= k:
                return True
            
            # Pruning: Not enough vertices left
            if candidates_mask.bit_count() < k - clique_size:
                return False
            
            # Pruning: Coloring bound
            # If the max clique in candidates is smaller than needed, fail
            # The coloring bound gives the max number of colors, which is an upper bound on clique size?
            # No, greedy coloring uses C colors. The clique size is at most C.
            # So if clique_size + C < k, then we cannot reach k.
            bound = greedy_color_bound(candidates_mask)
            if clique_size + bound < k:
                return False
            
            # Branching
            # Pick a vertex v from candidates
            # Heuristic: Pick vertex with highest degree in the induced subgraph?
            # Or just pick the first one.
            # Let's pick the vertex with the highest degree in the complement subgraph to prune faster.
            # Calculating degrees is O(M^2). M is small.
            
            cand_list = []
            temp = candidates_mask
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                cand_list.append(v)
                temp ^= lsb
            
            if not cand_list:
                return False
            
            # Sort by degree in the induced complement subgraph (descending)
            # deg[v] = (comp_adj[v] & candidates_mask).bit_count()
            # This is O(M) per vertex, O(M^2) total.
            cand_list.sort(key=lambda v: (comp_adj[v] & candidates_mask).bit_count(), reverse=True)
            
            v = cand_list[0]
            v_bit = 1 << v
            
            # Option 1: Include v
            # New candidates: neighbors of v in complement, intersected with current candidates
            # Also, we must remove v from candidates.
            # Standard clique search: when we include v, the next candidates must be neighbors of v 
            # AND have index > v? No, that's for maximal cliques enumeration.
            # For decision problem, we can just intersect with neighbors.
            # However, to avoid symmetry and speed up, we can restrict to vertices with index > v 
            # IF we iterate in a specific order. 
            # But here we are using masks. 
            # If we include v, the remaining clique must be formed from N(v) \cap candidates.
            new_mask = candidates_mask & comp_adj[v]
            new_mask &= ~v_bit # Remove v itself
            
            if backtrack(new_mask, clique_size + 1):
                return True
            
            # Option 2: Exclude v
            # Remove v from candidates
            candidates_mask &= ~v_bit
            if backtrack(candidates_mask, clique_size):
                return True
            
            return False

        initial_mask = (1 << n) - 1
        return backtrack(initial_mask, 0)

    def try_greedy_construction(n, seed):
        """
        Attempts to construct a C4-free graph on n vertices with alpha < 11.
        Returns edges if successful, else None.
        """
        random.seed(seed)
        
        # Initialize adjacency bitmasks
        adj_list = [0] * n
        
        # Order of vertex insertion
        # We can try different orders. 
        # One effective strategy: Add vertices one by one, connecting to existing vertices 
        # if it doesn't create a C4.
        
        # To maximize density (and thus minimize alpha), we want to add many edges.
        # Greedy: For each new vertex v, connect to all existing vertices u 
        # such that |N(u) \cap N(v)| < 2.
        
        # Since N(v) is empty initially, we can connect to all?
        # No, if we connect v to u1 and u2, and u1, u2 are connected to w, 
        # then w is a common neighbor of u1 and u2? No.
        # Condition: For any pair of vertices x, y in the graph, |N(x) \cap N(y)| <= 1.
        
        # When adding v, we add edges (v, u).
        # This changes N(v) and N(u).
        # We must ensure that for all existing x, y, the condition holds.
        # Adding (v, u) affects:
        # 1. Pairs involving v: (v, w) for all w.
        #    N(v) grows. N(w) grows (if w=u).
        #    We need |N(v) \cap N(w)| <= 1 for all w.
        #    Initially N(v) is empty. As we add edges, we must check.
        # 2. Pairs involving u: (u, w) for all w != v.
        #    N(u) grows (adds v). N(w) unchanged (unless w=v).
        #    We need |N(u) \cap N(w)| <= 1.
        #    Since N(u) gained v, we must check if v is in N(w).
        #    If v is in N(w), then v is a common neighbor of u and w.
        #    So we must ensure that for any w connected to v, w is NOT connected to u?
        #    Wait. If u-w is an edge, and v-w is an edge, and u-v is an edge, 
        #    then u, v, w form a triangle.
        #    Common neighbors of u and w? 
        #    If u-v-w is a path, v is common neighbor of u and w? No, v is adjacent to u and w.
        #    So v is in N(u) and N(w).
        #    So |N(u) \cap N(w)| includes v.
        #    If there is another common neighbor z, then C4.
        
        # Simplified Greedy Logic:
        # Maintain adj_list.
        # For v in range(n):
        #   candidates = [u for u in range(v)] # Only connect to previous vertices to avoid symmetry?
        #   Actually, we can connect to any u. But let's stick to u < v for simplicity and determinism.
        #   Shuffle candidates.
        #   For u in candidates:
        #      Check if adding (v, u) is safe.
        #      Safe if:
        #      1. For all w in N(u) \ {v}: v not in N(w) OR (if v in N(w), then no other common neighbor)
        #         Actually, simpler: 
        #         Let C = N(v) \cap N(u).
        #         If |C| >= 2, adding (v,u) creates C4? 
        #         No, C4 is a cycle of length 4.
        #         If |N(v) \cap N(u)| >= 2, say w1, w2, then v-w1-u-w2-v is a C4.
        #         So condition: |N(v) \cap N(u)| < 2.
        #      2. Does adding (v,u) create a C4 with v and u as adjacent vertices in the C4?
        #         A C4 is v-u-w-x-v.
        #         This requires u-w, w-x, x-v.
        #         This means there is a path of length 3 from v to u in the existing graph?
        #         No, v and u are the new edge.
        #         The C4 would be v-u-w-z-v.
        #         Edges: (v,u) new, (u,w) old, (w,z) old, (z,v) old.
        #         So we need to check if there exists w, z such that:
        #         w in N(u) and z in N(v) and w-z is an edge.
        #         This is equivalent to: N(u) \cap N(N(v)) is not empty?
        #         Or: For any w in N(u), if w is connected to any z in N(v), then C4.
        
        # This local check is getting complex. 
        # Let's use the global check is_c4_free_bitmask at the end of each vertex addition?
        # No, that's O(n^3) per edge addition, too slow.
        
        # Let's stick to the simple necessary condition: |N(v) \cap N(u)| < 2.
        # Is it sufficient?
        # If we maintain the invariant that the graph is C4-free, and we add an edge (v,u) 
        # such that |N(v) \cap N(u)| < 2, is the new graph C4-free?
        # Any new C4 must involve the new edge (v,u).
        # Case 1: v and u are opposite in C4. 
        #   Path v-w1-u-w2-v. 
        #   w1, w2 are common neighbors of v and u.
        #   We ensured |N(v) \cap N(u)| < 2, so this cannot happen.
        # Case 2: v and u are adjacent in C4.
        #   Cycle v-u-w-x-v.
        #   Edges: (v,u) new, (u,w) old, (w,x) old, (x,v) old.
        #   This implies w is a neighbor of u, x is a neighbor of v, and w-x is an edge.
        #   So we must check: Is there a w in N(u) and x in N(v) such that w-x is an edge?
        #   This is equivalent to checking if the bipartite graph between N(u) and N(v) has any edges.
        #   i.e., for any w in N(u), does w have any neighbor in N(v)?
        #   i.e., (adj_list[w] & adj_list[v]) != 0 for any w in N(u)?
        #   Or: Union of N(w) for w in N(u) intersects N(v)?
        
        # So the full check for adding (v,u) is:
        # 1. (adj_list[v] & adj_list[u]).bit_count() < 2
        # 2. For all w in N(u) \ {v}: 
        #       if (adj_list[w] & adj_list[v]) != 0: 
        #           # There is an edge between N(u) and N(v)
        #           # This creates a C4 v-u-w-x-v where x is in N(w) \cap N(v)
        #           # So we must NOT add the edge.
        
        # This check is O(deg(u) * n/word_size).
        
        order = list(range(n))
        random.shuffle(order)
        
        for v in order:
            # Candidates are all vertices u < v in the order? 
            # No, we are building the graph on vertices 0..n-1.
            # We process vertices in 'order'.
            # When processing v, we can connect to any previously processed vertex u.
            processed = set()
            candidates = []
            for u in order:
                if u == v: continue
                if u in processed:
                    candidates.append(u)
            random.shuffle(candidates)
            
            for u in candidates:
                # Check 1: Common neighbors
                common = adj_list[v] & adj_list[u]
                if common.bit_count() >= 2:
                    continue
                
                # Check 2: C4 with v,u adjacent
                # Check if any neighbor of u is connected to any neighbor of v
                # N(u) is adj_list[u]. N(v) is adj_list[v].
                # We need to check if there exists w in N(u) such that N(w) intersects N(v).
                # Note: w cannot be v (v not in N(u) yet).
                # Also, we should exclude u from N(v) check? No, u is not in N(v) yet.
                
                # Iterate through neighbors of u
                mask_u = adj_list[u]
                # We can optimize: if (adj_list[v] & mask_u) is not empty, 
                # then some neighbor of u is also a neighbor of v. 
                # Wait, that's Check 1.
                # Check 2 is: Does any neighbor w of u have a neighbor x in N(v)?
                # i.e., is there a path u-w-x-v of length 3?
                
                # Efficient check:
                # Let S = N(v).
                # For each w in N(u), check if N(w) intersects S.
                # This is O(deg(u) * n).
                
                valid = True
                temp_mask = mask_u
                while temp_mask:
                    lsb = temp_mask & -temp_mask
                    w = lsb.bit_length() - 1
                    # Check if N(w) intersects N(v)
                    # N(w) is adj_list[w]. N(v) is adj_list[v].
                    if (adj_list[w] & adj_list[v]) != 0:
                        valid = False
                        break
                    temp_mask ^= lsb
                
                if not valid:
                    continue
                
                # Add edge
                adj_list[v] |= (1 << u)
                adj_list[u] |= (1 << v)
            
            processed.add(v)
        
        # Verify C4-free
        if not is_c4_free_bitmask(adj_list, n):
            return None
        
        # Check Alpha
        # Fast greedy check first
        verts = list(range(n))
        random.shuffle(verts)
        independent = []
        for v in verts:
            # Check if v is independent of all in 'independent'
            # v is independent if (adj_list[v] & mask_independent) == 0
            mask_ind = 0
            for u in independent:
                mask_ind |= (1 << u)
            if (adj_list[v] & mask_ind) == 0:
                independent.append(v)
                if len(independent) >= 11:
                    return None
        
        # Exact check
        if has_independent_set_exact(adj_list, n, 11):
            return None
        
        return get_edges_from_adj_bitmask(adj_list, n)

    # Try to construct for n=43 down to 39
    for n in range(43, 38, -1):
        if time.time() > deadline - 10:
            break
        
        found = False
        # Run multiple seeds
        for seed in range(100):
            if time.time() > deadline - 5:
                break
            
            edges = try_greedy_construction(n, seed)
            if edges is not None:
                if n > best_n:
                    best_n = n
                    best_edges = edges
                found = True
                if n == 43:
                    return {'n': 43, 'edges': edges}
        
        # If we found a valid graph for this n, we might try to improve it?
        # No, we just move to next n if we want to maximize n.
        # But if we found n=40, we should try n=41, 42, 43 first.
        # The loop goes 43 -> 39.
        # If we find n=40, we break the inner loop? 
        # No, we should try to find a better one for the SAME n? 
        # No, n is fixed in the outer loop.
        # If we find ANY valid graph for n, we record it.
        # Then we move to n-1? 
        # Wait, if we find n=40, we don't need to check n=39 unless n=40 fails.
        # But we want MAX n.
        # So if we find n=43, return.
        # If we find n=42, record it, and continue to try n=41? No, 42 is better than 41.
        # So if we find a valid graph for n, we should stop trying smaller n?
        # Yes.
        
        if found:
            break

    return {'n': best_n, 'edges': best_edges}
