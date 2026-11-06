import time
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    def generate_graph(n, p, seed):
        import random
        rng = random.Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def max_clique(adj, n):
        # adj is a list of bitsets for the complement graph
        # We want to find the maximum clique in adj
        # Returns the bitset of the maximum clique
        
        best_clique = 0
        best_size = 0
        
        # Candidate set P, Excluded set X
        # We use a recursive branch and bound
        # P: vertices that can be added
        # X: vertices that have been excluded (to avoid duplicates)
        
        # Greedy coloring for upper bound
        def color_sort(P):
            # P is a bitset
            # Returns (coloring, order) where coloring[i] is the color of vertex i
            # and order is the list of vertices sorted by color (ascending)
            # The maximum color is the bound
            
            if P == 0:
                return 0, []
            
            U = P
            color = 0
            Q = []  # List of (vertex, color)
            
            while U:
                color += 1
                # Pick a vertex from U that is not adjacent to any vertex in the current color class?
                # Standard Tomita coloring:
                # 1. Select a vertex v from U
                # 2. Assign color to v
                # 3. Remove v and its neighbors from U for this color? No, remove v from U.
                #    Actually, for a valid coloring, neighbors must have different colors.
                #    Simple greedy:
                #    While U is not empty:
                #       Pick v from U
                #       Color[v] = color
                #       U = U \ {v} \ N(v)  -- Wait, this is for independent set in G, which is clique in Gbar.
                #       In Gbar, we need a proper coloring. So neighbors in Gbar cannot have same color.
                #       So if we pick v, we must remove v and its neighbors in Gbar from U for the *current* color assignment?
                #       No, that's for finding an independent set.
                #       For coloring:
                #       While U != 0:
                #           v = pick(U)
                #           color[v] = c
                #           U = U \ {v}
                #           # Neighbors of v in Gbar cannot have color c, so they remain in U for later colors?
                #           # No, the standard algorithm for upper bound in MaxClique:
                #           # We want to color P such that any clique has size <= max_color.
                #           # Greedy:
                #           c = 0
                #           U = P
                #           while U:
                #               c++
                #               Q = []
                #               while U:
                #                   v = pick(U) # e.g. least significant bit
                #                   Q.append(v)
                #                   U = U & ~(1 << v)
                #                   # Remove neighbors of v from U? 
                #                   # If u is neighbor of v in Gbar, u and v cannot have same color.
                #                   # So u must have color > c.
                #                   # Thus, we should remove v and its neighbors from U?
                #                   # Yes, because v takes color c, so its neighbors cannot take color c.
                #                   U = U & ~(adj[v] | (1 << v))
                #               # End while U for this color
                #               # Add Q to order?
                #               pass
            # Let's implement the standard Tomita coloring correctly.
            
            colors = {}
            order = []
            U = P
            c = 0
            while U:
                c += 1
                # Select vertices for color c
                # We iterate through U, picking a vertex, assigning color c, and removing it and its neighbors from U
                # This ensures no two adjacent vertices in Gbar get color c.
                while U:
                    # Pick least significant bit
                    v = (U & -U).bit_length() - 1
                    U &= ~(1 << v)
                    colors[v] = c
                    order.append(v)
                    # Remove neighbors of v in Gbar from U
                    U &= ~adj[v]
            
            # Sort order by color (descending is often better for search, but we return ascending color index in bound)
            # The bound is c
            # We need to return the bound and the ordered list for the search
            # Actually, for the search, we want to iterate vertices in order of decreasing color?
            # Or increasing? 
            # In Bron-Kerbosch for MaxClique, we usually expand vertices in the order they were colored.
            # Vertices with higher colors are "deeper" in the coloring tree.
            # Standard practice: Sort vertices by color (ascending or descending).
            # Let's return the bound (c) and the list of vertices sorted by color (descending) to prioritize promising branches?
            # Or ascending? 
            # Tomita's MCQ: "The vertices are ordered by their colors in non-increasing order."
            
            # Let's sort order by color descending
            order.sort(key=lambda x: colors[x], reverse=True)
            return c, order

        def expand(P, X, size):
            nonlocal best_clique, best_size
            
            # Check time
            if time.time() > deadline:
                return
                
            if P == 0:
                if size > best_size:
                    best_size = size
                    best_clique = X # Wait, X is the set of vertices we've chosen? 
                    # In standard BK for MaxClique:
                    # R is the current clique.
                    # P is candidates.
                    # X is excluded.
                    # Here I am using P and X differently.
                    # Let's stick to R, P, X.
                    pass
            
            # Let's restart the logic inside expand to be clear.
            # We pass R_size, P, X
            # But Python doesn't pass by reference easily for ints.
            # I'll use a class or nonlocal variables.
            
            # Actually, let's implement it properly with nonlocal R_clique.
            pass

        # Let's use a cleaner recursive structure with nonlocal state
        best_clique = 0
        best_size = 0
        
        def search(R_bitset, P_bitset, X_bitset, R_size):
            nonlocal best_clique, best_size
            
            if time.time() > deadline:
                return

            if P_bitset == 0 and X_bitset == 0:
                if R_size > best_size:
                    best_size = R_size
                    best_clique = R_bitset
                return
            
            # Upper bound via coloring
            bound, order = color_sort(P_bitset)
            
            if R_size + bound <= best_size:
                return
            
            # Iterate through vertices in order
            # We need to iterate through the vertices in 'order'
            # But 'order' contains all vertices in P_bitset.
            # We must ensure we only consider vertices still in P_bitset.
            # The coloring removed vertices from U, but 'order' has them all.
            # We iterate 'order', and check if v is in P_bitset.
            
            for v in order:
                if not (P_bitset & (1 << v)):
                    continue
                    
                # Branch on v
                # New P: P intersect N(v)
                # New X: X intersect N(v)
                # New R: R union {v}
                
                v_bit = 1 << v
                new_P = P_bitset & adj[v]
                new_X = X_bitset & adj[v]
                
                # Recurse
                search(R_bitset | v_bit, new_P, new_X, R_size + 1)
                
                # Move v from P to X
                P_bitset &= ~v_bit
                X_bitset |= v_bit
                
                # Update bound? 
                # The bound calculation was for the original P.
                # As we remove vertices, the bound decreases.
                # We can recompute or just check if R_size + bound <= best_size.
                # Since P is shrinking, the max possible clique size from P is non-increasing.
                # However, 'bound' was calculated for the initial P in this call.
                # It is an upper bound for any subset of P.
                # So if R_size + bound <= best_size, we can prune.
                # But bound is fixed for this call.
                # Actually, we should update the bound as we go? 
                # Tomita's algorithm updates the bound.
                # For simplicity and correctness, the initial bound is valid for the whole iteration.
                # If R_size + bound <= best_size, we can break?
                # No, because best_size might increase in a previous branch? 
                # No, we are in a loop. 
                # If the upper bound for the remaining vertices in P (which is a subset of the original P) 
                # is still <= best_size - R_size, we can prune.
                # The bound 'bound' is for the original P.
                # The remaining P is a subset. Its bound is <= 'bound'.
                # So if R_size + bound <= best_size, we can return/break.
                # But we check at the start of the loop?
                # Let's add a check inside the loop.
                
                if R_size + bound <= best_size:
                    break

        # Initial call
        # R = 0, P = All vertices, X = 0
        all_vertices = (1 << n) - 1
        search(0, all_vertices, 0, 0)
        
        return best_clique

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            # Fallback to seed behavior if time is up
            adj = generate_graph(n, p, seed)
            touched = set()
            for u in range(n):
                if adj[u]:
                    touched.add(u)
            results.append(sorted(touched))
            continue
            
        adj_orig = generate_graph(n, p, seed)
        
        # Build complement graph
        adj_comp = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                # If (u,v) is NOT in orig, it IS in comp
                if not (adj_orig[u] & (1 << v)):
                    adj_comp[u] |= (1 << v)
                    adj_comp[v] |= (1 << u)
        
        max_clique_bitset = max_clique(adj_comp, n)
        
        # The transversal is V \ MaxClique
        transversal = []
        for i in range(n):
            if not (max_clique_bitset & (1 << i)):
                transversal.append(i)
        
        results.append(sorted(transversal))
        
    return results
# EVOLVE-BLOCK-END
