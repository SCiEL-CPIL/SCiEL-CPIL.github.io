"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from random import Random
    from collections import defaultdict

    start_time = time.time()
    deadline = 115.0  # 120s limit, keep margin

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges_cc = []
        for u in range(k):
            for v in range(u + 1, k):
                edges_cc.append((u, v))
        
        edges_ci = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges_ci.append((u, v))
                    
        return edges_cc, edges_ci

    def solve_split_graph(n, k, edges_ci):
        # 1. Build shadow for each independent vertex
        shadows = {}
        for v in range(k, n):
            mask = 0
            for u, v_edge in edges_ci:
                if v_edge == v:
                    mask |= (1 << u)
            shadows[v] = mask
        
        # 2. Group by exact shadow
        shadow_groups = defaultdict(list)
        for v in range(k, n):
            mask = shadows[v]
            if mask != 0:
                shadow_groups[mask].append(v)
        
        if not shadow_groups:
            return [list(range(k))]
            
        masks = list(shadow_groups.keys())
        m = len(masks)
        
        # 3. Build complement graph of shadow intersections
        # A clique in the complement graph corresponds to a set of shadows with empty intersection.
        # We want to cover all shadows with the minimum number of cliques in the complement graph.
        # This is equivalent to the Clique Cover problem on the complement graph,
        # which is the same as the Coloring problem on the original intersection graph.
        
        # However, the "complement graph" here is defined on the set of unique shadows.
        # Edge (i, j) in complement graph exists if (masks[i] & masks[j]) == 0.
        
        comp_adj = [set() for _ in range(m)]
        for i in range(m):
            for j in range(i + 1, m):
                if (masks[i] & masks[j]) == 0:
                    comp_adj[i].add(j)
                    comp_adj[j].add(i)
        
        # 4. Solve Minimum Clique Cover on comp_adj using Branch and Bound
        # This is equivalent to finding the minimum number of colors to color the vertices
        # such that adjacent vertices in comp_adj have different colors.
        # Wait, no. 
        # A set of shadows S is a valid group if their intersection is non-empty.
        # This means for any pair in S, they are NOT adjacent in comp_adj.
        # So S is an Independent Set in comp_adj.
        # We want to partition V into minimum number of Independent Sets in comp_adj.
        # This is the Chromatic Number of comp_adj.
        
        # Let's use a standard Branch and Bound for Graph Coloring (Chromatic Number).
        # Vertices: 0..m-1
        # Colors: 0..K-1
        # Constraint: if i and j are adjacent in comp_adj, they must have different colors.
        
        # Order vertices by degree in comp_adj (descending) to prune faster
        degrees = [len(comp_adj[i]) for i in range(m)]
        order = sorted(range(m), key=lambda x: degrees[x], reverse=True)
        
        # Map new index to old index
        # We will color in the order of 'order'
        
        best_k = m # Upper bound: each vertex its own color (clique)
        
        # For B&B, we can use a simple recursive search with pruning.
        # State: current vertex index in 'order', color assignment array, current max color used.
        
        # To speed up, we can use a lower bound.
        # A simple lower bound is the size of the maximum clique in comp_adj.
        # Or just 1.
        
        # Since m is small (<= 60), we can try a simple B&B.
        # However, 60 is a bit large for naive B&B without strong pruning.
        # Let's try a heuristic first to get a good upper bound, then B&B to verify/improve.
        
        # Heuristic: Greedy Coloring (First Fit) on comp_adj
        # Order by degree
        colors = [-1] * m
        max_color = 0
        for idx in order:
            used = set()
            for neighbor in comp_adj[idx]:
                if colors[neighbor] != -1:
                    used.add(colors[neighbor])
            
            c = 0
            while c in used:
                c += 1
            colors[idx] = c
            max_color = max(max_color, c)
        
        best_k = max_color + 1
        
        # Now, try to improve best_k using B&B
        # We search for a coloring with k < best_k
        
        # To make B&B efficient, we use the following:
        # 1. Sort vertices by degree descending.
        # 2. At each step, assign the lowest possible color to the current vertex.
        # 3. Prune if current_color_used + 1 > best_k.
        
        # Actually, a simpler approach for small m:
        # We can iterate k from 1 to best_k-1 and check if k-coloring is possible.
        # But checking k-coloring is NP-complete.
        
        # Let's use a more robust B&B that minimizes the number of colors.
        
        # Precompute adjacency bitmasks for speed
        adj_masks = [0] * m
        for i in range(m):
            mask = 0
            for j in comp_adj[i]:
                mask |= (1 << j)
            adj_masks[i] = mask
            
        # We will use a recursive function that assigns colors to vertices in 'order'
        # vertices_assigned: list of vertices already assigned
        # color_of: dict vertex -> color
        
        # To optimize, we can use symmetry breaking:
        # The first vertex must get color 0.
        # The second vertex must get color 0 or 1 (if not adjacent to first).
        
        # Let's implement a standard B&B for Chromatic Number.
        
        # Since m <= 60, and the graph is sparse (p=0.7 means shadows are large, so intersections are likely non-empty,
        # meaning comp_adj is sparse), the chromatic number is likely small (2-5).
        
        # Let's try to find a coloring with k colors for k = 2, 3, ... up to best_k-1.
        # If we find one, we update best_k and continue to see if k-1 works?
        # Actually, we want the MINIMUM. So we start from k=1 and go up?
        # No, we have a good upper bound (greedy). We can try to prove that best_k-1 is impossible,
        # or find a better one.
        
        # Efficient strategy:
        # Try k = 2, 3, ...
        # If k-coloring is possible, best_k = k, break?
        # No, we want the minimum. So if 2-coloring works, that's the min (since k>=1 and 1 only works if no edges).
        
        # Let's check k=1 first.
        if m == 1:
            best_k = 1
        else:
            # Check if graph is empty (no edges in comp_adj) -> 1 color
            if not any(comp_adj[i] for i in range(m)):
                best_k = 1
            else:
                # Try k from 2 to best_k-1
                # We can stop as soon as we find a valid k-coloring, because we are iterating upwards?
                # Wait, if we find a valid 3-coloring, is it possible that a 2-coloring exists?
                # Yes. So we must check from 2 upwards until we find the first valid one.
                
                # But checking k-coloring is hard.
                # Let's use the B&B to search for a solution with a specific K.
                
                def is_k_colorable(K):
                    # Assign colors 0..K-1 to vertices in 'order'
                    # Use backtracking
                    # Pruning: if a vertex has all K colors blocked by neighbors, fail.
                    
                    # To speed up, we can use forward checking or just simple backtracking.
                    # Given m<=60 and K likely small, this should be fast if we order well.
                    
                    # We use a recursive helper
                    # state: idx (next vertex in order to assign), colors (list of length m, -1 if unassigned)
                    
                    # Optimization: If the remaining vertices can be colored with K colors,
                    # we check if the current vertex has available colors.
                    
                    # Let's use a global counter for nodes visited to abort if too slow?
                    # The deadline is 115s. We have 3 instances.
                    # m is at most 60.
                    
                    # Simple B&B:
                    colors = [-1] * m
                    
                    def backtrack(idx):
                        if idx == len(order):
                            return True
                            
                        v = order[idx]
                        
                        # Determine available colors
                        # A color c is available if no neighbor of v has color c.
                        
                        # We can iterate c from 0 to K-1
                        # To break symmetry, we can restrict the first vertex to color 0.
                        # But 'order' is fixed.
                        
                        # Heuristic: try colors that are "most constrained" for neighbors?
                        # Or just 0..K-1.
                        
                        for c in range(K):
                            # Check if c is valid for v
                            valid = True
                            for u in comp_adj[v]:
                                if colors[u] == c:
                                    valid = False
                                    break
                            
                            if valid:
                                colors[v] = c
                                if backtrack(idx + 1):
                                    return True
                                colors[v] = -1
                                
                        return False
                        
                    return backtrack(0)
                
                # Iterate K from 2 to best_k - 1
                # Note: if best_k is 2, we don't need to check K=1 (already checked) or K=2 (we want to see if < 2, i.e., 1).
                # We already checked if graph is empty (K=1).
                # So we check K=2, 3, ... up to best_k-1.
                # If we find a valid K, that is the minimum (since we started from 2).
                
                # Wait, what if the minimum is 1? We handled that.
                # What if the minimum is 2? We check K=2. If valid, return 2.
                # What if the minimum is 3? We check K=2 (fail), then K=3 (success).
                
                # So we just loop K from 2 to best_k - 1.
                # If any K works, that's the answer.
                
                # But wait, if best_k from greedy is 5, and the true min is 3.
                # We check K=2 (fail), K=3 (success). Answer 3.
                
                # If no K < best_k works, answer is best_k.
                
                found = False
                for K in range(2, best_k):
                    if is_k_colorable(K):
                        best_k = K
                        found = True
                        break # Since we iterate upwards, the first success is the minimum.
                        
                # If not found, best_k remains the greedy value.
                
        # 5. Construct cliques from the coloring
        # We need the actual coloring to group vertices.
        # If best_k == 1: all shadows form one group.
        # If best_k > 1: we need to run the coloring algorithm again to get the assignment,
        # or store it during the search.
        
        # Let's re-run to get the assignment for the final best_k.
        # Or, we can modify is_k_colorable to return the coloring.
        
        def find_coloring(K):
            colors = [-1] * m
            def backtrack(idx):
                if idx == len(order):
                    return True, colors
                v = order[idx]
                for c in range(K):
                    valid = True
                    for u in comp_adj[v]:
                        if colors[u] == c:
                            valid = False
                            break
                    if valid:
                        colors[v] = c
                        res, col = backtrack(idx + 1)
                        if res:
                            return True, col
                        colors[v] = -1
                return False, None
            return backtrack(0)
            
        if best_k == 1:
            # All in one group
            final_colors = [0] * m
        else:
            res, final_colors = find_coloring(best_k)
            if not res:
                # Fallback to greedy coloring if B&B failed (shouldn't happen if is_k_colorable was true)
                # Re-run greedy
                final_colors = [-1] * m
                max_c = 0
                for idx in order:
                    used = set()
                    for neighbor in comp_adj[idx]:
                        if final_colors[neighbor] != -1:
                            used.add(final_colors[neighbor])
                    c = 0
                    while c in used:
                        c += 1
                    final_colors[idx] = c
                    max_c = max(max_c, c)
                    
        # Group masks by color
        color_to_masks = defaultdict(list)
        for i in range(m):
            color_to_masks[final_colors[i]].append(masks[i])
            
        cliques = []
        cliques.append(list(range(k)))
        
        for color, mask_list in color_to_masks.items():
            # The intersection of all masks in mask_list is non-empty (by construction of coloring)
            # We need to find the common vertices.
            common_mask = mask_list[0]
            for mask in mask_list[1:]:
                common_mask &= mask
            
            # Get vertices
            clique_verts = [u for u in range(k) if (common_mask >> u) & 1]
            
            # Get independent vertices for these masks
            indep_verts = []
            for mask in mask_list:
                indep_verts.extend(shadow_groups[mask])
                
            clique = clique_verts + indep_verts
            if len(clique) >= 2:
                cliques.append(clique)
                
        return cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline:
            # Fallback
            n, k, p, seed = spec
            edges_cc, edges_ci = instance(n, k, p, seed)
            cliques = [list(range(k))]
            for u, v in edges_ci:
                cliques.append([u, v])
            results.append(cliques)
            continue
            
        n, k, p, seed = spec
        edges_cc, edges_ci = instance(n, k, p, seed)
        
        cliques = solve_split_graph(n, k, edges_ci)
        
        results.append(cliques)
        
    return results
# EVOLVE-BLOCK-END
