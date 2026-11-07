"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    import heapq

    def instance(splits, seed, palette=5):
        rng = Random(seed)
        faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
        edges = {(min(a, b), max(a, b)) for f in faces for a, b in zip(f, f[1:] + f[:1])}
        n = 8
        for _ in range(splits):
            face = faces.pop(rng.randrange(len(faces)))
            inner = list(range(n, n + 4))
            n += 4
            for i in range(4):
                a, b = face[i], face[(i + 1) % 4]
                edges.add((min(a, inner[i]), max(a, inner[i])))
                edges.add((min(inner[i], inner[(i + 1) % 4]), max(inner[i], inner[(i + 1) % 4])))
                faces.append([a, b, inner[(i + 1) % 4], inner[i]])
            faces.append(inner)
        lists = [sorted(rng.sample(range(palette), 3)) for _ in range(n)]
        return n, sorted(edges), lists

    def color_instance(n, edges, lists):
        # Build adjacency
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Convert to sets for faster lookup
        adj_set = [set(neigh) for neigh in adj]
        
        # We need to find a valid coloring where each vertex v takes a color from lists[v]
        # and adjacent vertices have different colors.
        
        # Strategy: Backtracking with forward checking and MRV (Minimum Remaining Values) heuristic.
        # Since the graphs are planar bipartite quadrangulations with lists of size 3, they are 3-list-colorable.
        # The constraint graph is sparse.
        
        # Initial domain for each vertex is its list.
        domain = [list(l) for l in lists]
        
        # Check if any domain is empty initially
        for i in range(n):
            if not domain[i]:
                return None
        
        # Heuristic: Order vertices by degree (descending) or by domain size (ascending).
        # Let's use a dynamic MRV approach with a priority queue.
        # However, simple backtracking with a good static order might suffice for these sizes.
        # Let's try a static order based on degree first, as it's simpler and often effective for sparse graphs.
        # Or better, use a recursive backtracking with MRV.
        
        # Given n can be up to 8 + 4*100 = 408, backtracking might be risky if not guided well.
        # But these are bipartite and planar.
        # Let's use a standard backtracking with MRV and forward checking.
        
        # To optimize, we can also use bitmasks for domains since colors are 0..4.
        # But lists are small (size 3).
        
        # Let's implement a recursive solver.
        # State: assignment array, domains.
        
        # To avoid deep recursion limits (though 408 is fine), we'll use standard recursion.
        
        # Pre-compute neighbors
        # We will maintain:
        #   assignment: list of -1 or color
        #   domains: list of lists of available colors
        
        assignment = [-1] * n
        # Copy domains
        doms = [list(d) for d in domain]
        
        # To speed up, we can pre-compute a static order? No, MRV is dynamic.
        # Let's use a helper to pick the next vertex.
        
        def get_mrv_vertex(unassigned):
            min_len = 10
            best_v = -1
            for v in unassigned:
                if len(doms[v]) < min_len:
                    min_len = len(doms[v])
                    best_v = v
                    if min_len == 0:
                        break
            return best_v

        # For efficiency, let's just iterate through all unassigned vertices each time?
        # n is ~400. Backtracking depth is 400. 400*400 = 160k operations per level? No, just to find MRV.
        # It might be slow if we do it naively.
        
        # Alternative: Since the graph is bipartite, we can try to color one part then the other?
        # No, list coloring doesn't decouple so simply.
        
        # Let's try a simpler greedy first, and if it fails, fall back to backtracking.
        # Actually, the prompt says "all are colourable by the Alon-Tarsi theorem".
        # A simple greedy might fail.
        
        # Let's implement a robust backtracking with MRV.
        # Optimization: Use a list of unassigned vertices.
        
        unassigned = list(range(n))
        
        def backtrack():
            if not unassigned:
                return True
            
            # Find MRV vertex
            # To make this faster, we can just pick the first vertex with smallest domain.
            # Since n is small, linear scan is O(n).
            mrv_v = -1
            min_size = 10
            for v in unassigned:
                size = len(doms[v])
                if size < min_size:
                    min_size = size
                    mrv_v = v
                    if size == 0:
                        break
            
            if min_size == 0:
                return False
            
            # Try each color in the domain of mrv_v
            # Heuristic: try colors that conflict least? Or just iterate.
            # Let's iterate.
            # Save state
            saved_doms = [None] * n
            # We only need to save the domains of neighbors of mrv_v that are unassigned
            # Actually, we modify doms[neighbor] by removing the color.
            # We need to restore them.
            
            # Better approach:
            # 1. Remove mrv_v from unassigned
            # 2. For each color c in doms[mrv_v]:
            #    a. Assign assignment[mrv_v] = c
            #    b. For each neighbor u of mrv_v in unassigned:
            #       if c in doms[u], remove it. Remember to restore.
            #    c. Recurse
            #    d. Restore
            
            # To avoid copying entire doms list, we track changes.
            changes = []
            
            # Remove mrv_v from unassigned
            # We'll use a set for unassigned for O(1) removal? Or list and swap?
            # List removal is O(n). n=400. Acceptable.
            
            # Let's use a boolean array for assigned status to speed up checks?
            # Or just keep the list and remove.
            
            # Find index of mrv_v in unassigned
            idx = unassigned.index(mrv_v)
            unassigned[idx] = unassigned[-1]
            unassigned.pop()
            
            for c in doms[mrv_v]:
                assignment[mrv_v] = c
                
                # Forward checking
                valid = True
                for u in adj_set[mrv_v]:
                    if u in unassigned: # Check if u is still unassigned
                        # Wait, 'u in unassigned' is O(n).
                        # Better to maintain a set of unassigned vertices.
                        pass
                
                # Let's switch to a set for unassigned to make membership checks O(1).
                # I'll refactor to use a set.
                # Actually, let's just use the list and be careful.
                # Given the constraints, a set is better.
                
                # Let's restart the function with a set for unassigned.
                return False # Placeholder, logic needs restructuring
                
            # Restore
            unassigned.append(mrv_v)
            # Note: The order in unassigned doesn't strictly matter for MRV, but it affects which vertex is picked if ties.
            
            return False

        # The above sketch is getting messy. Let's write a clean recursive backtracking.
        
        def solve():
            # Use a set for unassigned vertices for O(1) lookup
            unassigned_set = set(range(n))
            
            # Recursive helper
            def bt():
                if not unassigned_set:
                    return True
                
                # Find MRV vertex
                mrv_v = None
                min_size = 10
                for v in unassigned_set:
                    size = len(doms[v])
                    if size < min_size:
                        min_size = size
                        mrv_v = v
                        if size == 0:
                            break
                
                if min_size == 0:
                    return False
                
                # Save the domain of mrv_v? No, we iterate over it.
                # We need to save the state of neighbors' domains.
                # List of (neighbor, color_removed)
                changes = []
                
                # Remove mrv_v from unassigned_set
                unassigned_set.remove(mrv_v)
                
                # Try each color
                for c in doms[mrv_v]:
                    assignment[mrv_v] = c
                    # Apply forward checking
                    for u in adj_set[mrv_v]:
                        if u in unassigned_set:
                            if c in doms[u]:
                                doms[u].remove(c)
                                changes.append(u)
                                if not doms[u]:
                                    # Conflict
                                    break
                    else:
                        # No conflict in the loop, or loop finished normally?
                        # If break was hit, we need to handle it.
                        # Using for-else is tricky with break inside.
                        # Let's use a flag.
                        pass
                    
                    # Check if any neighbor became empty
                    conflict = False
                    for u in changes:
                        if not doms[u]:
                            conflict = True
                            break
                    
                    if not conflict:
                        if bt():
                            return True
                    else:
                        # Just restore and continue to next color
                        pass
                    
                    # Restore changes
                    for u in changes:
                        doms[u].append(c) # Note: order might change, but we use sets? No, lists.
                        # Wait, if we remove and append, the list order changes.
                        # It's better to keep domains as sets or sorted lists.
                        # Let's use sets for domains to make add/remove O(1) and avoid order issues.
                    
                    changes = []
                
                # Restore mrv_v to unassigned_set
                unassigned_set.add(mrv_v)
                
                return False
            
            return bt()

        # Let's refactor to use Sets for domains to make it cleaner and faster.
        
        doms_set = [set(l) for l in lists]
        
        # Check initial emptiness
        for i in range(n):
            if not doms_set[i]:
                return None
            
        assignment = [-1] * n
        unassigned_set = set(range(n))
        
        def bt():
            if not unassigned_set:
                return True
            
            # Find MRV
            mrv_v = None
            min_size = 10
            for v in unassigned_set:
                size = len(doms_set[v])
                if size < min_size:
                    min_size = size
                    mrv_v = v
                    if size == 0:
                        break
            
            if min_size == 0:
                return False
            
            unassigned_set.remove(mrv_v)
            
            # Try colors
            for c in list(doms_set[mrv_v]):
                assignment[mrv_v] = c
                doms_set[mrv_v].remove(c) # Remove from its own domain? Not strictly necessary if we don't revisit, but good for consistency if we did.
                
                changes = []
                conflict = False
                for u in adj_set[mrv_v]:
                    if u in unassigned_set:
                        if c in doms_set[u]:
                            doms_set[u].discard(c)
                            changes.append(u)
                            if not doms_set[u]:
                                conflict = True
                                break
                
                if not conflict:
                    if bt():
                        return True
                
                # Restore
                for u in changes:
                    doms_set[u].add(c)
                doms_set[mrv_v].add(c)
            
            unassigned_set.add(mrv_v)
            return False
        
        if bt():
            # Convert to Python ints
            return [int(c) for c in assignment]
        else:
            return None

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        coloring = color_instance(n, edges, lists)
        if coloring is None:
            # Should not happen per problem statement, but handle gracefully
            # Return a dummy list to maintain structure, though it will score 0
            coloring = [0] * n
        # Ensure types are Python int
        out.append([int(c) for c in coloring])
    return out
# EVOLVE-BLOCK-END
