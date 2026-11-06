import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    def build_adj(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def edges_from_adj(adj, n):
        edges = []
        for i in range(n):
            mask = adj[i]
            # Only consider j > i to avoid duplicates
            mask &= ~((1 << (i + 1)) - 1)
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                edges.append([i, j])
                mask ^= lsb
        return edges

    def check_k4(adj, n):
        """Returns a tuple (a, b, c, d) if a K4 is found, else None."""
        for a in range(n):
            mask_a = adj[a]
            # Iterate neighbors b > a
            mask_b = mask_a & ~((1 << (a + 1)) - 1)
            while mask_b:
                lsb_b = mask_b & -mask_b
                b = lsb_b.bit_length() - 1
                mask_b ^= lsb_b
                
                # Common neighbors of a and b, with index > b
                common = mask_a & adj[b]
                common &= ~((1 << (b + 1)) - 1) # c > b
                common &= ~(1 << a) # a is not in common neighbors in a simple graph context for this check? 
                # Actually, adj[a] has b, adj[b] has a. common has both. We remove a and b.
                common &= ~(1 << a)
                common &= ~(1 << b)
                
                # Now find an edge (c, d) in common
                # We need c < d.
                # Iterate c in common
                mask_c = common
                while mask_c:
                    lsb_c = mask_c & -mask_c
                    c = lsb_c.bit_length() - 1
                    mask_c ^= lsb_c
                    
                    # Check if any neighbor of c in common is > c
                    # Neighbors of c in common: adj[c] & common
                    # We need d > c.
                    candidates = adj[c] & common
                    candidates &= ~((1 << (c + 1)) - 1)
                    if candidates:
                        # Found K4: a, b, c, d
                        # d is the lowest set bit in candidates
                        lsb_d = candidates & -candidates
                        d = lsb_d.bit_length() - 1
                        return (a, b, c, d)
        return None

    def check_ind6(adj, n):
        """Returns a tuple of 6 vertices if an independent set is found, else None."""
        # Independent set in G is a clique in G_complement.
        # Let's work with the complement adjacency masks.
        # comp_adj[i] has bit j set if (i, j) is NOT an edge in G (and i != j).
        comp_adj = []
        for i in range(n):
            # All bits 1 to n-1, except i, except adj[i]
            full_mask = (1 << n) - 1
            full_mask &= ~(1 << i)
            comp_adj.append(full_mask ^ adj[i])
            
        # Find a clique of size 6 in the complement graph.
        # Backtracking search for clique of size 6.
        # To make it fast, we can use a heuristic: pick a vertex, recurse on its neighbors.
        # Since we only need to find ONE, we can be aggressive.
        
        def find_clique(vertices, size):
            if size == 0:
                return []
            if not vertices:
                return None
            
            # Heuristic: pick a vertex with high degree in the induced subgraph?
            # Or just pick the first one.
            # Let's pick a random vertex to diversify? No, deterministic is fine for check.
            # Actually, for speed, we can just iterate.
            
            # Optimization: If the number of vertices < size, return None.
            if len(vertices) < size:
                return None
                
            # Pick a pivot vertex
            # To speed up, we can order vertices by degree in the complement subgraph.
            # But for n=36, a simple recursive search might be okay if we prune well.
            
            # Let's try a simple recursive backtracking.
            # We maintain a list of candidate vertices.
            
            # To avoid deep recursion overhead in Python, we might need to be careful.
            # But n=36 is small.
            
            # Let's try picking the vertex with the most neighbors in the current candidate set.
            # This is expensive to compute every time.
            # Let's just pick the first vertex.
            
            v = vertices[0]
            rest = vertices[1:]
            
            # Neighbors of v in the complement, restricted to rest
            neighbors = comp_adj[v] & 0
            for u in rest:
                if (comp_adj[v] >> u) & 1:
                    neighbors |= (1 << u)
            
            # Convert neighbors mask to list for recursion? 
            # It's better to pass a list or mask.
            # Let's pass a list of candidate vertices.
            
            next_candidates = [u for u in rest if (comp_adj[v] >> u) & 1]
            
            res = find_clique(next_candidates, size - 1)
            if res is not None:
                return [v] + res
            
            # If not found with v, try without v?
            # This is standard clique finding.
            # But for n=36, this might be too slow if called frequently in local search.
            
            # Alternative: Use a randomized check for Ind6 in the local search loop, 
            # and a rigorous check only when accepting a solution.
            # But we need to fix violations. If we don't find the exact Ind6, we can't fix it.
            # However, we can fix "potential" Ind6s by adding edges.
            
            # Let's return None if the simple path fails, and rely on the caller to handle it.
            # Or, we can implement a faster bitset-based clique finder.
            
            return None

        # The above recursive function is too slow for a local search step.
        # We need a faster heuristic for Ind6 detection in the loop.
        
        # Heuristic: Random sampling.
        # If we find an Ind6, we return it.
        # If we don't, we assume there isn't one (risk of false negative).
        # But for the final verification, we need to be exact.
        
        # Let's implement a fast exact check for n <= 40 using bitsets and backtracking with pruning.
        # We'll use a global flag to stop early.
        
        found = [None]
        
        def backtrack(candidates, current_set):
            if found[0]:
                return
            if len(current_set) == 6:
                found[0] = tuple(current_set)
                return
            if not candidates:
                return
                
            # Pruning: if len(candidates) + len(current_set) < 6, return
            if len(candidates) + len(current_set) < 6:
                return
                
            # Pick a vertex
            # To speed up, we can pick a vertex with high degree in the complement subgraph.
            # Let's just pick the first one for simplicity in this draft, but optimize if needed.
            v = candidates[0]
            rest = candidates[1:]
            
            # Neighbors of v in complement
            next_cands = [u for u in rest if (comp_adj[v] >> u) & 1]
            
            backtrack(next_cands, current_set + [v])
            if found[0]:
                return
            
            # Try without v
            backtrack(rest, current_set)

        # To make this fast, we should order candidates by degree.
        # Let's just use random sampling for the local search step to find a violation.
        # If we find one, we fix it.
        # If we don't find one in K checks, we assume it's valid for that step.
        # At the end, we do a rigorous check.
        
        return None # Placeholder, we will use a different strategy for the loop

    # Strategy:
    # 1. Find a valid graph for n=35 using local search.
    # 2. Try to extend to n=36.
    
    def local_search(n, adj, max_steps=1000):
        """
        Performs local search on adj (list of ints) for n vertices.
        Returns (success, adj).
        """
        for step in range(max_steps):
            if time.time() > deadline - 5.0:
                break
                
            # Check K4
            k4 = check_k4(adj, n)
            if k4:
                a, b, c, d = k4
                # Flip a random edge in this K4
                verts = [a, b, c, d]
                random.shuffle(verts)
                u, v = verts[0], verts[1]
                if (adj[u] >> v) & 1:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                continue
            
            # Check Ind6 (Heuristic: Random Sampling)
            ind6_found = False
            # Try a few random samples
            for _ in range(20):
                # Pick 6 random vertices
                sample = random.sample(range(n), 6)
                # Check if they form an independent set
                is_ind = True
                for i in range(6):
                    for j in range(i + 1, 6):
                        if (adj[sample[i]] >> sample[j]) & 1:
                            is_ind = False
                            break
                    if not is_ind:
                        break
                if is_ind:
                    ind6_found = True
                    # Flip a random edge in this Ind6
                    verts = sample
                    random.shuffle(verts)
                    u, v = verts[0], verts[1]
                    if (adj[u] >> v) & 1:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                    else:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                    break
            
            if ind6_found:
                continue
            
            # If no violations found (heuristically), we consider it valid for now.
            # But we need a rigorous check before accepting.
            # Let's do a rigorous check here.
            # Rigorous K4 check is fast.
            # Rigorous Ind6 check is the bottleneck.
            
            # Let's implement a faster rigorous Ind6 check using bitsets.
            # We'll check if the complement graph has a clique of size 6.
            
            comp_adj = []
            for i in range(n):
                full_mask = (1 << n) - 1
                full_mask &= ~(1 << i)
                comp_adj.append(full_mask ^ adj[i])
            
            # Check for clique of size 6 in comp_adj
            # Use a recursive search with pruning.
            # To make it fast, we limit the search depth or use a heuristic.
            # For n=36, a clique of size 6 is relatively easy to find if it exists.
            # If it doesn't exist, the search might take longer, but we can cap it.
            
            found_clique = [False]
            
            def check_clique(candidates, current):
                if found_clique[0]:
                    return
                if len(current) == 6:
                    found_clique[0] = True
                    return
                if not candidates:
                    return
                if len(candidates) + len(current) < 6:
                    return
                    
                # Pick a vertex
                # Heuristic: Pick the vertex with the most neighbors in candidates
                # This is O(n^2) per step, which might be slow.
                # Let's just pick the first vertex.
                v = candidates[0]
                rest = candidates[1:]
                
                # Neighbors of v in comp_adj, restricted to rest
                next_cands = [u for u in rest if (comp_adj[v] >> u) & 1]
                
                check_clique(next_cands, current + [v])
                if found_clique[0]:
                    return
                
                check_clique(rest, current)

            # To avoid stack overflow or excessive time, we'll just run this.
            # If it's too slow, we'll rely on the heuristic.
            # But for n=35/36, it should be okay.
            
            # Actually, the recursive call above is a bit slow in Python.
            # Let's stick to the heuristic for the loop, and do a rigorous check
            # only when we think we have a solution.
            
            # For now, let's assume if the heuristic didn't find an Ind6 in 20 samples,
            # and K4 is gone, we might have a solution.
            # We will verify it rigorously outside.
            return True, adj
            
        return False, adj

    # Step 1: Find a valid 35-vertex graph
    best_n = 0
    best_edges = []
    
    # Try to find a valid graph for n=35
    target_n = 35
    found = False
    for attempt in range(50):
        if time.time() > deadline - 10.0:
            break
            
        adj = [0] * target_n
        for i in range(target_n):
            for j in range(i + 1, target_n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    
        success, adj = local_search(target_n, adj, max_steps=500)
        
        if success:
            # Rigorous verification
            # K4 check
            if check_k4(adj, target_n) is None:
                # Ind6 check (Rigorous)
                # We need a rigorous Ind6 checker.
                # Let's implement a fast one using bitsets.
                
                def has_ind6(adj, n):
                    comp_adj = []
                    for i in range(n):
                        full_mask = (1 << n) - 1
                        full_mask &= ~(1 << i)
                        comp_adj.append(full_mask ^ adj[i])
                    
                    # Find clique of size 6
                    # Use a simple backtracking with bitmasks for candidates
                    def find_clique(cand_mask, size):
                        if size == 0:
                            return True
                        if not cand_mask:
                            return False
                        if bin(cand_mask).count('1') < size:
                            return False
                            
                        # Pick a vertex
                        # Heuristic: pick vertex with highest degree in cand_mask
                        # For simplicity, pick lowest bit
                        lsb = cand_mask & -cand_mask
                        v = lsb.bit_length() - 1
                        cand_mask ^= lsb
                        
                        # Neighbors of v in cand_mask
                        neighbors = comp_adj[v] & cand_mask
                        
                        if find_clique(neighbors, size - 1):
                            return True
                        
                        # Try without v?
                        # This is a standard subset search.
                        # To prune, we can check if the remaining candidates can form a clique of size 'size'.
                        # This is complex. Let's just do a simple DFS.
                        
                        # Wait, the above only tries one path. We need to try all.
                        # But for n=35, a clique of 6 is either easy to find or doesn't exist.
                        # If it doesn't exist, the search space is large.
                        # Let's use a randomized check for rigorous verification too?
                        # No, "checked exactly".
                        
                        # Let's use a more robust clique finder.
                        # We'll iterate through vertices.
                        return False
                    
                    # The above is incomplete. Let's use a different rigorous check.
                    # Since n is small, we can use itertools for the final check if it's fast enough.
                    # C(35, 6) is ~1.6 million. This is too slow for Python in a tight loop,
                    # but okay for a final check.
                    
                    # Let's do the final check with itertools.
                    import itertools
                    for combo in itertools.combinations(range(n), 6):
                        is_ind = True
                        for i in range(6):
                            for j in range(i + 1, 6):
                                if (adj[combo[i]] >> combo[j]) & 1:
                                    is_ind = False
                                    break
                            if not is_ind:
                                break
                        if is_ind:
                            return True
                    return False

                if not has_ind6(adj, target_n):
                    best_n = target_n
                    best_edges = edges_from_adj(adj, target_n)
                    found = True
                    break

    if not found:
        # Fallback: Try to find a smaller graph, e.g., n=30
        # Or just return a valid small graph.
        # Let's try n=30 quickly.
        target_n = 30
        for attempt in range(10):
            if time.time() > deadline - 5.0:
                break
            adj = [0] * target_n
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            success, adj = local_search(target_n, adj, max_steps=200)
            if success:
                if check_k4(adj, target_n) is None:
                    # Check Ind6
                    import itertools
                    valid = True
                    for combo in itertools.combinations(range(target_n), 6):
                        is_ind = True
                        for i in range(6):
                            for j in range(i + 1, 6):
                                if (adj[combo[i]] >> combo[j]) & 1:
                                    is_ind = False
                                    break
                            if not is_ind:
                                break
                        if is_ind:
                            valid = False
                            break
                    if valid:
                        best_n = target_n
                        best_edges = edges_from_adj(adj, target_n)
                        break

    # Step 2: Try to extend to 36
    if best_n == 35 and time.time() < deadline - 20.0:
        # Take the 35-vertex adj and add vertex 35
        # We need to reconstruct adj from best_edges
        adj35 = build_adj(35, best_edges)
        adj36 = [0] * 36
        for i in range(35):
            adj36[i] = adj35[i]
            
        # Connect vertex 35 to random subset
        for i in range(35):
            if random.random() < 0.5:
                adj36[i] |= (1 << 35)
                adj36[35] |= (1 << i)
                
        # Run local search on 36
        success, adj36 = local_search(36, adj36, max_steps=1000)
        
        if success:
            if check_k4(adj36, 36) is None:
                # Check Ind6
                import itertools
                valid = True
                for combo in itertools.combinations(range(36), 6):
                    is_ind = True
                    for i in range(6):
                        for j in range(i + 1, 6):
                            if (adj36[combo[i]] >> combo[j]) & 1:
                                is_ind = False
                                break
                            if not is_ind:
                                break
                    if is_ind:
                        valid = False
                        break
                if valid:
                    best_n = 36
                    best_edges = edges_from_adj(adj36, 36)

    return {'n': best_n, 'edges': best_edges}
