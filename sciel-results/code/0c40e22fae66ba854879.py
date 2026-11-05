import random
import time

def construct():
    start_time = time.time()
    timeout = 140  # Safety margin before 150s limit
    
    n = 20
    # adj[i] is a bitmask of neighbors
    adj = [0] * n
    
    # Precompute all triples for faster checking
    triples = [(i, j, k) for i in range(n) for j in range(i+1, n) for k in range(j+1, n)]
    
    # Helper to check if graph is K3,3-free
    # Returns True if free, or (False, violation_info)
    def is_k33_free(adj):
        for (i, j, k) in triples:
            # Common neighbors of i, j, k
            common = adj[i] & adj[j] & adj[k]
            # Exclude i, j, k themselves
            mask = (1 << i) | (1 << j) | (1 << k)
            valid_common = common & ~mask
            # If there are at least 3 such vertices, they form a K3,3 with i,j,k
            if bin(valid_common).count('1') >= 3:
                return False, (i, j, k), valid_common
        return True, None, None

    # Helper to get number of edges
    def count_edges(adj):
        total = 0
        for i in range(n):
            total += bin(adj[i]).count('1')
        return total // 2

    # Initialize with a known valid dense graph
    # Start with a random graph and clean it up, or start from a structured graph
    # Let's try starting with a complete bipartite graph K10,10 which is K3,3-free
    # Wait, K10,10 contains K3,3. We need to be careful.
    # A random graph with p=0.5 is a good start.
    
    random.seed(42)
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    best_edge_count = 0
    best_adj = None

    # Phase 1: Clean the graph
    # We remove edges until it is K3,3-free
    # Use a smarter removal strategy: remove the edge that is in the most violations?
    # For now, simple random removal is okay, but let's try to be more targeted.
    
    # To speed up cleaning, we can count how many times each edge participates in a violation.
    # But first, let's just get a valid graph.
    
    while time.time() - start_time < timeout - 20:
        is_free, viol_triple, viol_common = is_k33_free(adj)
        if is_free:
            break
        
        # Collect all violations to make a better choice?
        # For now, pick one.
        v_list = [x for x in range(n) if (viol_common >> x) & 1]
        u, v, w = viol_triple
        u_list = [u, v, w]
        
        # Pick random u and v to remove
        ru = random.choice(u_list)
        rv = random.choice(v_list)
        
        # Remove edge
        adj[ru] &= ~(1 << rv)
        adj[rv] &= ~(1 << ru)
        
    # Ensure validity
    while True:
        is_free, viol_triple, viol_common = is_k33_free(adj)
        if is_free:
            break
        v_list = [x for x in range(n) if (viol_common >> x) & 1]
        u, v, w = viol_triple
        u_list = [u, v, w]
        ru = random.choice(u_list)
        rv = random.choice(v_list)
        adj[ru] &= ~(1 << rv)
        adj[rv] &= ~(1 << ru)

    current_edges = count_edges(adj)
    best_edge_count = current_edges
    best_adj = [a for a in adj]

    # Phase 2: Improvement (Add edges)
    # Try to add random missing edges
    # Precompute list of all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Local search: try adding edges, if it creates a violation, try to fix it by removing an edge
    # This is a local move: add (u,v), if invalid, find a violation involving (u,v) and remove an edge from it.
    
    # To make this faster, we need to quickly find if adding (u,v) creates a violation.
    # Adding (u,v) only affects triples containing u or v.
    # Let's precompute triples containing each vertex?
    # Or just check all triples? n=20, triples ~ 1140. It's fast enough.
    
    # Strategy:
    # 1. Iterate through missing edges.
    # 2. Try to add.
    # 3. If valid, keep.
    # 4. If invalid, identify the specific violation caused by the new edge.
    #    The new edge (u,v) is in the graph.
    #    A violation involving (u,v) must have (u,v) as part of the K3,3.
    #    This means either:
    #    a) u and v are in the same part of the bipartition (size 3), and there are 3 common neighbors in the other part.
    #    b) u and v are in different parts? No, an edge connects vertices.
    #    In a K3,3, edges only exist between the two parts.
    #    So if we add edge (u,v), it can only be part of a K3,3 if u and v are in different parts?
    #    Wait, K3,3 has parts A={a1,a2,a3} and B={b1,b2,b3}. Edges are ai-bj.
    #    If we add edge (u,v), it connects u and v.
    #    Case 1: u in A, v in B. Then we need 2 other vertices in A (neighbors of v) and 2 other vertices in B (neighbors of u).
    #    Case 2: u in A, v in A? No, no edges within A.
    #    So u and v must be in different parts.
    #    This implies u and v are NOT common neighbors of each other?
    #    Actually, the check is:
    #    For a triple (i,j,k), if they have 3 common neighbors, it's a K3,3.
    #    If we add (u,v), the only new triples that can become invalid are those where {u,v} is a subset of the 3 vertices in one part, OR {u,v} is a subset of the 3 vertices in the other part?
    #    No, the vertices in the triple are the ones in ONE part.
    #    So if we add (u,v), and u,v end up in the same part (say A), then we need 3 common neighbors in B.
    #    But u and v are in A, so they are not neighbors of each other in the bipartite sense?
    #    Wait, in the graph, u and v ARE neighbors now.
    #    But in the K3,3 subgraph, vertices in the same part are NOT adjacent.
    #    So if we add edge (u,v), u and v cannot be in the same part of the K3,3.
    #    Therefore, u and v must be in DIFFERENT parts.
    #    Let A be the part containing u, and B be the part containing v.
    #    Then we need 2 other vertices in A (neighbors of v) and 2 other vertices in B (neighbors of u).
    #    Let N(v) \ {u} be the neighbors of v. Let N(u) \ {v} be the neighbors of u.
    #    We need to find 2 vertices a1, a2 in N(v) \ {u} and 2 vertices b1, b2 in N(u) \ {v}
    #    such that all edges between {a1,a2} and {b1,b2} exist.
    #    And also edges from u to b1,b2 (already there) and v to a1,a2 (already there).
    #    And u-v (just added).
    #    So the condition is:
    #    There exist a1, a2 in N(v) \ {u} and b1, b2 in N(u) \ {v} such that
    #    a1-b1, a1-b2, a2-b1, a2-b2 all exist.
    #    This is equivalent to finding a K2,2 between N(v)\{u} and N(u)\{v}.
    
    # This is a more targeted check.
    # Let's implement this targeted check for speed.
    
    def can_add_edge(u, v):
        # Returns True if adding (u,v) keeps graph K3,3-free
        # If it creates a violation, return the violating edges to remove?
        # For now, just return False.
        
        # Get neighbors of u (excluding v) and v (excluding u)
        nu = [x for x in range(n) if x != v and (adj[u] >> x) & 1]
        nv = [x for x in range(n) if x != u and (adj[v] >> x) & 1]
        
        # We need to check if there is a K2,2 between nu and nv
        # i.e., exist a1, a2 in nu and b1, b2 in nv such that all 4 cross edges exist.
        # This is O(|nu|^2 * |nv|^2). Max 18^2 * 18^2 ~ 100k. Might be slow if done for every edge.
        # But we only do this for missing edges.
        
        # Optimization:
        # For each pair a1, a2 in nu, check if they have 2 common neighbors in nv.
        # Common neighbors of a1 and a2 in nv:
        # common = adj[a1] & adj[a2] & mask(nv)
        # If count >= 2, then we found a K2,2.
        
        # Precompute mask for nv?
        mask_nv = 0
        for x in nv:
            mask_nv |= (1 << x)
            
        for i in range(len(nu)):
            a1 = nu[i]
            for j in range(i+1, len(nu)):
                a2 = nu[j]
                common = adj[a1] & adj[a2] & mask_nv
                if bin(common).count('1') >= 2:
                    return False
        return True

    # Wait, is this sufficient?
    # The K3,3 formed by adding (u,v) must have u and v in different parts.
    # Part A contains u. Part B contains v.
    # Vertices in A: u, a1, a2.
    # Vertices in B: v, b1, b2.
    # Edges required:
    # u-v (added)
    # u-b1, u-b2 (u in A, b in B) -> b1, b2 must be neighbors of u.
    # v-a1, v-a2 (v in B, a in A) -> a1, a2 must be neighbors of v.
    # a1-b1, a1-b2, a2-b1, a2-b2 (cross edges)
    
    # So yes, the condition is exactly:
    # There exist a1, a2 in N(v) and b1, b2 in N(u) such that all cross edges exist.
    # Note: a1, a2 are neighbors of v. b1, b2 are neighbors of u.
    # And we need edges between {a1,a2} and {b1,b2}.
    
    # This check is correct.
    
    # However, it's possible that adding (u,v) creates a K3,3 where u and v are NOT in the bipartition parts?
    # No, every edge in a K3,3 connects the two parts. So any edge added must be a cross edge.
    # So u and v must be in different parts.
    
    # What if the K3,3 doesn't use the new edge (u,v)?
    # Then the graph was already invalid. But we start with a valid graph and only add edges.
    # So any new violation MUST involve the new edge.
    
    # So this check is sufficient.
    
    # Let's refine the loop.
    
    # To speed up can_add_edge, we can precompute something?
    # For now, let's just use it.
    
    # We also need a repair mechanism.
    # If can_add_edge returns False, we can try to remove an edge from the K2,2 found.
    # But finding the specific K2,2 is part of the check.
    # Let's modify can_add_edge to return the violation if it exists.
    
    def check_add(u, v):
        nu = [x for x in range(n) if x != v and (adj[u] >> x) & 1]
        nv = [x for x in range(n) if x != u and (adj[v] >> x) & 1]
        
        mask_nv = 0
        for x in nv:
            mask_nv |= (1 << x)
            
        for i in range(len(nu)):
            a1 = nu[i]
            for j in range(i+1, len(nu)):
                a2 = nu[j]
                common = adj[a1] & adj[a2] & mask_nv
                if bin(common).count('1') >= 2:
                    # Found violation
                    # Pick two from common
                    common_list = [x for x in range(n) if (common >> x) & 1]
                    b1 = common_list[0]
                    b2 = common_list[1]
                    # Violation involves edges:
                    # (a1,b1), (a1,b2), (a2,b1), (a2,b2)
                    # And (u,v), (u,b1), (u,b2), (v,a1), (v,a2)
                    # We can remove one of the cross edges, e.g., (a1,b1)
                    return False, (a1, b1)
        return True, None

    # Main loop
    # Try to add edges in a random order
    # If successful, keep.
    # If failed, try to repair by removing the suggested edge.
    # If repair works (i.e., graph becomes valid again), keep the add and the removal.
    # This is a net change of 0 edges, but might allow future adds.
    # Actually, if we add (u,v) and remove (a1,b1), the edge count stays same.
    # But the structure changes.
    
    # Let's try a simpler approach first:
    # Just try to add edges. If it fails, skip.
    # This is what the current code does, but with a faster check.
    
    # Let's also try random restarts if stuck?
    # Or simulated annealing?
    
    # For now, let's stick to the improved check.
    
    # We need to be careful with time.
    
    while time.time() - start_time < timeout:
        # Shuffle edges
        random.shuffle(all_edges)
        improved = False
        
        for (u, v) in all_edges:
            # Check if edge exists
            if (adj[u] >> v) & 1:
                continue
            
            # Tentatively add
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            # Check validity using fast check
            # But wait, the fast check only checks violations INVOLVING (u,v).
            # Is it possible that adding (u,v) makes an OLD triple invalid without involving (u,v)?
            # No, because the graph was valid before.
            # Any new K3,3 must use at least one new edge.
            # Since we only added one edge, it must use (u,v).
            
            ok, repair_edge = check_add(u, v)
            
            if ok:
                # Keep the edge
                current_edges += 1
                if current_edges > best_edge_count:
                    best_edge_count = current_edges
                    best_adj = [a for a in adj]
                improved = True
                # Continue scanning?
                # Yes, adding an edge might allow others.
            else:
                # Revert
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                
                # Try to repair?
                # If we revert, we are back to valid state.
                # We could try to remove repair_edge from the graph (if it exists) and then add (u,v).
                # But we already reverted.
                # Let's try a "swap" move:
                # 1. Add (u,v)
                # 2. Remove (a,b) where (a,b) is from the violation.
                # 3. Check if valid.
                # If valid, we have a net 0 change but different structure.
                # This might help escape local optima.
                
                # Let's implement this swap.
                if repair_edge:
                    a, b = repair_edge
                    # Check if (a,b) is an edge
                    if (adj[a] >> b) & 1:
                        # Perform swap
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        adj[a] &= ~(1 << b)
                        adj[b] &= ~(1 << a)
                        
                        # Check if valid
                        # We need to check if the swap introduced any OTHER violation.
                        # The swap removed (a,b) and added (u,v).
                        # Violations involving (u,v) are fixed by removing (a,b)?
                        # The violation was a K3,3 involving (u,v) and (a,b) was one of the edges.
                        # Removing (a,b) breaks that specific K3,3.
                        # But could it create a new one? No, removing edges doesn't create K3,3.
                        # Could it leave another K3,3 involving (u,v)?
                        # Possibly, if there were multiple K2,2s.
                        # So we should check full validity.
                        
                        is_free, _, _ = is_k33_free(adj)
                        if is_free:
                            # Swap was successful
                            # Edge count unchanged, but structure improved?
                            # We don't count this as "improved" for edge count, but it might enable future adds.
                            # Let's mark it as improved to continue the loop.
                            improved = True
                        else:
                            # Revert swap
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                            adj[a] |= (1 << b)
                            adj[b] |= (1 << a)
                
        if not improved:
            # No improvements in this pass.
            # Try random restart?
            # Or just break if we are close to timeout.
            if time.time() - start_time > timeout - 5:
                break
            # Otherwise, continue with a new shuffle
            # Maybe perturb the graph?
            # Let's try a few random edge removals to escape local optimum.
            for _ in range(5):
                u, v = random.choice(all_edges)
                if (adj[u] >> v) & 1:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    current_edges -= 1
            # Then try to add back.
            # This is a simple perturbation.

    # Convert best_adj to list of edges
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
                
    return edges

