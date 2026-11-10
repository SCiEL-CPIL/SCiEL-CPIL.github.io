import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # Precompute permutations of 4 elements
    perms = list(itertools.permutations(range(4)))
    
    # For each permutation, precompute the list of (i, j) pairs that MUST be 1 in M
    # M is 4x4. Q3 edges are (i, j) where j != perm[i].
    # So for a perm p, required edges are (i, j) for all i, j != p[i].
    required_edges_by_perm = []
    for p in perms:
        edges = []
        for i in range(4):
            for j in range(4):
                if j != p[i]:
                    edges.append((i, j))
        required_edges_by_perm.append(edges)

    def has_q3(adj):
        # adj is a list of 16 integers, each a 16-bit mask
        # Returns True if Q3 is found, else False
        # To speed up, we can early exit.
        
        # We need to check all 8-subsets.
        # To make it faster, we can precompute list of 8-subsets.
        # But creating the list every time is slow.
        # However, 12870 is small.
        
        # Optimization: 
        # A Q3 is bipartite. 
        # If the graph is not bipartite, it might still contain Q3.
        # But Q3 itself is bipartite.
        
        # Let's iterate over 8-subsets.
        # To avoid generating combinations repeatedly, we can generate them once outside?
        # No, the graph changes. The subsets of vertices are fixed.
        
        # We'll generate the 8-subsets once in the outer scope?
        # No, let's just use itertools.combinations inside, it's C-implemented.
        
        for subset in itertools.combinations(range(16), 8):
            # Map subset vertices to 0..7
            # We need to check partitions of these 8 vertices into A (4) and B (4).
            # There are 70 partitions.
            
            # To check a partition (A, B):
            # We form the 4x4 matrix M where M[i][j] = 1 if edge(subset[A[i]], subset[B[j]]) exists.
            # Then check if any perm condition is met.
            
            # Optimization: 
            # Instead of iterating all 70 partitions, we can fix the first 4 vertices as A?
            # No, A can be any 4.
            
            # Let's iterate over 4-subsets of the 8.
            # There are 70.
            # For each 4-subset A, B is the complement.
            
            # To speed up matrix construction and check:
            # We can precompute the neighbors of each vertex in the subset.
            
            verts = list(subset)
            
            # For each of the 70 4-subsets A of verts
            for A in itertools.combinations(verts, 4):
                B = tuple(v for v in verts if v not in A)
                
                # Build 4x4 matrix as list of 4 integers (bitmasks for rows)
                # Row i corresponds to A[i]. Bit j corresponds to B[j].
                rows = []
                for i in range(4):
                    u = A[i]
                    mask = 0
                    for j in range(4):
                        v = B[j]
                        if adj[u] & (1 << v):
                            mask |= (1 << j)
                    rows.append(mask)
                
                # Check against 24 perms
                # For a perm p, we need M[i][j] == 1 for all j != p[i].
                # This is equivalent to: 
                # For each i, the bits in rows[i] must cover all bits except p[i].
                # i.e., rows[i] must have 1s in positions {0,1,2,3} \ {p[i]}.
                # This means rows[i] & (~ (1 << p[i])) must equal (~ (1 << p[i])) & 15 ?
                # No, rows[i] might have extra 1s? No, M is binary.
                # If M[i][j] is 1, it's 1.
                # We require M[i][j] == 1 for all j != p[i].
                # So the set of 1s in row i must be a superset of {0,1,2,3} \ {p[i]}.
                # Since the row is 4 bits, and we require 3 specific bits to be 1,
                # the row mask must be equal to (15 ^ (1 << p[i])) OR (1 << p[i])?
                # No, the bit at p[i] can be 0 or 1.
                # So we just need: (rows[i] & (15 ^ (1 << p[i]))) == (15 ^ (1 << p[i]))
                
                is_q3 = False
                for p in perms:
                    ok = True
                    for i in range(4):
                        # Required bits: all except p[i]
                        req_mask = 15 ^ (1 << p[i])
                        if (rows[i] & req_mask) != req_mask:
                            ok = False
                            break
                    if ok:
                        is_q3 = True
                        break
                
                if is_q3:
                    return True
                    
        return False

    def remove_edge_from_q3(adj, edge_to_remove):
        # This function is not used directly in the new logic, 
        # we just pick a random edge to remove if Q3 is found.
        pass

    best_edges = 0
    best_adj = None
    
    # Initialize with a dense random graph
    # Start with p=0.9
    adj = [0] * 16
    for u in range(16):
        for v in range(u+1, 16):
            if random.random() < 0.9:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    
    # Local Search
    for restart in range(20):
        if time.time() > deadline:
            break
            
        # If not first restart, randomize
        if restart > 0:
             adj = [0] * 16
             for u in range(16):
                 for v in range(u+1, 16):
                     if random.random() < 0.9:
                         adj[u] |= (1 << v)
                         adj[v] |= (1 << u)
        
        # Phase 1: Remove edges to break Q3
        changed = True
        while changed and time.time() < deadline:
            changed = False
            if has_q3(adj):
                # Find a Q3 and remove a random edge from it?
                # has_q3 doesn't return the Q3.
                # We need a version that returns an edge to remove.
                # Or just remove a random edge from the graph? 
                # That's inefficient.
                # Let's modify has_q3 to return an edge.
                
                # Actually, let's just pick a random edge and remove it?
                # No, that's too slow.
                
                # Let's create a helper that finds a Q3 and returns an edge.
                
                # To keep the code simple and within limits, 
                # let's assume has_q3 is fast enough to just check.
                # But we need to remove an edge.
                
                # Let's modify the inner loop to find an edge.
                
                # We'll re-implement the check to return an edge.
                
                edge_removed = False
                # We need to find a specific Q3.
                # Let's just remove a random edge from the densest part?
                # No, let's just remove a random edge from the graph.
                # This is a heuristic.
                
                # Better: Modify has_q3 to yield an edge.
                # For now, let's just remove a random edge and re-check.
                # This might be slow if many Q3s exist.
                
                # Let's pick a random edge (u, v) present in adj.
                edges_present = []
                for u in range(16):
                    for v in range(u+1, 16):
                        if adj[u] & (1 << v):
                            edges_present.append((u, v))
                
                if edges_present:
                    u, v = random.choice(edges_present)
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                    changed = True
                    # Re-check in next iteration
                
        # Phase 2: Add edges
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to add each missing edge
            # Shuffle order to avoid bias
            missing_edges = []
            for u in range(16):
                for v in range(u+1, 16):
                    if not (adj[u] & (1 << v)):
                        missing_edges.append((u, v))
            
            random.shuffle(missing_edges)
            
            for u, v in missing_edges:
                if time.time() > deadline:
                    break
                # Add edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
                if has_q3(adj):
                    # Remove it
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                else:
                    improved = True
        
        # Update best
        current_edges = sum(bin(adj[u]).count('1') for u in range(16)) // 2
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = adj[:]
            
    # Convert best_adj to list of edges
    if best_adj is None:
        # Fallback
        best_adj = [0] * 16
        for u in range(16):
            for v in range(u+1, 16):
                best_adj[u] |= (1 << v)
                best_adj[v] |= (1 << u)
        # This fallback is K16, which contains Q3. 
        # We need a valid fallback.
        # Let's return a sparse graph that is definitely Q3-free.
        # e.g., a bipartite graph K8,8? K8,8 contains Q3? 
        # Q3 is bipartite 4+4. K8,8 contains any bipartite graph on 8 vertices.
        # So K8,8 contains Q3.
        # We need a graph with no Q3.
        # A random graph with p=0.1 is likely Q3-free.
        # Or just return an empty graph? Score 0.
        # Let's try to return the last valid state if best_adj is None?
        # The loop should find at least one.
        
        # If we still have None, return empty list.
        return []

    edges = []
    for u in range(16):
        for v in range(u+1, 16):
            if best_adj[u] & (1 << v):
                edges.append([u, v])
                
    return edges
