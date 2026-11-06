import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n = 16
    
    # Precompute Q3 masks for K_4,4
    # Vertices 0-3 in A, 4-7 in B
    # Edge (i, j) where i in [0,3], j in [4,7] maps to bit i*4 + (j-4)
    # Total 16 bits.
    
    def is_q3_mask(mask):
        # Check if the bipartite graph defined by mask is Q3
        # Q3 is 3-regular
        deg_a = [0]*4
        deg_b = [0]*4
        for i in range(4):
            for j in range(4):
                if mask & (1 << (i*4 + j)):
                    deg_a[i] += 1
                    deg_b[j] += 1
        if all(d == 3 for d in deg_a) and all(d == 3 for d in deg_b):
            return True
        return False

    q3_masks = []
    # Iterate over all 12-edge subsets of 16 edges
    # This is C(16, 12) = 1820, which is fast
    all_edges = list(range(16))
    for combo in itertools.combinations(all_edges, 12):
        mask = 0
        for idx in combo:
            mask |= (1 << idx)
        if is_q3_mask(mask):
            q3_masks.append(mask)
            
    # There are 24 such masks? 
    # Let's assume q3_masks is not empty.
    
    # adj[i] is a bitmask of neighbors
    adj = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
            
    def get_edge_count():
        return sum(bin(adj[i]).count("1") for i in range(n)) // 2

    def find_q3_edge():
        """
        Finds an edge that is part of a Q3 subgraph.
        Returns (u, v) or None.
        """
        # Iterate over all 8-vertex subsets
        # Optimization: We can just pick random 8-sets if we want speed, 
        # but for correctness of "maximal", we should scan.
        # 12870 subsets * 35 splits * 24 masks is ~10M ops per full scan.
        # This might be too slow if we do many scans.
        
        # Better: Iterate over 4-sets A and 4-sets B?
        # If we pick A and B, we check if G[A,B] contains Q3.
        # Number of pairs (A,B) with |A|=4, |B|=4, A cap B = empty.
        # C(16,4) * C(12,4) / 2 ? No, A and B are ordered in the mask logic?
        # The mask logic assumes A is the first 4 bits, B is the next 4.
        # So we need to map the actual vertices to 0..3 and 4..7.
        
        # Let's stick to 8-subsets for simplicity, but optimize the inner loop.
        
        # To speed up, we can check if the induced subgraph has >= 12 edges.
        
        vertices = range(n)
        # Shuffle vertices to find Q3s faster? No, we need to find ALL to ensure maximal?
        # For the removal phase, finding ANY is enough.
        # For the addition phase, we need to check if adding an edge creates a Q3.
        # That requires checking if the new edge is in a Q3.
        
        # Let's implement a helper to check if a specific edge (u,v) is in a Q3.
        pass

    # Phase 1: Remove edges to make Q3-free
    # We will do this until no Q3 is found.
    # To speed up finding, we can use random sampling of 8-sets initially, 
    # then exhaustive if needed.
    
    # Actually, a full scan is ~0.1s in Python if optimized? 
    # 12870 * 35 = 450k splits.
    # For each split, we construct mask (8 bitwise ops) and check against ~24 masks.
    # 450k * 24 = 10.8M bitwise ops.
    # Python is slow. 10M ops might take 1-2 seconds.
    # We can do ~50-100 scans in 115s.
    # Each scan removes at least 1 edge.
    # K16 has 120 edges. We need to remove ~60-70 edges.
    # So ~70 scans. 70 * 2s = 140s. Too slow.
    
    # Optimization:
    # 1. Precompute all 8-subsets.
    # 2. Precompute all splits for each 8-subset.
    # 3. Use bit operations efficiently.
    
    # Let's try a different approach for Phase 1:
    # Start with a known dense Q3-free graph and try to add edges.
    # What is a good starting graph?
    # Turan graph T(16, 3)? No, T(16,3) is 3-partite.
    # T(16,3) has parts of size 5,5,6.
    # Edges: 5*5 + 5*6 + 5*6 = 25 + 30 + 30 = 85 edges.
    # Does T(16,3) contain Q3?
    # Q3 is bipartite. Bipartite graphs are 2-colorable.
    # A subgraph of a 3-partite graph is not necessarily bipartite?
    # Wait, any subgraph of a bipartite graph is bipartite.
    # T(16,3) is NOT bipartite (it has triangles).
    # But Q3 is bipartite.
    # Can a 3-partite graph contain a bipartite subgraph? Yes.
    # Example: K_{3,3} is a subgraph of T(6,2) which is bipartite.
    # T(16,3) contains K_{5,5} as a subgraph? No, it's 3-partite.
    # It contains K_{5,5} between part 1 and part 2? Yes, all edges between part 1 and 2 exist.
    # So T(16,3) contains K_{5,5}.
    # K_{5,5} contains Q3 (since K_{4,4} contains Q3).
    # So T(16,3) is NOT Q3-free.
    
    # What about T(16, 4)? 4-partite.
    # Parts 4,4,4,4.
    # Edges: 4 * C(4,2) * 16? No.
    # Edges between parts: 4*4 * 6 pairs = 16 * 6 = 96 edges.
    # Does T(16,4) contain Q3?
    # It contains K_{4,4} between any two parts.
    # K_{4,4} contains Q3.
    # So T(16,4) is NOT Q3-free.
    
    # We need a graph where no 4 vertices in one "side" and 4 in another form a Q3.
    # This suggests we need to break the K_{4,4} structures.
    
    # Let's go back to the Removal heuristic but optimize it.
    # We can use NumPy for mask checks?
    
    # Alternative: 
    # Use a SAT solver? No, standard library only.
    
    # Let's try to optimize the Python loop for find_q3.
    # We can precompute the list of 8-subsets and their splits.
    
    subsets_8 = list(itertools.combinations(range(n), 8))
    # For each subset, precompute the list of (A, B) splits
    # A and B are tuples of 4 vertices.
    # And the mapping of these vertices to 0..7 for the mask.
    
    precomputed_checks = []
    for sub in subsets_8:
        # Generate all ways to split sub into two sets of 4
        # We can just take all 4-subsets of sub as A, and the rest as B.
        # To avoid duplicates (A,B) and (B,A), we can enforce a canonical order, 
        # but for mask checking, we just need to check one orientation?
        # No, the mask is directed from A to B.
        # If we swap A and B, the mask changes (bits are permuted).
        # However, Q3 is symmetric.
        # So checking (A,B) is sufficient if we check all (A,B) pairs.
        # Number of pairs is C(8,4) = 70.
        # We can just iterate all 70.
        
        for A in itertools.combinations(sub, 4):
            B = tuple(v for v in sub if v not in A)
            # Map A to 0..3 and B to 4..7
            # We need to know which bit corresponds to which edge in adj
            # Edge (a, b) where a in A, b in B.
            # Bit index = idx_in_A * 4 + idx_in_B
            # We need to construct the mask from adj.
            
            # To speed up, we can precompute the bit positions for each (A,B) pair?
            # No, the vertices change.
            
            # We'll just do it on the fly but efficiently.
            precomputed_checks.append((A, B))
            
    # This list will have 12870 * 70 = 900k entries.
    # Storing this is fine.
    
    def check_q3(adj):
        # Returns an edge (u,v) in a Q3, or None
        # To speed up, we can skip if edge count < 12? No, induced subgraph edge count.
        
        # We can iterate through precomputed_checks
        # But 900k iterations per call is slow in Python.
        
        # Let's try to reduce the search space.
        # We only need to find ONE Q3.
        # We can randomize the order of checks.
        
        checks = precomputed_checks
        random.shuffle(checks) # This is O(N) and modifies list. 
        # Better to use random indices.
        
        # Actually, let's just iterate and break early.
        
        for A, B in checks:
            # Construct mask
            mask = 0
            # Map A to 0..3
            # Map B to 0..3
            # We need indices.
            # Since A and B are tuples of ints, we can use list.index? No, slow.
            # We can precompute a dict for each check? No.
            
            # Let's precompute the "bit positions" for each check.
            # For a given (A, B), we need to know for each pair (a,b) in AxB, 
            # what is the bit index in the 16-bit mask.
            # It is i*4 + j.
            # We also need to know which bit in adj[a] corresponds to b.
            # It is 1 << b.
            
            # So for each check (A, B), we can precompute a list of (a, b, mask_bit)
            # where mask_bit is the bit to set in the 16-bit mask.
            
            pass

    # Let's refactor precomputation to include bit positions.
    
    precomputed_checks_v2 = []
    for sub in subsets_8:
        for A in itertools.combinations(sub, 4):
            B = tuple(v for v in sub if v not in A)
            # Create mapping
            # We need to iterate i in 0..3, j in 0..3
            # a = A[i], b = B[j]
            # mask_bit = i*4 + j
            # adj_bit = 1 << b
            
            # Store (A, B, list_of_tuples)
            # list_of_tuples = [(1<<b, mask_bit) for i in range(4) for j in range(4)]
            # Wait, we need to check if adj[a] has bit b.
            # So we need (a, b, mask_bit).
            
            pairs = []
            for i in range(4):
                a = A[i]
                for j in range(4):
                    b = B[j]
                    pairs.append((a, b, i*4 + j))
            
            precomputed_checks_v2.append(pairs)
            
    # Now check_q3 is:
    def find_q3_edge_fast(adj):
        # Shuffle indices for randomization
        indices = list(range(len(precomputed_checks_v2)))
        random.shuffle(indices)
        
        for idx in indices:
            pairs = precomputed_checks_v2[idx]
            mask = 0
            # Build mask
            # Optimization: If we find a 0 bit, we can still continue, 
            # but if we know it's not Q3, we skip.
            # We must build the full mask to check against q3_masks.
            
            for a, b, mb in pairs:
                if adj[a] & (1 << b):
                    mask |= (1 << mb)
            
            # Check if mask contains any Q3
            # q3_masks is a list of 16-bit ints.
            # We can check if (mask & q3_mask) == q3_mask for any q3_mask.
            # To speed up, we can group q3_masks?
            # Or just iterate.
            
            for qm in q3_masks:
                if (mask & qm) == qm:
                    # Found Q3.
                    # Return an edge from this Q3.
                    # We need to map the mask bits back to (a,b).
                    # The pairs list has the mapping.
                    # We can pick the first bit set in qm.
                    # Find index of first set bit in qm.
                    # bit_pos = (qm & -qm).bit_length() - 1
                    # Then find (a,b) in pairs where mb == bit_pos.
                    
                    # Actually, we can just return the first edge in the Q3.
                    # Let's find one.
                    for a, b, mb in pairs:
                        if qm & (1 << mb):
                            return (a, b)
                    # Should not reach here
                    return (pairs[0][0], pairs[0][1])
                    
        return None

    # Phase 1: Removal
    # Loop until no Q3
    # To avoid infinite loops or slow convergence, we limit iterations.
    
    max_iter = 1000
    for _ in range(max_iter):
        if time.time() > deadline:
            break
            
        edge = find_q3_edge_fast(adj)
        if edge is None:
            break
            
        u, v = edge
        # Remove edge
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
    # Phase 2: Addition
    # Try to add missing edges.
    # For each missing edge (u,v):
    #   Add it.
    #   Check if Q3 created.
    #   If yes, remove the new edge (revert).
    #   If no, keep it.
    
    # To speed up, we can just try random missing edges.
    
    missing_edges = []
    for u in range(n):
        for v in range(u+1, n):
            if not (adj[u] & (1 << v)):
                missing_edges.append((u, v))
                
    random.shuffle(missing_edges)
    
    for u, v in missing_edges:
        if time.time() > deadline:
            break
            
        # Tentatively add
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        # Check if Q3 exists
        # We can use find_q3_edge_fast, but it's slow.
        # We only care if the NEW edge is in a Q3.
        # We can check if there is a Q3 containing (u,v).
        
        # How to check if (u,v) is in a Q3?
        # We can modify find_q3_edge_fast to check only Q3s containing (u,v).
        # Or just run the full check.
        # If a Q3 is found, it might not contain (u,v).
        # If it doesn't contain (u,v), then the graph was already not Q3-free?
        # No, we ensured it was Q3-free before adding.
        # So any Q3 found MUST contain (u,v).
        
        edge_in_q3 = find_q3_edge_fast(adj)
        
        if edge_in_q3 is not None:
            # Q3 found. It must contain (u,v) or be pre-existing (impossible).
            # So we must remove (u,v) to restore Q3-freeness.
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        else:
            # Keep the edge
            pass

    # Convert adj to edge list
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if adj[u] & (1 << v):
                edges.append([u, v])
                
    return edges

# Run once to ensure no syntax errors
# construct()
