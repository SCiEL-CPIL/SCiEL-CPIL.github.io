import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    # Precompute all 8-subsets and their 4+4 partitions (bipartitions)
    # For each 8-subset S, and each 4-subset A of S, B = S \ A
    # We store (A, B) as tuples of vertex indices.
    # There are C(16,8) * C(8,4) = 12870 * 70 = 900900 such pairs.
    # This is a bit large for memory if we store full tuples, but we can optimize.
    # Actually, we can just generate them on the fly in the check function?
    # No, precomputing is better for speed in the inner loop.
    # Let's precompute the list of (mask_A, mask_B) for each 8-subset?
    # Or just list of (tuple_A, tuple_B).
    
    # To save memory and time, let's precompute the list of all (A, B) pairs for all 8-subsets.
    # Total pairs: 900,900. Each pair is two tuples of 4 ints.
    # This might take ~50-100MB of memory. Acceptable.
    
    # Alternative: Precompute for each 8-subset, the list of 70 partitions.
    # Let's just generate the list of all (A, B) pairs once.
    
    all_partitions = []
    for S in itertools.combinations(range(16), 8):
        S_set = set(S)
        for A in itertools.combinations(S, 4):
            A_set = set(A)
            B = tuple(v for v in S if v not in A_set)
            # We can represent A and B as bitmasks for faster neighbor checking?
            # Or just keep as tuples.
            # Let's keep as tuples for clarity, but use bitmasks in the check.
            all_partitions.append((A, B))
            
    # Precompute permutations of 4
    perms = list(itertools.permutations(range(4)))
    
    # Precompute required masks for each perm
    # For a perm p, row i must have 1s in all positions j != p[i].
    # req_mask[i] = 15 ^ (1 << p[i])
    perm_req_masks = []
    for p in perms:
        masks = [15 ^ (1 << p[i]) for i in range(4)]
        perm_req_masks.append(masks)

    def has_q3_and_edge(adj):
        """
        Checks if the graph defined by adj (list of 16 ints) contains a Q3.
        Returns (True, (u, v)) if Q3 found, where (u, v) is an edge in the Q3.
        Returns (False, None) otherwise.
        """
        # Iterate through precomputed partitions
        # To speed up, we can shuffle the partitions? No, deterministic is fine for now, 
        # but random order might help find "easier" edges to remove? 
        # Actually, finding ANY edge is enough.
        
        # Optimization: 
        # Instead of checking all 900k partitions every time, 
        # we can check only those that were "close" to being a Q3? 
        # That's complex. Let's stick to full check but optimized.
        
        # 900k iterations * 24 perms * 4 rows = ~86 million operations per check.
        # In Python, this is too slow (~1-2 seconds per check).
        # We need a faster check.
        
        # Better approach:
        # Iterate over 8-subsets. For each, check if it contains a Q3.
        # C(16,8) = 12870.
        # For each 8-subset, there are 70 partitions.
        # Total 900k checks.
        
        # Can we prune?
        # A Q3 has 12 edges.
        # If the induced subgraph on 8 vertices has < 12 edges, it can't be a Q3.
        # We can count edges in the induced subgraph first.
        
        for S in itertools.combinations(range(16), 8):
            # Count edges in induced subgraph
            edge_count = 0
            for i in range(8):
                u = S[i]
                # Count neighbors in S
                # We can do this with bitmasks
                mask_u = adj[u]
                # Intersect with mask of S
                # Let's precompute mask for S? No, S changes.
                # Let's just iterate.
                for j in range(i+1, 8):
                    v = S[j]
                    if mask_u & (1 << v):
                        edge_count += 1
                        
            if edge_count < 12:
                continue
                
            # If edge_count >= 12, check for Q3 structure
            # We need to check if the induced graph IS a Q3 (isomorphic).
            # Q3 is the only 3-regular bipartite graph on 8 vertices?
            # No, K4,4 has 16 edges. Q3 has 12.
            # Are there other 12-edge bipartite graphs on 8 vertices?
            # Yes. So we must check isomorphism.
            
            # Check all 70 partitions of S into A and B
            S_set = set(S)
            for A in itertools.combinations(S, 4):
                A_set = set(A)
                B = tuple(v for v in S if v not in A_set)
                
                # Build 4x4 adjacency matrix M (rows A, cols B)
                # M[i][j] = 1 if edge(A[i], B[j])
                # Use bitmasks for rows
                rows = []
                for i in range(4):
                    u = A[i]
                    mask = 0
                    for j in range(4):
                        v = B[j]
                        if adj[u] & (1 << v):
                            mask |= (1 << j)
                    rows.append(mask)
                    
                # Check if this matrix corresponds to a Q3
                # Q3 is 3-regular. So each row must have exactly three 1s?
                # Yes, in Q3, each vertex has degree 3.
                # So if any row has popcount != 3, it's not a Q3.
                # This is a very fast filter!
                
                valid = True
                for r in rows:
                    if bin(r).count('1') != 3:
                        valid = False
                        break
                        
                if not valid:
                    continue
                
                # If all rows have 3 ones, check if it's a Q3.
                # A 3-regular bipartite graph on 4+4 vertices is a Q3 if it is connected?
                # No, K4,4 minus a perfect matching is Q3.
                # K4,4 has 16 edges. Q3 has 12.
                # K4,4 - M (perfect matching) has 12 edges and is 3-regular.
                # Is it isomorphic to Q3?
                # Yes, Q3 is exactly K4,4 minus a perfect matching.
                # So, if the induced subgraph is 3-regular and bipartite (which it is by construction of A,B),
                # it is a Q3 IF AND ONLY IF the "missing" edges form a perfect matching?
                # Wait, if it's 3-regular, the missing edges in K4,4 form a 1-regular graph (perfect matching).
                # So ANY 3-regular bipartite graph on 8 vertices (4+4) is isomorphic to Q3?
                # Let's verify.
                # Number of 3-regular bipartite graphs on 4+4:
                # We need to choose 3 neighbors for each of 4 A-nodes from 4 B-nodes.
                # This is equivalent to a 4x4 0-1 matrix with row sums 3 and col sums 3.
                # This is equivalent to a permutation matrix P such that M = J - P.
                # So M is determined by P.
                # There are 4! = 24 such matrices.
                # Are they all isomorphic to Q3?
                # Q3 is the standard one.
                # Any 3-regular bipartite graph on 8 vertices is isomorphic to Q3?
                # Yes, Q3 is the unique 3-regular bipartite graph on 8 vertices.
                # Proof: Q3 is the cube. It is 3-regular, bipartite, 8 vertices.
                # Uniqueness: The only 3-regular bipartite graph on 2n vertices for n=4 is the complete bipartite K4,4 minus a perfect matching.
                # Since K4,4 is unique and perfect matchings are all equivalent under isomorphism of K4,4,
                # all such graphs are isomorphic.
                
                # So, if we find a partition (A, B) such that the induced subgraph is 3-regular (all degrees 3),
                # it IS a Q3.
                
                # We already checked row sums are 3.
                # We must also check column sums are 3.
                
                col_sums = [0, 0, 0, 0]
                for r in rows:
                    col_sums[0] += (r & 1)
                    col_sums[1] += (r & 2) >> 1
                    col_sums[2] += (r & 4) >> 2
                    col_sums[3] += (r & 8) >> 3
                    
                if all(c == 3 for c in col_sums):
                    # Found a Q3!
                    # Return an edge from it.
                    # Pick the first edge in A[0]
                    u = A[0]
                    # Find a neighbor in B
                    # Since row 0 has 3 ones, pick the first set bit
                    r0 = rows[0]
                    j = 0
                    while not (r0 & (1 << j)):
                        j += 1
                    v = B[j]
                    return True, (u, v)
                    
        return False, None

    best_edges = 0
    best_adj = None
    
    # Local Search with Multiple Restarts
    for restart in range(50):
        if time.time() > deadline:
            break
            
        # Initialize dense graph
        adj = [0] * 16
        p = 0.9
        for u in range(16):
            for v in range(u+1, 16):
                if random.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        # Phase 1: Remove edges to break Q3
        # This loop might be slow if has_q3_and_edge is slow.
        # But has_q3_and_edge is now much faster due to pruning (edge count < 12).
        
        while time.time() < deadline:
            found, edge = has_q3_and_edge(adj)
            if not found:
                break
            u, v = edge
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            
        # Phase 2: Add edges greedily
        improved = True
        while improved and time.time() < deadline:
            improved = False
            missing_edges = []
            for u in range(16):
                for v in range(u+1, 16):
                    if not (adj[u] & (1 << v)):
                        missing_edges.append((u, v))
            
            random.shuffle(missing_edges)
            
            for u, v in missing_edges:
                if time.time() > deadline:
                    break
                
                # Try adding
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
                found, _ = has_q3_and_edge(adj)
                if found:
                    # Revert
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
        # Fallback: Empty graph
        return []
        
    edges = []
    for u in range(16):
        for v in range(u+1, 16):
            if best_adj[u] & (1 << v):
                edges.append([u, v])
                
    return edges
