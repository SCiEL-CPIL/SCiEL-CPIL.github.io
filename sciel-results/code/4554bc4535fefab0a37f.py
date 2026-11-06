import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin (evaluator kills at 150s)
    
    n = 16
    
    # Pre-compute all edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    num_edges = len(all_edges)
    
    # Represent graph as a set of edge indices for fast membership checks
    # Also maintain adjacency bitmasks for fast intersection
    
    def build_adj_bitmasks(edge_set):
        adj = [0] * n
        for e in edge_set:
            u, v = e
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def is_q3_free(adj_bitmasks):
        """
        Check if the graph contains a subgraph isomorphic to Q3.
        Q3 is the 3-cube. It is bipartite, 3-regular, 8 vertices, 12 edges.
        A key property: Q3 is the unique 3-regular bipartite graph on 8 vertices?
        Actually, we can check for any 8 vertices inducing a subgraph that contains Q3.
        
        Faster check:
        Iterate over all pairs of vertices (u, v) that are NOT connected? 
        Or iterate over all 8-subsets? C(16,8) = 12870. This is small enough.
        For each subset S of 8 vertices, check if G[S] contains Q3.
        
        How to check if G[S] contains Q3?
        Q3 is bipartite with parts of size 4.
        Let S = {v0, ..., v7}.
        We need to check if there is a partition S = A U B, |A|=|B|=4, such that
        G[S] contains all edges of a 3-regular bipartite graph between A and B?
        No, G[S] just needs to *contain* Q3. It can have extra edges.
        
        So, for a fixed S, does there exist a mapping of Q3 to S such that all 12 edges are in G?
        This is equivalent to: Is there a subgraph of G[S] isomorphic to Q3?
        
        Since G[S] is small (max 28 edges), we can brute force the isomorphism or use invariants.
        Invariants of Q3:
        - 8 vertices
        - 12 edges
        - 3-regular
        - Bipartite
        - Diameter 3
        
        If G[S] has a subgraph isomorphic to Q3, then:
        1. G[S] must have at least 12 edges.
        2. G[S] must be bipartite? No, G[S] can have triangles, but the Q3 subgraph itself is bipartite.
           So we just need to find a set of 12 edges in G[S] forming Q3.
        
        Algorithm for check(S):
        1. If number of edges in G[S] < 12: return False.
        2. Try all partitions of S into two sets A and B of size 4? No, that's C(8,4)=70 partitions.
           For each partition (A, B), check if the bipartite subgraph G[A,B] contains a 3-regular subgraph?
           Actually, Q3 is the *only* 3-regular bipartite graph on 8 vertices?
           Let's verify.
           Number of 3-regular bipartite graphs on 8 vertices (4+4):
           The adjacency matrix is a 4x4 binary matrix with row/col sums 3.
           This is equivalent to a 4x4 matrix with exactly one 0 in each row and col.
           There are 4! = 24 such matrices.
           Are they all isomorphic to Q3?
           Q3 adjacency matrix (bipartite part):
           1 1 1 0
           1 1 0 1
           1 0 1 1
           0 1 1 1
           This is the complement of the identity matrix.
           Any permutation of rows/cols preserves isomorphism class.
           The graph defined by J-I (where J is all-ones) is Q3.
           So, yes, any 3-regular bipartite graph on 8 vertices is isomorphic to Q3.
           
           Therefore: G[S] contains Q3 if and only if there exists a partition S = A U B, |A|=4, |B|=4,
           such that the bipartite graph between A and B in G[S] is 3-regular.
           i.e., every vertex in A has exactly 3 neighbors in B, and every vertex in B has exactly 3 neighbors in A.
           (Note: Edges within A or within B do not matter for the Q3 subgraph, as long as the cross edges form Q3).
           
           So, check(S):
           For each of the 70 partitions (A, B) of S:
               Check if deg_G(v, B) == 3 for all v in A, and deg_G(u, A) == 3 for all u in B.
               If yes, return True (Q3 found).
           Return False.
        """
        # Pre-compute list of 8-subsets? No, iterate on the fly or pre-compute once.
        # C(16,8) = 12870.
        # For each subset, 70 partitions.
        # Total checks: 12870 * 70 = ~900k partition checks.
        # Each check involves 8 degree lookups.
        # This is fast enough for a single check? 
        # 900k * 8 = 7.2M operations. In Python, this might take ~0.5s - 1s.
        # We need to do this many times in local search. 
        # Optimization: Use bitmasks.
        
        # To speed up, we can pre-compute the list of 8-subsets.
        # And maybe cache results? No, graph changes.
        
        # Let's try to optimize the inner loop.
        # Instead of iterating all 12870 subsets every time, 
        # we can check only subsets that are "likely" to contain Q3.
        # But for correctness, we must check all.
        
        # Let's pre-compute all 8-subsets as lists of vertices.
        subsets_8 = list(combinations(range(n), 8))
        
        # Pre-compute all partitions of 8 items into two sets of 4.
        # Represent partitions as (mask_A, mask_B) where mask_A has 4 bits set.
        # But we need to map global vertices to local indices 0..7.
        # This mapping overhead is high.
        
        # Alternative:
        # For a fixed subset S (tuple of 8 vertices), we can construct a 8x8 adjacency matrix or bitmasks.
        # Then check partitions.
        
        # Let's implement the check efficiently using bitmasks on the global adj list.
        
        for S in subsets_8:
            # S is a tuple of 8 vertices
            # Count edges in G[S]
            # If < 12, skip
            edge_count = 0
            for i in range(8):
                u = S[i]
                # Count neighbors in S
                mask_S = 0
                for j in range(8):
                    mask_S |= (1 << S[j])
                # Popcount of adj[u] & mask_S
                # Python int bit_count() is fast
                deg_u_in_S = (adj_bitmasks[u] & mask_S).bit_count()
                edge_count += deg_u_in_S
            if edge_count < 24: # 12 edges * 2
                continue
            
            # Check partitions
            # We need to iterate over all ways to split S into A and B of size 4.
            # There are C(8,4) = 70 such splits.
            # To avoid double counting (A,B) and (B,A), we can just fix one element in A?
            # Or just iterate all 70.
            
            # Pre-compute the 70 masks for 4 bits in 8 positions?
            # The positions are local 0..7.
            # But we need to map local to global.
            
            # Let's generate the 70 masks once.
            # mask_local has 4 bits set.
            # For each mask_local, A = {S[i] for i where mask_local has bit i}, B = rest.
            
            # Pre-compute list of (mask_A_local, list_A_indices, list_B_indices)
            # This is static for all S.
            
            # Let's do this outside the loop if possible, but S changes.
            # The structure of A and B relative to S is what matters.
            
            # Optimization:
            # For a given S, create a local adjacency matrix (8x8) or 8 bitmasks (8 bits each).
            local_adj = [0] * 8
            for i in range(8):
                u = S[i]
                mask_S = 0
                for j in range(8):
                    mask_S |= (1 << S[j])
                # Get bits corresponding to S
                # This is tricky. 
                # Better: create a map global -> local index for S.
                # Since n=16, we can use a small array.
            
            # Simpler:
            # For each partition mask (4 bits in 8 positions):
            # A_indices = [i for i in range(8) if (mask >> i) & 1]
            # B_indices = [i for i in range(8) if not ((mask >> i) & 1)]
            # Check if for all i in A_indices, deg(S[i], B) == 3
            # and for all j in B_indices, deg(S[j], A) == 3
            
            # Pre-compute the 70 masks
            # This list is constant.
            
            # To make it very fast, we can pre-compute for each of the 70 masks:
            # list of (local_idx_in_A, local_idx_in_B) pairs? No.
            # Just the lists of indices.
            
            # Let's pre-compute partitions once.
            if not hasattr(is_q3_free, 'partitions'):
                parts = []
                for mask in range(1 << 8):
                    if mask.bit_count() == 4:
                        A = [i for i in range(8) if (mask >> i) & 1]
                        B = [i for i in range(8) if not ((mask >> i) & 1)]
                        parts.append((A, B))
                # Remove duplicates? (A,B) and (B,A) are different masks, but the condition is symmetric?
                # If G[A,B] is 3-regular, then G[B,A] is the same graph.
                # So checking (A,B) is enough. But iterating all 70 is fine.
                is_q3_free.partitions = parts
            
            for A, B in is_q3_free.partitions:
                # Check if bipartite subgraph between A and B is 3-regular
                valid = True
                # Check degrees from A to B
                for i in A:
                    u = S[i]
                    # Count neighbors of u in B
                    cnt = 0
                    for j in B:
                        v = S[j]
                        if (adj_bitmasks[u] >> v) & 1:
                            cnt += 1
                    if cnt != 3:
                        valid = False
                        break
                if not valid:
                    continue
                # Check degrees from B to A
                for j in B:
                    v = S[j]
                    cnt = 0
                    for i in A:
                        u = S[i]
                        if (adj_bitmasks[v] >> u) & 1:
                            cnt += 1
                    if cnt != 3:
                        valid = False
                        break
                if valid:
                    return False # Q3 found
            
        return True # No Q3 found

    # Initialize with Turán graph T(16, 3)
    # Partitions: 0-5, 6-10, 11-15 (sizes 6, 5, 5)
    # Edges between different partitions are included.
    # Edges within partitions are excluded.
    
    initial_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            # Determine partition
            def get_part(v):
                if v < 6: return 0
                elif v < 11: return 1
                else: return 2
            
            if get_part(i) != get_part(j):
                initial_edges.add((i, j))
                
    current_edges = set(initial_edges)
    adj_bitmasks = build_adj_bitmasks(current_edges)
    
    # Verify initial graph is Q3-free (should be)
    # is_q3_free(adj_bitmasks)
    
    best_edges = set(current_edges)
    best_count = len(current_edges)
    
    # Local Search: Try to add edges
    # Candidate edges: Edges within partitions (currently missing)
    # And maybe some other edges? No, all cross-partition edges are already present.
    # So we can only add edges within the 3 partitions.
    
    candidate_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in current_edges:
                candidate_edges.append((i, j))
                
    random.shuffle(candidate_edges)
    
    # Iterative improvement
    # We try to add edges one by one. If adding an edge creates a Q3, we remove it.
    # To escape local optima, we can also try swapping: remove one edge, add another?
    # But adding is strictly better if possible.
    
    # Strategy:
    # 1. Greedily add candidate edges if they don't create Q3.
    # 2. If stuck, try random restarts or perturbations.
    
    # Since the initial graph is dense and Q3-free, adding even one edge might create a Q3.
    # Let's check how many edges we can add.
    
    # To speed up Q3 check, we only need to check subsets S that contain the new edge?
    # No, adding an edge (u,v) can only create a Q3 that uses that edge.
    # So we only need to check 8-subsets S that contain both u and v.
    # Number of such subsets: C(14, 6) = 3003.
    # This is much faster than 12870.
    
    def is_q3_free_after_add(adj_bitmasks, u, v):
        # Check if adding (u,v) creates a Q3.
        # Only check subsets S containing u and v.
        # Pre-compute list of such subsets?
        # Or iterate over all 8-subsets and filter.
        
        # For efficiency, let's iterate over all 8-subsets and check if u,v in S.
        # 12870 subsets. Filter is cheap.
        # But we can be smarter.
        
        # Let's just use the full check for now, but it might be slow.
        # 3003 subsets * 70 partitions * 8 checks = ~1.7M ops.
        # This is acceptable per step if we have many steps.
        
        # Let's implement a faster check specific to (u,v).
        
        # Find all 6 other vertices from the remaining 14.
        # C(14, 6) = 3003.
        
        # Pre-compute list of 6-subsets from 0..15 excluding u,v?
        # This depends on u,v.
        
        # Let's just iterate over all 8-subsets and skip those not containing u,v.
        # This is simple and likely fast enough given the low number of steps needed.
        
        for S in combinations(range(n), 8):
            if u not in S or v not in S:
                continue
            
            # Check if S contains a Q3 using edge (u,v)
            # We know (u,v) is in the graph now.
            # We need to check if there is a partition A,B of S such that
            # G[S] (with new edge) has a 3-regular bipartite subgraph.
            
            # Since we only added (u,v), the only change is that (u,v) is now an edge.
            # If (u,v) was not in the Q3, the Q3 would have existed before.
            # So we only care about Q3s that use (u,v).
            
            # This implies (u,v) must be a cross-edge in the bipartition (A,B).
            # So u and v must be in different parts.
            
            # Iterate partitions (A,B) of S.
            # For each partition:
            #   If u and v are in the same part, skip (edge (u,v) is not a cross edge).
            #   If u in A and v in B (or vice versa):
            #      Check if the bipartite graph is 3-regular.
            #      Note: Before adding (u,v), the degree of u to B was 2 (since it's now 3)
            #      and degree of v to A was 2.
            #      Other degrees were unchanged.
            
            for A, B in is_q3_free.partitions:
                # Determine if u, v are in different parts
                u_in_A = u in S[A.index(u)] # This is wrong, A is list of local indices
                # We need to map u to local index in S.
                # This is slow.
                
                # Let's stick to the full check for correctness first.
                # If too slow, optimize.
                
                # Actually, let's just run the full is_q3_free check.
                # It's 12870 subsets.
                # If we do this 100 times, it's 1.2M subset checks.
                # Each check is ~70 * 8 = 560 ops.
                # Total ~700M ops. Might be too slow in Python.
                
                pass
        
        # Fallback: Use full check
        return is_q3_free(adj_bitmasks)
    
    # Let's refine the strategy:
    # 1. Start with Turan graph.
    # 2. Try to add all candidate edges (within partitions).
    # 3. For each candidate edge, check if adding it creates a Q3.
    #    To speed up, we only check subsets containing the edge.
    
    def check_q3_with_edge(adj_bitmasks, u, v):
        # Check if adding (u,v) creates a Q3.
        # We temporarily add (u,v) to adj_bitmasks? No, we can just account for it.
        # Or we can modify adj_bitmasks, check, then revert.
        
        # Modify
        adj_bitmasks[u] |= (1 << v)
        adj_bitmasks[v] |= (1 << u)
        
        # Check only subsets containing u and v
        # Pre-compute list of 8-subsets containing u and v?
        # C(14, 6) = 3003 subsets.
        
        # Let's generate them on the fly or pre-compute for each pair?
        # Pre-computing for all 120 pairs is 120 * 3003 = 360k subsets.
        # Storing them is fine.
        
        if not hasattr(check_q3_with_edge, 'subsets_by_pair'):
            cache = {}
            for u in range(n):
                for v in range(u+1, n):
                    # Find all 6-subsets from remaining 14
                    rem = [x for x in range(n) if x != u and x != v]
                    subs = []
                    for comb in combinations(rem, 6):
                        S = tuple(sorted((u, v) + comb))
                        subs.append(S)
                    cache[(u, v)] = subs
            check_q3_with_edge.subsets_by_pair = cache
        
        subsets = check_q3_with_edge.subsets_by_pair.get((u, v), check_q3_with_edge.subsets_by_pair.get((v, u), []))
        
        for S in subsets:
            # Check if S contains Q3
            # Use the partition logic
            # S is a tuple of 8 vertices.
            
            # Optimization: 
            # We know (u,v) is an edge.
            # In a Q3, (u,v) must be a cross-edge.
            # So u and v must be in different parts A, B.
            
            for A, B in is_q3_free.partitions:
                # Check if u and v are in different parts
                # We need to know if u is in A or B.
                # A and B are lists of local indices (0..7).
                # We need to map u to local index.
                
                # This mapping is the bottleneck.
                # Let's pre-compute the local index of each vertex in S?
                # No, S changes.
                
                # Alternative:
                # Instead of local indices, let's store partitions as sets of global vertices?
                # No, partitions depend on S.
                
                # Let's just do the degree check directly.
                # For a partition (A_local, B_local):
                # A_global = {S[i] for i in A_local}
                # B_global = {S[i] for i in B_local}
                
                # If u in A_global and v in B_global (or vice versa):
                #   Check if all vertices in A_global have degree 3 to B_global.
                #   Check if all vertices in B_global have degree 3 to A_global.
                
                # To speed up, we can pre-compute for each S:
                # For each partition, the sets A_global and B_global.
                # But S has 12870 possibilities.
                
                # Let's try to just run the check and see if it's fast enough.
                # 3003 subsets * 70 partitions = 210k partition checks per edge.
                # If we try 50 edges, that's 10M checks.
                # Each check is ~16 degree lookups.
                # 160M ops. Might be ~10-20s in Python.
                # This is borderline.
                
                # Let's optimize the inner loop.
                
                # Pre-compute local indices for u and v in S?
                # We can create a dict {vertex: local_idx} for S.
                # This is O(8) per subset.
                
                # Let's do it.
                idx_map = {v: i for i, v in enumerate(S)}
                u_idx = idx_map[u]
                v_idx = idx_map[v]
                
                for A, B in is_q3_free.partitions:
                    # Check if u_idx in A and v_idx in B, or vice versa
                    if (u_idx in A and v_idx in B) or (u_idx in B and v_idx in A):
                        # Check 3-regularity
                        valid = True
                        # Check A vertices
                        for i in A:
                            vert = S[i]
                            cnt = 0
                            for j in B:
                                other = S[j]
                                if (adj_bitmasks[vert] >> other) & 1:
                                    cnt += 1
                            if cnt != 3:
                                valid = False
                                break
                        if not valid:
                            continue
                        # Check B vertices
                        for j in B:
                            vert = S[j]
                            cnt = 0
                            for i in A:
                                other = S[i]
                                if (adj_bitmasks[vert] >> other) & 1:
                                    cnt += 1
                            if cnt != 3:
                                valid = False
                                break
                        if valid:
                            # Revert change
                            adj_bitmasks[u] &= ~(1 << v)
                            adj_bitmasks[v] &= ~(1 << u)
                            return False # Q3 found
            
        # Revert change
        adj_bitmasks[u] &= ~(1 << v)
        adj_bitmasks[v] &= ~(1 << u)
        return True # No Q3 found

    # Main loop
    for e in candidate_edges:
        if time.time() > deadline - 5:
            break
        u, v = e
        # Check if adding e creates Q3
        if check_q3_with_edge(adj_bitmasks, u, v):
            # Safe to add
            current_edges.add(e)
            adj_bitmasks[u] |= (1 << v)
            adj_bitmasks[v] |= (1 << u)
            if len(current_edges) > best_count:
                best_count = len(current_edges)
                best_edges = set(current_edges)
        else:
            # Creating Q3, skip
            pass
            
    # If we have time, try to remove some edges and add others?
    # Or just return best.
    
    return [[u, v] for u, v in best_edges]

# EVOLVE-BLOCK-END
