import time
import random
import heapq
from collections import defaultdict

def construct():
    deadline = time.time() + 112
    n = 16
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Precompute all Q3 subgraphs (sets of 8 vertices) for fast lookup
    # A Q3 is isomorphic to the 3-cube. We need to identify all 8-vertex subsets
    # that contain a Q3 subgraph.
    # Actually, checking if a specific 8-set induces a Q3 is faster than checking subgraph isomorphism.
    # But wait, the task says "no subgraph isomorphic to Q3". This means the graph must not CONTAIN a Q3.
    # It does NOT mean the induced subgraph on 8 vertices must not be Q3.
    # So a set of 8 vertices might contain a Q3 even if the induced subgraph has extra edges.
    
    # However, for the removal strategy:
    # If we start with K16, it contains MANY Q3s.
    # We need to remove edges until no Q3 remains.
    # An edge (u,v) is "critical" if it is part of some Q3.
    
    # Efficient check: Does the current graph contain a Q3?
    # We can iterate over all 8-subsets? C(16,8) = 12870.
    # For each 8-subset, check if it contains a Q3.
    # A graph on 8 vertices contains a Q3 if there is a labeling of the 8 vertices
    # such that the edges of the Q3 are present in the graph.
    
    # Better approach for removal:
    # 1. Find one Q3 in the current graph.
    # 2. Remove an edge from that Q3.
    # 3. Repeat.
    
    # How to find a Q3 quickly?
    # We can precompute the "Q3 templates" on 8 vertices.
    # Actually, it's easier to just check all 8-subsets for Q3 existence.
    
    # Let's precompute which 8-subsets of 16 vertices can potentially host a Q3?
    # No, any 8 vertices can host a Q3 if the edges are there.
    
    # Let's optimize the "find a Q3" routine.
    # Given adj masks, find if there exists a Q3.
    # We can iterate over all 8-subsets.
    # For a specific 8-subset S, let G_S be the induced subgraph.
    # G_S contains a Q3 iff there is a bijection from Q3 vertices to S
    # such that all Q3 edges map to edges in G_S.
    
    # Since Q3 is fixed, we can precompute all edge sets of Q3.
    # Q3 vertices: 0..7. Edges: (i,j) where popcount(i xor j) == 1.
    q3_edges = []
    for i in range(8):
        for j in range(i+1, 8):
            if bin(i ^ j).count('1') == 1:
                q3_edges.append((i, j))
                
    # There are 12 Q3 edges.
    
    def subset_has_q3(vertices, adj):
        # vertices: tuple of 8 ints
        # Check if induced subgraph on these vertices contains a Q3.
        # We need to check if there is an isomorphism from Q3 to a subgraph of G[vertices].
        # Since Q3 has 8 vertices, the isomorphism is a permutation of vertices.
        # We can try all 8! = 40320 permutations? Too slow if done naively for 12870 subsets.
        # 12870 * 40320 ~ 500M ops. Too slow in Python.
        
        # Optimization:
        # 1. Check degree sequence? Q3 is 3-regular.
        #    If G[vertices] has a vertex with degree < 3, it can't contain a Q3?
        #    No, the Q3 vertices must have degree >= 3 in G[vertices].
        #    So if any vertex in S has degree < 3 in G[S], skip.
        # 2. Check if G[S] has at least 12 edges.
        
        # Let's implement a backtracking search for Q3 within the 8-vertex subset.
        # Actually, we can just check all 8! mappings?
        # Or use a faster isomorphism check for small graphs.
        
        # Let's try a simple backtracking.
        # Map Q3 vertices 0..7 to vertices in S.
        # q3_adj: list of sets of neighbors in Q3.
        q3_adj = [set() for _ in range(8)]
        for u, v in q3_edges:
            q3_adj[u].add(v)
            q3_adj[v].add(u)
            
        # Precompute q3_adj as bitmasks for speed
        q3_adj_mask = [0] * 8
        for u in range(8):
            for v in q3_adj[u]:
                q3_adj_mask[u] |= (1 << v)
        
        # Backtracking
        # perm[i] = vertex in S assigned to Q3 vertex i
        # We need to ensure that for all Q3 edges (i,j), perm[i] and perm[j] are adjacent in G.
        
        # Optimization: Order Q3 vertices by degree (all 3).
        # Use symmetry breaking?
        
        # Let's just do a DFS.
        # If it's too slow, we'll optimize later.
        
        # First, quick check:
        # Degrees in G[S]
        deg_in_s = [0] * 8
        for i in range(8):
            for j in range(i+1, 8):
                if adj[vertices[i]] & (1 << vertices[j]):
                    deg_in_s[i] += 1
                    deg_in_s[j] += 1
        
        # All vertices in Q3 have degree 3.
        # So in G[S], the vertices corresponding to Q3 must have degree >= 3.
        # If any vertex in S has degree < 3, it cannot be part of a Q3?
        # Wait, the Q3 is a SUBGRAPH. So the vertices used by the Q3 must have degree >= 3 in G[S].
        # If there are 8 vertices in S, and we need to pick 8 vertices for the Q3,
        # then ALL vertices in S must have degree >= 3 in G[S].
        # Because the Q3 uses all 8 vertices of S?
        # NO. The Q3 is a subgraph of G. It uses 8 vertices.
        # The 8-subset S is just a candidate set of vertices.
        # If G[S] contains a Q3, it means there is a set of 8 vertices in S (which is S itself)
        # that forms a Q3?
        # Yes, because S has exactly 8 vertices.
        # So if G[S] contains a Q3, the Q3 must use all 8 vertices of S.
        # Therefore, all 8 vertices must have degree >= 3 in G[S].
        
        if any(d < 3 for d in deg_in_s):
            return False
            
        # Also, sum of degrees must be >= 24 (12 edges * 2).
        if sum(deg_in_s) < 24:
            return False
            
        # Now check isomorphism.
        # We can use a simple recursive backtracking.
        # Map q3_vertex -> s_vertex
        mapping = [-1] * 8
        used = [False] * 8
        
        # Order q3 vertices to fix early.
        # Let's fix q3 vertex 0 to s vertex 0 to break symmetry?
        # No, we need to check all.
        # But we can fix q3 vertex 0 to any s vertex.
        # Let's iterate s_vertex for q3_vertex 0.
        
        def backtrack(q3_v):
            if q3_v == 8:
                return True
            
            for s_v in range(8):
                if used[s_v]:
                    continue
                
                # Check if mapping is consistent so far
                # For all neighbors w of q3_v in Q3, if w is already mapped,
                # then mapping[w] must be a neighbor of s_v in G[S].
                
                valid = True
                for w in q3_adj[q3_v]:
                    if mapping[w] != -1:
                        # Check if s_v and mapping[w] are adjacent
                        if not (adj[vertices[s_v]] & (1 << vertices[mapping[w]])):
                            valid = False
                            break
                            
                if not valid:
                    continue
                
                mapping[q3_v] = s_v
                used[s_v] = True
                
                if backtrack(q3_v + 1):
                    return True
                    
                mapping[q3_v] = -1
                used[s_v] = False
                
            return False
            
        # Symmetry breaking: Fix q3 vertex 0 to s vertex 0?
        # No, because the Q3 might map 0 to 1, etc.
        # But we can fix q3 vertex 0 to s vertex 0 without loss of generality?
        # No, because the Q3 is unlabeled.
        # However, we can fix the mapping of the first vertex to reduce permutations.
        # Let's just run the backtrack. 8! is 40320.
        # With pruning, it should be fast.
        
        # To speed up, we can order s_vertices by degree.
        # Vertices with lower degree in G[S] should be tried first?
        # Or higher?
        # Let's just try all.
        
        return backtrack(0)
        
    # This might be too slow if called for every 8-subset in every iteration.
    # Number of 8-subsets: 12870.
    # If we do this in every iteration of edge removal, and we remove ~100 edges,
    # that's 1.2M checks. Each check is a backtrack.
    # This might be too slow.
    
    # Alternative:
    # Instead of checking all 8-subsets, we can find a Q3 more directly.
    # Pick a random edge (u,v).
    # Check if it is part of a Q3.
    # If yes, remove it.
    # If no, try another edge.
    
    # How to check if edge (u,v) is part of a Q3?
    # A Q3 containing (u,v) must have u and v as adjacent vertices in the Q3.
    # In Q3, adjacent vertices differ by 1 bit.
    # Let's map u to 000 and v to 001 (WLOG).
    # Then the other 6 vertices of the Q3 must be in the graph and connected appropriately.
    
    # This is a constraint satisfaction problem.
    # Let's stick to the 8-subset check but optimize it.
    # Or, precompute the list of all 8-subsets and check them in random order.
    
    # Let's try a different approach:
    # Start with K16.
    # While graph contains a Q3:
    #   Find a Q3 (by iterating 8-subsets in random order).
    #   Remove a random edge from that Q3.
    
    # To make it faster, we can cache which 8-subsets are "potential" Q3 hosts.
    # But the graph changes.
    
    # Let's implement the removal strategy with the 8-subset check.
    # If it's too slow, we'll switch to a simpler heuristic.
    
    def find_q3(adj):
        # Iterate 8-subsets in random order
        all_8_subsets = []
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        for m in range(l+1, n):
                            for p in range(m+1, n):
                                for q in range(p+1, n):
                                    for r in range(q+1, n):
                                        all_8_subsets.append((i, j, k, l, m, p, q, r))
                                        
        random.shuffle(all_8_subsets)
        
        for subset in all_8_subsets:
            if time.time() > deadline:
                return None
            if subset_has_q3(subset, adj):
                return subset
        return None
    
    # This is likely too slow. Let's optimize subset_has_q3.
    # Or use a different method.
    
    # New Idea:
    # Use the fact that Q3 is bipartite.
    # But that doesn't help much.
    
    # Let's try to speed up subset_has_q3 by using bitmasks for the induced subgraph.
    # Represent the induced subgraph on 8 vertices as a 64-bit mask (or 8x8 matrix).
    
    def subset_has_q3_fast(vertices, adj):
        # Build adjacency matrix of G[vertices]
        # adj_mat[i][j] = 1 if vertices[i] and vertices[j] are adjacent
        adj_mat = [[0]*8 for _ in range(8)]
        deg = [0]*8
        for i in range(8):
            for j in range(i+1, 8):
                if adj[vertices[i]] & (1 << vertices[j]):
                    adj_mat[i][j] = 1
                    adj_mat[j][i] = 1
                    deg[i] += 1
                    deg[j] += 1
                    
        if any(d < 3 for d in deg):
            return False
        if sum(deg) < 24:
            return False
            
        # Backtrack
        # Precompute q3_adj_mask
        # We can use a global q3_adj_mask
        
        mapping = [-1] * 8
        used = [False] * 8
        
        # Order s_vertices by degree to prune earlier?
        # Let's try to map q3 vertices in a fixed order.
        # q3 vertices: 0..7
        # s vertices: 0..7 (indices in vertices)
        
        # We can use a heuristic: map q3 vertex 0 to the s vertex with min degree?
        # Or max?
        
        def backtrack(q3_v):
            if q3_v == 8:
                return True
            
            # Try all unused s_vertices
            # Sort candidates by degree?
            candidates = [s_v for s_v in range(8) if not used[s_v]]
            # Sort by degree in G[S]
            candidates.sort(key=lambda x: deg[x])
            
            for s_v in candidates:
                # Check consistency
                valid = True
                for w in q3_adj[q3_v]:
                    if mapping[w] != -1:
                        if not adj_mat[s_v][mapping[w]]:
                            valid = False
                            break
                            
                if not valid:
                    continue
                
                mapping[q3_v] = s_v
                used[s_v] = True
                
                if backtrack(q3_v + 1):
                    return True
                    
                mapping[q3_v] = -1
                used[s_v] = False
                
            return False
            
        return backtrack(0)
        
    # Still might be slow.
    # Let's try a different strategy:
    # Instead of finding ANY Q3, we can use a local search on the COMPLEMENT.
    # Start with empty complement (K16).
    # Add edges to complement (remove from G) to break Q3s.
    # Minimize number of edges in complement.
    
    # This is equivalent to the previous strategy.
    
    # Let's try to implement a faster Q3 check.
    # We can precompute all 8! permutations of Q3 vertices?
    # No, we can precompute the edge sets of all Q3 embeddings in K8.
    # There are 8! / 48 = 840 distinct Q3 embeddings in K8?
    # No, the number of subgraphs isomorphic to Q3 in K8 is 8! / |Aut(Q3)|.
    # |Aut(Q3)| = 48.
    # So 40320 / 48 = 840.
    
    # We can precompute the 840 edge masks for Q3 in K8.
    # Then, for a given 8-subset, check if any of these 840 masks is a subset of the induced edge mask.
    
    # This is much faster!
    
    # Precompute Q3 edge masks in K8
    q3_edge_masks = set()
    # Generate all permutations of 0..7
    import itertools
    for perm in itertools.permutations(range(8)):
        # perm[i] is the vertex in K8 corresponding to Q3 vertex i
        # Build edge mask
        mask = 0
        for u, v in q3_edges:
            # Edge between perm[u] and perm[v]
            a, b = perm[u], perm[v]
            if a > b:
                a, b = b, a
            # Encode edge (a,b) in 8x8 matrix?
            # 28 edges. Use a 28-bit mask.
            # Edge index: a*7 + (b-1) - (a*(a+1)//2)?
            # Let's use a simple index: a*8+b
            idx = a * 8 + b
            mask |= (1 << idx)
        q3_edge_masks.add(mask)
        
    # Actually, the set of edges is the same for all isomorphic copies.
    # We just need to check if the induced subgraph contains the Q3 edge set.
    # So we can precompute the set of edge masks for Q3.
    # But since all Q3s are isomorphic, the edge set (as a set of pairs) is the same up to relabeling.
    # So we just need to check if the induced subgraph has at least one Q3.
    # This is equivalent to checking if the induced subgraph is "Q3-saturated"?
    # No, just if it CONTAINS a Q3.
    
    # So, for a given 8-subset, we build the edge mask of the induced subgraph.
    # Then we check if any of the 840 Q3 masks is a subset of it.
    
    # Let's precompute the 840 masks.
    # Note: The masks are in terms of vertex indices 0..7.
    # When we check a specific 8-subset, we map the subset vertices to 0..7.
    # So the mask is in the local 0..7 coordinate system.
    # Since all Q3s are isomorphic, the set of masks is the same.
    
    # Let's verify:
    # The number of distinct edge masks for Q3 in K8 is 1?
    # No, because the edges are between specific vertices.
    # But if we relabel the vertices, the edge mask changes.
    # So there are 840 distinct masks.
    
    # Let's generate them.
    
    # Wait, the code above generates masks based on the permutation.
    # But the mask is defined by the edges (a,b) where a,b are in 0..7.
    # So yes, we get 840 distinct masks.
    
    # Now, for a given 8-subset, we compute the induced edge mask.
    # And check if any q3_mask is a subset of it.
    
    def subset_has_q3_mask(vertices, adj):
        # Build induced edge mask
        mask = 0
        for i in range(8):
            for j in range(i+1, 8):
                if adj[vertices[i]] & (1 << vertices[j]):
                    a, b = i, j
                    idx = a * 8 + b
                    mask |= (1 << idx)
                    
        # Check if any q3_mask is subset of mask
        for q3_mask in q3_edge_masks:
            if (mask & q3_mask) == q3_mask:
                return True
        return False
        
    # This is O(840) per subset.
    # 12870 subsets * 840 = 10.8M ops.
    # This is fast in Python if optimized?
    # Maybe.
    
    # Let's use this.
    
    # Strategy:
    # 1. Start with K16 (adj masks all 1s except self).
    # 2. While time remains:
    #    a. Find a Q3 (by iterating 8-subsets in random order).
    #    b. If found, remove a random edge from that Q3.
    #    c. If not found, stop.
    
    # To make it faster, we can keep a list of "active" 8-subsets that might contain a Q3.
    # But initially, all do.
    
    # Let's implement this.
    
    adj = [0] * n
    for i in range(n):
        adj[i] = (1 << n) - 1
        adj[i] &= ~(1 << i)
        
    # Precompute all 8-subsets
    all_8_subsets = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    for m in range(l+1, n):
                        for p in range(m+1, n):
                            for q in range(p+1, n):
                                for r in range(q+1, n):
                                    all_8_subsets.append((i, j, k, l, m, p, q, r))
                                    
    random.shuffle(all_8_subsets)
    
    # Iterate
    while time.time() < deadline - 5:
        found = False
        # Shuffle subsets to avoid bias
        random.shuffle(all_8_subsets)
        
        for subset in all_8_subsets:
            if time.time() > deadline - 5:
                break
            if subset_has_q3_mask(subset, adj):
                # Found a Q3
                # Remove a random edge from this subset
                # The edges of the Q3 are not explicitly known, but we can pick a random edge in the subset
                # and hope it's part of the Q3?
                # Better: pick a random edge from the induced subgraph on this subset.
                # If it's not part of the Q3, it might not break the Q3.
                # But it's a heuristic.
                
                # Let's pick a random edge from the subset
                edges_in_subset = []
                for i in range(8):
                    for j in range(i+1, 8):
                        if adj[subset[i]] & (1 << subset[j]):
                            edges_in_subset.append((subset[i], subset[j]))
                            
                if edges_in_subset:
                    u, v = random.choice(edges_in_subset)
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    found = True
                    break
        if not found:
            break
            
    # Count edges
    count = sum(adj[u].bit_count() for u in range(n)) // 2
    
    # Convert to edge list
    result = []
    for u in range(n):
        mask = adj[u]
        while mask:
            bit = mask & (-mask)
            v = bit.bit_length() - 1
            if v > u:
                result.append([u, v])
            mask -= bit
            
    return result

# Precompute q3_edge_masks at module level
q3_edges = []
for i in range(8):
    for j in range(i+1, 8):
        if bin(i ^ j).count('1') == 1:
            q3_edges.append((i, j))

q3_edge_masks = set()
import itertools
for perm in itertools.permutations(range(8)):
    mask = 0
    for u, v in q3_edges:
        a, b = perm[u], perm[v]
        if a > b:
            a, b = b, a
        idx = a * 8 + b
        mask |= (1 << idx)
    q3_edge_masks.add(mask)

# Convert to list for faster iteration
q3_edge_masks_list = list(q3_edge_masks)
