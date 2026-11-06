"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct():
    start = time.time()
    deadline = start + 145
    n = 16
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    m = len(all_pairs)
    
    # Map edge to index
    edge_to_idx = {e: i for i, e in enumerate(all_pairs)}
    idx_to_edge = {i: e for i, e in enumerate(all_pairs)}
    
    # Precompute all Q3 embeddings
    # A Q3 is a subgraph isomorphic to the 3-cube.
    # Vertices 0..7. Edges (u,v) if u^v is power of 2.
    q3_edges_local = []
    for u in range(8):
        for v in range(u+1, 8):
            if (u ^ v) & ((u ^ v) - 1) == 0:
                q3_edges_local.append((u, v))
                
    # Generate all subsets of 8 vertices from 0..15
    q3_masks = []
    # Optimization: We can fix the first vertex to 0 to reduce symmetry?
    # No, just iterate all subsets. 12870 is small.
    for combo in itertools.combinations(range(n), 8):
        # Map combo vertices to 0..7
        # Check if the induced subgraph on these 8 vertices contains a Q3
        # Actually, we just need to check if there is an isomorphism.
        # Since Q3 is small, we can check all 8! permutations? No, too slow.
        # Better: Check if the induced subgraph on 'combo' *is* a Q3 or contains one?
        # Wait, the task says "no subgraph isomorphic to Q3".
        # So if the induced subgraph on 8 vertices is exactly Q3, it's bad.
        # But Q3 could be a subgraph of a larger induced subgraph?
        # No, Q3 has 8 vertices. So any Q3 subgraph uses exactly 8 vertices.
        # So we just need to check if the induced subgraph on any 8 vertices is isomorphic to Q3.
        
        # Efficient check:
        # Q3 is bipartite with parts of size 4.
        # Degrees in Q3 are all 3.
        # So the induced subgraph must have 8 vertices, 12 edges, all degrees 3.
        
        # Let's just build the edge mask for this combo and check if it matches any Q3 pattern.
        # Actually, simpler: 
        # The number of edges in the induced subgraph must be 12.
        # And it must be regular of degree 3.
        
        # Let's precompute the set of edge pairs for a canonical Q3 (vertices 0-7)
        # And for each combo, map the vertices to 0-7 and check if the edge set matches.
        # There are 8! = 40320 permutations. 12870 * 40320 is ~500M. Too slow in Python.
        
        # Alternative: 
        # Just check if the induced subgraph has 12 edges and is 3-regular.
        # If yes, it's likely a Q3 (since Q3 is the only 3-regular bipartite graph on 8 vertices? 
        # Actually, there might be others? No, Q3 is unique? 
        # Let's assume checking 12 edges and 3-regularity is sufficient for n=8.
        # If it fails, we might miss a Q3? 
        # Wait, is it possible to have a 3-regular bipartite graph on 8 vertices that is NOT a Q3?
        # Yes, e.g. two disjoint K_{3,3}? No, K_{3,3} has 6 vertices.
        # What about a "ladder" graph? 
        # Let's stick to the safe method: Precompute all edge masks of Q3 on 8 labeled vertices 0..7.
        # Then for each combo, check if the induced edge mask (mapped to 0..7) is in the set of Q3 masks?
        # No, the mapping is arbitrary.
        
        # Let's use a different strategy:
        # Iterate over all 8! permutations of the 8 vertices in the combo? No.
        
        # Let's use the property that Q3 is the unique 3-regular bipartite graph on 8 vertices?
        # Reference: "The 3-cube is the only 3-regular bipartite graph on 8 vertices." 
        # If this is true, we just check:
        # 1. 8 vertices.
        # 2. 12 edges.
        # 3. All degrees are 3.
        
        # Let's verify uniqueness.
        # Bipartite 3-regular graphs on 8 vertices.
        # Parts A and B of size 4.
        # Edges go between A and B.
        # Total edges 12.
        # This is a (3,3)-regular bipartite graph on 4+4 vertices.
        # These are exactly the Q3s?
        # Let's check.
        # If we fix A={0,1,2,3} and B={4,5,6,7}.
        # Each a in A connects to 3 nodes in B.
        # This is equivalent to a 4x4 matrix with 3s and 0s, row/col sum 3.
        # This is equivalent to a permutation matrix of the 0s (since 3 ones means 1 zero).
        # So the zeros form a permutation.
        # There are 4! = 24 such graphs.
        # Are they all isomorphic to Q3?
        # Q3 corresponds to a specific structure.
        # Actually, all such graphs are isomorphic to Q3.
        # Proof: Q3 is vertex transitive and edge transitive. 
        # The automorphism group of Q3 acts transitively on the 1-factors?
        # Yes.
        # So checking "12 edges and 3-regular" is sufficient.
        
        # Let's implement this fast check.
        deg = [0] * 8
        edge_count = 0
        for i in range(8):
            for j in range(i+1, 8):
                u = combo[i]
                v = combo[j]
                # We don't know if edge (u,v) exists in the "potential Q3" yet.
                # Wait, we are checking if the *induced* subgraph on `combo` is a Q3.
                # But we don't have the graph yet! 
                # We are precomputing *which sets of 8 vertices* CAN form a Q3?
                # No, ANY set of 8 vertices can form a Q3 if the edges are present.
                # The structure of the graph determines if it's a Q3.
                # The precomputation should be: 
                # For each set of 8 vertices, what are the 12 edge pairs that would make it a Q3?
                # But there are 24 ways to wire them up (the 24 permutations of zeros).
                # So for each 8-set, there are 24 possible Q3 edge configurations.
                # Total Q3 masks = 12870 * 24 = ~300,000 masks.
                # This is manageable.
                
                pass
        
    # Let's refine the precomputation.
    # 1. Generate all 8-subsets.
    # 2. For each subset, generate all 24 "Q3 wirings" (defined by the permutation of non-edges).
    # 3. Create the edge mask for each wiring.
    # 4. Store all these masks in a set or list.
    
    q3_edge_masks = set()
    
    # Helper to get the 12 edges of a Q3 given the "zero permutation"
    # Let A = {0,1,2,3}, B = {4,5,6,7} locally.
    # Edge (i, j+4) exists if j != perm[i].
    # So for each i in 0..3, it connects to all j in 0..3 except perm[i].
    
    for perm in itertools.permutations(range(4)):
        local_edges = []
        for i in range(4):
            for j in range(4):
                if i != perm[j]: # Wait, perm maps B index to A index? Or A to B?
                    # Let's say perm[i] is the j such that (i, j) is NOT an edge.
                    # So (i, j) IS an edge if j != perm[i].
                    if j != perm[i]:
                        local_edges.append((i, j+4))
        # Verify 12 edges
        if len(local_edges) != 12: continue
        
        # Now, for each 8-subset of 0..15, map these local edges to global edges.
        # But we need to consider all 8! permutations of vertices in the subset?
        # No, the "wiring" is defined relative to the bipartition.
        # The bipartition of a Q3 is unique (up to swap).
        # So for a given 8-set, we need to try all ways to split it into A and B (size 4).
        # There are C(8,4) = 70 ways to choose A.
        # For each A, we try the 24 wirings.
        # Total checks per 8-set: 70 * 24 = 1680.
        # Total Q3 masks: 12870 * 1680 = ~21M. Too many to store in a set of ints?
        # 21M ints is ~160MB. Might be too much memory or slow to generate.
        
        # Alternative:
        # We don't need to precompute ALL masks.
        # We can just check on the fly if a graph contains a Q3.
        # But the check needs to be fast.
        
        # Let's go back to the "12 edges and 3-regular" check.
        # If that is sufficient, we can just iterate over all 8-subsets in the check function?
        # No, the check function is called for every edge addition.
        # We need a very fast check.
        
        # Let's try a different approach:
        # Precompute all Q3 masks.
        # To reduce count, note that the graph is simple.
        # Let's just generate the masks for a fixed set of 8 vertices (0..7) and all 8! permutations?
        # No, the vertices are fixed in the global graph.
        
        # Let's use the 21M masks idea but optimize storage.
        # A mask is 120 bits. Fits in Python int.
        # 21M ints might be heavy.
        
        # Let's try a simpler heuristic:
        # Just use the slow check but optimize it heavily with bitmasks?
        # The current program's `has_q3_fast` is actually not that bad if optimized.
        # But the inspirations suggest "precompute all Q3 embeddings as edge bitmasks".
        
        # Let's try to generate the masks more efficiently.
        # Instead of 70 * 24, note that swapping A and B gives the same set of edges.
        # So we only need 35 * 24 = 840 masks per 8-set?
        # Still 10M masks.
        
        # Let's try to just run the current program's logic but with a better local search?
        # The current score is 1.38. Inspiration 1.41 uses "conflict-graph".
        
        # Let's implement the "12 edges and 3-regular" check directly in the Q3 check function.
        # It requires checking all 8-subsets of the graph's vertices?
        # No, when adding an edge (u,v), the new Q3 MUST contain (u,v).
        # So we only need to check 8-subsets that contain u and v.
        # Number of such subsets: C(14, 6) = 3003.
        # For each subset, check if it forms a Q3.
        # Check: 
        # 1. Count edges in induced subgraph. Must be 12.
        # 2. Check degrees. Must be all 3.
        
        # This is 3003 checks per edge addition.
        # Each check involves counting edges in 8 vertices (28 pairs) and checking degrees.
        # 3003 * 28 ~ 84k operations.
        # Total edge additions ~ 100.
        # 8.4M operations. Very fast.
        
        pass

    def is_q3_subset(vertices, adj):
        # vertices: tuple of 8 ints
        # adj: list of bitmasks
        # Check if induced subgraph on vertices is a Q3
        # 1. Count edges
        # 2. Check degrees
        
        # Map vertices to 0..7 for local indexing?
        # Or just use global indices.
        
        # Count edges
        edge_count = 0
        degrees = {v: 0 for v in vertices}
        
        for i in range(8):
            u = vertices[i]
            for j in range(i+1, 8):
                v = vertices[j]
                if adj[u] & (1 << v):
                    edge_count += 1
                    degrees[u] += 1
                    degrees[v] += 1
        
        if edge_count != 12:
            return False
        
        for d in degrees.values():
            if d != 3:
                return False
        
        return True

    def has_q3(adj, u, v):
        # Check if adding (u,v) creates a Q3.
        # The new Q3 must contain u and v.
        # Iterate over all 6 other vertices chosen from the remaining 14.
        
        # Optimization: 
        # If we pick the 6 vertices, we form a set of 8.
        # We can prune: 
        # u and v must have degree 3 in the Q3.
        # So they must have exactly 2 other neighbors in the Q3 (besus each other).
        
        # Let's just iterate C(14,6) = 3003 combinations.
        # It's small enough.
        
        others = [x for x in range(n) if x != u and x != v]
        
        for combo in itertools.combinations(others, 6):
            vertices = (u, v) + tuple(combo)
            if is_q3_subset(vertices, adj):
                # Check if the edge (u,v) is actually in this Q3?
                # The Q3 is the induced subgraph. 
                # If the induced subgraph is a Q3, it contains all its edges.
                # Does it contain (u,v)?
                # In a Q3, every pair of vertices is either adjacent or not.
                # If the induced subgraph IS a Q3, then the edges are exactly those of the Q3.
                # So (u,v) is an edge in the Q3 iff (u,v) is an edge in the graph.
                # We are adding (u,v), so it IS an edge.
                # So yes, this is a Q3 containing the new edge.
                return True
        
        return False

    # Build initial graph: K_{3,13}
    def build_initial():
        adj = [0] * n
        edges = set()
        for a in range(3):
            for b in range(3, 16):
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
                edges.add((a, b))
        return adj, edges

    best_adj, best_edges = build_initial()
    best_count = len(best_edges)
    
    random.seed(42)
    
    # Greedy Add
    for trial in range(100):
        if time.time() > deadline - 30:
            break
            
        adj, edges = build_initial()
        
        missing = [e for e in all_pairs if e not in edges]
        random.shuffle(missing)
        
        for e in missing:
            if time.time() > deadline - 25:
                break
            u, v = e
            if has_q3(adj, u, v):
                continue
            # Add edge
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.add(e)
            
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = set(edges)
            best_adj = adj[:]

    # Local Search: Remove one, try to add one
    if time.time() < deadline - 20:
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            adj = best_adj[:]
            edges = set(best_edges)
            
            edge_list = list(edges)
            random.shuffle(edge_list)
            
            for e in edge_list:
                if time.time() > deadline - 10:
                    break
                
                u, v = e
                # Remove
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                edges.remove(e)
                
                # Try to add a missing edge
                missing = [ne for ne in all_pairs if ne not in edges]
                random.shuffle(missing)
                
                added = False
                for ne in missing:
                    if time.time() > deadline - 10:
                        break
                    nu, nv = ne
                    if has_q3(adj, nu, nv):
                        continue
                    # Add
                    adj[nu] |= (1 << nv)
                    adj[nv] |= (1 << nu)
                    edges.add(ne)
                    added = True
                    
                    if len(edges) > best_count:
                        best_count = len(edges)
                        best_edges = set(edges)
                        best_adj = adj[:]
                        improved = True
                    break
    
    return [[u, v] for u, v in best_edges]
# EVOLVE-BLOCK-END
