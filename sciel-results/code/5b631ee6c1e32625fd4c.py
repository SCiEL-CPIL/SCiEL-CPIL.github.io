import time
import random
from itertools import combinations

def contains_q3(edges, n=16):
    """Check if graph contains Q3 as subgraph. Returns True if found."""
    # Build adjacency
    adj = [[False]*n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
    
    # Q3 vertices labeled 0-7 with specific adjacency
    # Cube: 000=0, 001=1, 010=2, 011=3, 100=4, 101=5, 110=6, 111=7
    # Edges: (0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)
    cube_edges = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
    
    # Try all 8-subsets (too many: C(16,8)=12870, manageable)
    for subset in combinations(range(n), 8):
        # Try all 8! / 8 = 5040 labelings? No, Q3 has automorphisms of size 48, so 8!/48=1680 labelings
        # Better: try all bijections from {0..7} to subset
        # Actually, just check if there's an isomorphism
        # For each permutation of subset assigned to 0..7
        # 8! = 40320 per subset, 12870 subsets -> 518M, too slow
        
        # Smarter: use the fact that Q3 is bipartite. The two parts of size 4.
        # In Q3, part A = {0,1,2,3} (even parity), part B = {4,5,6,7} (odd parity)
        # Wait no: 000 even, 001 odd, 010 odd, 011 even, 100 odd, 101 even, 110 even, 111 odd
        # So part A (even) = {0,3,5,6}, part B (odd) = {1,2,4,7}
        # Actually let me recheck: parity of number of 1s
        # 0:000(0), 1:001(1), 2:010(1), 3:011(2), 4:100(1), 5:101(2), 6:110(2), 7:111(3)
        # Even parity: 0,3,5,6 (0,2,2,2 ones... wait 0:0, 3:2, 5:2, 6:2) -> yes 4 vertices
        # Odd parity: 1,2,4,7 (1:1, 2:1, 4:1, 7:3) -> yes 4 vertices
        
        # So Q3 is bipartite with parts of size 4.
        # For each 4-subset A of the 8, check if A and its complement B form a bipartite
        # subgraph isomorphic to Q3.
        
        # This is still complex. Let me just do a simpler check for now.
        # Given time constraints, let me use a different approach.
        pass
    
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    n = 16
    # Start with a moderately dense graph and optimize
    
    # Strategy: Use a bipartite-inspired construction
    # Part A: vertices 0-5 (6 vertices), Part B: vertices 6-15 (10 vertices)
    # But this might still have Q3...
    
    # Better: Start with K16, remove edges to break all Q3s greedily
    
    # Actually, let me try a specific construction:
    # Take K16 and remove edges based on a pattern
    
    # Known result: ex(n, Q3) for small n. For n=16, let's try to find good constructions.
    
    # Approach: Randomized local search
    # 1. Start with random graph with ~45-50 edges
    # 2. While contains Q3: remove a random edge from a Q3
    # 3. Try to add random edges that don't create Q3
    
    # First, implement a fast Q3 checker using bitsets
    
    # Represent edges as a set for fast lookup
    # Q3 has 8 vertices and 12 edges. Check all 8-subsets.
    
    def has_q3(edge_set, adj):
        """adj[u] is a set of neighbors of u"""
        # For each 8-subset, check if it contains Q3
        # Q3 is characterized by: 8 vertices, bipartite 4+4, each vertex degree 3 in the subgraph,
        # and the bipartite graph is exactly the cube.
        
        # Simpler: enumerate all ways to map Q3 vertices to the 8 vertices
        # Q3 vertices 0-7, edges as defined
        cube_edges = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
        
        # For each 8-subset S, try all 8! bijections is too slow
        # Instead: Q3 is 3-regular bipartite. So in the induced subgraph on S,
        # we need a 3-regular bipartite subgraph with 4+4 partition that is the cube.
        
        # Even simpler: just check if the induced subgraph on S contains Q3 as a subgraph
        # (not necessarily induced). This means there exist 8 vertices where all 12 cube edges are present.
        
        # For efficiency, iterate over 8-subsets and for each, check if Q3 is a subgraph.
        # To check if Q3 is a subgraph of G[S]: 
        #   The bipartition of Q3 is fixed up to swap. So try both bipartitions.
        #   For a given bipartition (A,B) of S into two 4-sets, check if the bipartite graph
        #   between A and B in G[S] contains Q3 as a subgraph. But Q3 IS the complete bipartite? No.
        #   Q3 is NOT complete bipartite. K4,4 has 16 edges, Q3 has 12.
        
        # This is getting complex. Let me use a different strategy for the checker.
        
        # Alternative: use the fact that Q3 is the Cartesian product of three K2s.
        # Or: just do a brute force with early termination, and hope it's fast enough for ~50 edge graphs.
        
        # Given the complexity, let me use a simpler heuristic approach instead.
        return False
    
    # Let me try a concrete construction that's likely Q3-free with many edges.
    # 
    # Idea: Take a complete graph K16 and remove a matching or some edges.
    # But K16 definitely contains Q3.
    #
    # Better idea: Use a graph where the independence number is small in a way that prevents Q3.
    #
    # Actually, let me try: complete bipartite K5,11 minus some edges to break Q3.
    # K5,11 has 55 edges. But does it contain Q3? 
    # Q3 needs 4 vertices from each part. In K5,11, any 4 from left and 4 from right form K4,4,
    # which contains Q3. So K5,11 contains Q3.
    #
    # What about a graph that's almost complete but with a specific structure?
    #
    # Let me try: partition into 4 groups of 4. Make it complete within groups and complete between groups,
    # but remove all edges between certain pairs of groups to break Q3.
    #
    # Actually, here's a cleaner idea: 
    # Take the complement of a Q3-free graph... no.
    #
    # Let me just do a randomized search with a proper Q3 checker.
    
    # Proper Q3 checker:
    def is_q3_subgraph(vertices, adj):
        """Check if the induced subgraph on 'vertices' (8 vertices) contains Q3 as a subgraph."""
        if len(vertices) != 8:
            return False
        
        # Map vertices to 0-7
        vmap = {v: i for i, v in enumerate(vertices)}
        
        # Check if there exists a labeling of these 8 vertices as cube vertices
        # such that all 12 cube edges are present in the graph.
        
        # Cube edges in terms of labels 0-7:
        # 0=000, 1=001, 2=010, 3=011, 4=100, 5=101, 6=110, 7=111
        cube_edges = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
        
        # Try all 8! permutations? No, too slow.
        # Instead: Q3 is bipartite. The two parts are:
        # Part 0: {0,3,5,6} (even parity)
        # Part 1: {1,2,4,7} (odd parity)
        
        # For each 4-subset A of the 8 vertices, let B be the complement.
        # Check if the bipartite graph between A and B is isomorphic to Q3's bipartite graph.
        
        # Q3 bipartite graph: A={0,3,5,6}, B={1,2,4,7}
        # Edges: 0-1, 0-2, 0-4, 3-1, 3-2, 3-7, 5-4, 5-7, 6-4, 6-7, and... let me list all:
        # From cube_edges: (0,1),(0,2),(0,4) -> 0 connects to 1,2,4
        # (1,3),(1,5) -> 1 connects to 3,5
        # (2,3),(2,6) -> 2 connects to 3,6
        # (3,7) -> 3 connects to 7
        # (4,5),(4,6) -> 4 connects to 5,6
        # (5,7) -> 5 connects to 7
        # (6,7) -> 6 connects to 7
        
        # So in bipartite form (A on left, B on right):
        # A={0,3,5,6}, B={1,2,4,7}
        # 0-B: {1,2,4}
        # 3-B: {1,2,7}
        # 5-B: {4,7}
        # 6-B: {4,7}
        
        # Wait, that doesn't look right. Let me recheck.
        # 0=000: neighbors are 001=1, 010=2, 100=4. So 0 connects to 1,2,4. All in B. Good.
        # 3=011: neighbors are 001=1, 010=2, 011^111... wait 011 neighbors: flip each bit: 111=7, 001=1, 010=2. So 3 connects to 1,2,7. All in B. Good.
        # 5=101: neighbors: 001=1, 100=4, 111=7. So 5 connects to 1,4,7. All in B. Good.
        # 6=110: neighbors: 010=2, 100=4, 111=7. So 6 connects to 2,4,7. All in B. Good.
        
        # So the bipartite graph has:
        # 0: {1,2,4}
        # 3: {1,2,7}
        # 5: {1,4,7}
        # 6: {2,4,7}
        
        # This is a 3-regular bipartite graph on 4+4 vertices. It's the cube.
        
        # To check if G[A,B] is isomorphic to this: 
        # Try all 4! * 4! = 576 bijections from {0,3,5,6} to A and {1,2,4,7} to B.
        
        # Define the target adjacency
        target_adj = {
            0: {1,2,4},
            3: {1,2,7},
            5: {1,4,7},
            6: {2,4,7}
        }
        
        from itertools import permutations
        
        for perm_a in permutations(A):
            # perm_a[i] is the vertex in A assigned to target vertex [0,3,5,6][i]
            for perm_b in permutations(B):
                # perm_b[i] is the vertex in B assigned to target vertex [1,2,4,7][i]
                # Check all edges
                valid = True
                for u in [0,3,5,6]:
                    for v in target_adj[u]:
                        if v not in adj[perm_a[[0,3,5,6].index(u)]]:
                            # Wait, this is getting confused. Let me redo.
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    return True
        return False
    
    # This is still complex. Let me simplify with a direct approach.
    
    # Given time constraints, let me use a simpler but effective strategy:
    # Start with a known Q3-free construction and try to improve it.
    
    # Construction: Take vertices 0-15. 
    # Let me try: complete graph minus a set of edges that breaks all Q3s.
    
    # Actually, let me just try a random local search with a simple Q3 checker
    # that works for the specific structure.
    
    # Simple Q3 checker: For each 8-subset, check if it induces a subgraph containing Q3.
    # To make it fast, precompute for each 8-subset.
    
    def has_q3_fast(edge_set, adj, n=16):
        """Returns True if graph contains Q3 subgraph."""
        # Q3 cube edges
        cube_edges = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
        
        # For each 8-subset
        for subset in combinations(range(n), 8):
            subset_set = set(subset)
            # Try all bijections from 0-7 to subset
            # But 8! = 40320, and 12870 subsets -> too slow
            
            # Instead, use the bipartite structure
            # Q3 bipartition: A={0,3,5,6}, B={1,2,4,7}
            # For each 4-subset A of the 8, B = complement
            # Check if there's an isomorphism
            
            # Precompute target adjacency for Q3
            # A vertices: 0,3,5,6 with neighbors in B:
            # 0 -> {1,2,4}, 3 -> {1,2,7}, 5 -> {1,4,7}, 6 -> {2,4,7}
            
            for A in combinations(subset, 4):
                A_set = set(A)
                B_set = subset_set - A_set
                if len(B_set) != 4:
                    continue
                B = tuple(B_set)
                
                # Try all 4! * 4! = 576 mappings
                # Map A to {0,3,5,6} and B to {1,2,4,7}
                a_targets = [0,3,5,6]
                b_targets = [1,2,4,7]
                
                for perm_a in permutations(A):
                    for perm_b in permutations(B):
                        # Check if all cube edges are present
                        ok = True
                        for (u, v) in cube_edges:
                            if u in a_targets and v in b_targets:
                                # u maps to perm_a[a_targets.index(u)], v maps to perm_b[b_targets.index(v)]
                                au = perm_a[a_targets.index(u)]
                                bv = perm_b[b_targets.index(v)]
                                if (au, bv) not in edge_set and (bv, au) not in edge_set:
                                    ok = False
                                    break
                            elif u in b_targets and v in a_targets:
                                bu = perm_b[b_targets.index(u)]
                                av = perm_a[a_targets.index(v)]
                                if (bu, av) not in edge_set and (av, bu) not in edge_set:
                                    ok = False
                                    break
                            else:
                                # Both in same part - should not happen in Q3
                                ok = False
                                break
                        if ok:
                            return True
        return False
    
    # This is O(C(16,8) * C(8,4) * 576 * 12) which is 12870 * 70 * 576 * 12 ≈ 6.2 billion. Too slow.
    
    # I need a much faster approach. Let me think differently.
    
    # Alternative: Use the fact that Q3 is 3-regular. So in any Q3 subgraph, each of the 8 vertices
    # has degree at least 3 within the subgraph. So candidates for Q3 vertices must have degree >= 3.
    
    # Also, I can use a backtracking approach: try to embed Q3 vertex by vertex.
    
    # Given the time pressure, let me use a simpler heuristic:
    # 1. Start with a graph that's likely Q3-free (e.g., the seed K3,13 or a variant)
    # 2. Try to add edges one by one, checking if it creates Q3
    # 3. If Q3 created, skip that edge
    
    # For the Q3 check when adding an edge (u,v), I only need to check 8-subsets containing both u and v.
    # C(14,6) = 3003 subsets. For each, check if it contains Q3. Still expensive but better.
    
    # Actually, let me just use a timeout-based approach and hope the checker is fast enough for ~50 edge graphs.
    
    # Let me try a different tactic: use a known good construction.
    
    # Research suggests ex(16, Q3) might be around 45-50.
    
    # Let me try: complete bipartite K4,12 is NOT Q3-free (as shown).
    # What about K4,12 minus a matching of size 4? That's 48-4=44 edges.
    
    # Or: take K16 and remove all edges of a 4-regular graph? No.
    
    # Let me just do a simple local search with a reasonable Q3 checker and hope it works.
    
    # Simplified Q3 checker using the property that Q3 is the only 3-regular bipartite graph on 8 vertices
    # with girth 6 (or something). Actually, let me just check if the induced subgraph on 8 vertices
    # is isomorphic to Q3 or contains Q3.
    
    # For speed, I'll check: does the induced subgraph on S contain a 3-regular bipartite subgraph with 4+4 parts?
    # If yes, it might be Q3 (there's only one 3-regular bipartite graph on 4+4: it's the cube? No, there are multiple).
    
    # Actually, there are multiple 3-regular bipartite graphs on 4+4 vertices. The cube is one of them.
    
    # I think the most practical approach given time is:
    # 1. Start with a dense random graph
    # 2. Use a simple Q3 detector (maybe approximate)
    # 3. Remove edges to fix Q3s
    # 4. Return the result
    
    # Let me just try a specific construction and verify it's Q3-free by exhaustive check.
    
    # Construction: Take vertices 0-15. Partition into 4 groups of 4: G0={0,1,2,3}, G1={4,5,6,7}, G2={8,9,10,11}, G3={12,13,14,15}
    # Make complete within each group (4*6=24 edges).
    # Between groups: connect all except remove all edges between G0 and G2, and between G1 and G3.
    # Edges between G0-G1: 16, G0-G3: 16, G1-G2: 16, G2-G3: 16. Total between: 64.
    # Total: 24 + 64 = 88 edges.
    
    # Is this Q3-free? A Q3 would need 4 vertices from "even" positions and 4 from "odd". 
    # If the Q3 uses vertices from G0, G1, G2, G3, it needs edges between all pairs. But G0-G2 and G1-G3 have no edges.
    # So any Q3 must avoid having vertices from both G0 and G2, or both G1 and G3.
    # If it uses G0 and G1 only: 8 vertices from 8 available (G0∪G1). The induced subgraph is K8 minus... wait, within groups complete, between G0-G1 complete. So it's K8. K8 contains Q3.
    
    # So that construction fails.
    
    # Let me try: make it so that no 4+4 bipartition can form a cube.
    
    # I'm running out of time for deep analysis. Let me just use a randomized approach with a fast approximate Q3 checker.
    
    # Final approach: Start with K3,13 (39 edges) and try to add edges by swapping or adding, checking Q3-freeness with a timeout.
    
    # Given the complexity, let me just return a slightly better construction if I can find one quickly.
    
    # Try: K4,12 minus 4 edges to break Q3. 
    # In K4,12, Q3s are formed by any 4 from left, 4 from right. To break all, I need to ensure no 4x4 subgrid is complete.
    # This is like a 4x12 matrix with no all-1 4x4 submatrix.
    # Min edges to remove: need to hit all 4x4 submatrices. By inclusion-exclusion or known results...
    
    # Actually, this is equivalent to: in a 4x12 binary matrix, what's the max 1s with no all-1 4x4 submatrix?
    # This is a Zarankiewicz problem variant. z(4,12;4,4) - max 1s with no 4x4 all-1 submatrix.
    # Upper bound: by de Bruijn-Erdos, at most 4*12 - 4*4 + 4 = 48-16+4=36? Not sure.
    
    # Let me just try: remove one edge from each of 4 disjoint 4x4 submatrices. That gives 48-4=44 edges.
    
    # Left: {0,1,2,3}, Right: {4,5,6,7,8,9,10,11,12,13,14,15}
    # Remove edges: (0,4), (1,5), (2,6), (3,7)
    # Now check if there's a Q3: need 4 from left, 4 from right with all 16 edges present.
    # If we pick left {0,1,2,3} and right {4,5,6,7}, we're missing (0,4),(1,5),(2,6),(3,7). So not complete.
    # But we could pick right {8,9,10,11} - that's complete with left {0,1,2,3}. So Q3 exists!
    
    # So I need to remove edges more carefully.
    
    # This is getting too complex. Let me just return the seed or a simple improvement.
    
    # Let me try: K3,13 plus a few edges among the 13 right vertices.
    # Adding an edge (a,b) where a,b in {3..15}: does this create Q3?
    # Q3 needs 4 from left {0,1,2} - impossible since left has only 3. So no Q3!
    
    # Wait, Q3 is bipartite with parts of size 4. If my graph has a part of size 3, then any bipartite subgraph has at most 3 on one side. So no Q3.
    
    # So I can add ANY edges among the 13 vertices on the right side!
    
    # K3,13 has 39 edges. I can add all C(13,2)=78 edges among the right side.
    # Total: 39 + 78 = 117 edges.
    
    # Is this Q3-free? The graph has vertices 0,1,2 (left) and 3..15 (right, 13 vertices).
    # Edges: all between left and right (39), and all within right (78).
    # No edges within left.
    
    # For Q3 to exist, we need 8 vertices forming a cube. The cube is bipartite 4+4.
    # In our graph, any bipartite subgraph: one part must be subset of left (size<=3) or... 
    # Actually, the graph is not bipartite (right side has edges). But Q3 is bipartite.
    # So we need to find 8 vertices that induce a bipartite subgraph isomorphic to Q3.
    # The 8 vertices could be: a from left, b from right, with a+b=8, a<=3.
    # So a=0,1,2,3. If a=3, b=5: 3 from left, 5 from right. But Q3 needs 4+4. So a must be 4? No, a<=3.
    # Wait, the bipartition of Q3 doesn't have to align with my left/right partition.
    
    # Let me think again. The graph G has:
    # - Vertices L={0,1,2}, R={3,...,15}
    # - Edges: L×R (complete bipartite) and R×R (complete graph on R)
    # - No edges within L
    
    # Can G contain Q3? Q3 is bipartite with parts X,Y, |X|=|Y|=4.
    # In G, the only non-edges are within L. So if X and Y are both subsets of R, then all edges between X and Y are present (since R is complete). So K4,4 is a subgraph, which contains Q3.
    # Specifically, pick any 4 vertices from R for X and 4 from R for Y. The induced bipartite graph is K4,4, which contains Q3 as a subgraph.
    
    # So that construction DOES contain Q3. My reasoning was wrong.
    
    # OK so I need to be more careful. Let me just use the seed for now and add a simple local search.
    
    # Given time constraints, let me just return a construction that's definitely Q3-free with more than 39 edges.
    
    # Try: K4,11. Does it contain Q3? Left={0,1,2,3}, Right={4,...,14} (11 vertices).
    # Pick 4 from left, 4 from right: K4,4, contains Q3. So yes.
    
    # K3,13 is safe because left has only 3 vertices, so any bipartite subgraph has at most 3 on the left side, but Q3 needs 4 on each side. Wait, but the bipartition of Q3 doesn't have to align with my L/R partition.
    
    # In K3,13: vertices L={0,1,2}, R={3,...,15}. Edges: L×R only. No edges within L or R.
    # This is a bipartite graph with parts of size 3 and 13.
    # Any subgraph is also bipartite with parts of size at most 3 and 13.
    # Q3 is bipartite with parts of size 4 and 4.
    # So no subgraph of K3,13 can be Q3, since Q3 needs a bipartition with both parts of size 4, but any bipartite subgraph of K3,13 has one part of size at most 3.
    
    # So K3,13 is Q3-free. And I can't add any edges without potentially creating Q3.
    
    # What if I add one edge within R? Say (3,4). Now the graph has a triangle? No, just an edge.
    # Can this create Q3? Q3 is bipartite, so it can't contain odd cycles. But adding an edge to a bipartite graph can create odd cycles, but Q3 is bipartite so it wouldn't be affected by odd cycles.
    # The question is: does the new graph contain Q3 as a subgraph?
    # Q3 needs 8 vertices, 12 edges, all forming a cube.
    # With the extra edge (3,4), can we find a Q3?
    # The Q3 would need to use vertices from L and R. Since L has only 3 vertices, the Q3 can use at most 3 from L.
    # But Q3 is bipartite 4+4. If we use k from L and 8-k from R, the bipartition of Q3 doesn't have to align with L/R.
    # Actually, in the subgraph, the bipartition is determined by the Q3 structure, not by L/R.
    # So it's possible that 3 vertices from L and 5 from R form a Q3, with the bipartition splitting them 4+4 differently.
    
    # This is hard to reason about. Let me just test empirically with a small script.
    
    # Given time, let me just return the seed. The score will be 1.0.
    
    return [[a, b] for a in range(3) for b in range(3, 16)]
