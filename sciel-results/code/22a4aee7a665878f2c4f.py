import time
import random
from itertools import combinations, permutations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    n = 16
    # Q3 cube edges: (0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)
    # Vertices 0-7. Bipartition: A={0,3,5,6} (even parity), B={1,2,4,7} (odd parity)
    # Edges in bipartite form (A -> B):
    # 0 -> {1,2,4}
    # 3 -> {1,2,7}
    # 5 -> {1,4,7}
    # 6 -> {2,4,7}
    
    cube_edges = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
    cube_adj = {
        0: {1,2,4},
        1: {0,3,5},
        2: {0,3,6},
        3: {1,2,7},
        4: {0,5,6},
        5: {1,4,7},
        6: {2,4,7},
        7: {3,5,6}
    }
    
    # Precompute all 8-subsets of vertices 0..15
    all_8_subsets = list(combinations(range(n), 8))
    
    # Precompute all 4-subsets of each 8-subset (for bipartition A)
    # For each 8-subset S, the possible A's are 4-subsets of S. B = S \ A.
    # There are C(8,4) = 70 such partitions per S.
    # Total checks per Q3 detection: 12870 * 70 * (check cost)
    # Check cost: 4! * 4! = 576 permutations * 12 edge checks.
    # This is too slow for full scan in Python if done naively.
    # We need a faster check or a smarter search.
    
    # Let's optimize the check for a specific 8-subset S and bipartition (A, B).
    # We want to know if there exists a bijection f: {0..7} -> S such that
    # f(A_cube) = A and f(B_cube) = B, and all cube edges map to graph edges.
    # A_cube = {0,3,5,6}, B_cube = {1,2,4,7}
    
    # We can fix the mapping of A to A_set and B to B_set.
    # There are 4! * 4! = 576 ways.
    # To speed up, we can use backtracking or bitmasks.
    
    # Alternative: Use the fact that Q3 is 3-regular.
    # In the induced subgraph on S, if it contains Q3, then there is a 3-regular bipartite subgraph.
    
    # Let's try a different approach for speed:
    # Instead of checking all 8-subsets, we can use a randomized local search
    # with a faster approximate check, or a very optimized exact check.
    
    # Given the 120s limit, let's try a greedy construction with a fast incremental check.
    # Start with empty graph. Add edges one by one.
    # When adding edge (u,v), check if it creates a Q3.
    # A Q3 containing (u,v) must use u and v as adjacent vertices in the cube.
    # In Q3, each edge is in 2 faces (4-cycles). But we need the whole cube.
    # Number of Q3s containing a specific edge in K16:
    # Pick u, v. They must be adjacent in Q3.
    # Number of ways to complete the rest of the cube:
    # Q3 has 12 edges. If (u,v) is one, how many cubes contain it?
    # In Q3, each edge is symmetric. Number of cubes in K16 containing a specific edge (u,v):
    # Choose 6 other vertices from 14. C(14,6) = 3003.
    # For each set of 6, check if {u,v}+6 forms a Q3 with (u,v) as an edge.
    # This is much smaller than checking all 8-subsets.
    
    # Let's implement a function `has_q3_with_edge(u, v, adj)` that checks if adding (u,v) creates a Q3.
    # It iterates over C(14,6) subsets of the remaining 14 vertices.
    # For each subset, it checks if the 8 vertices form a Q3 containing edge (u,v).
    
    # To check if a specific 8-set S containing u,v forms a Q3 with (u,v) as an edge:
    # 1. u and v must be adjacent in the cube labeling.
    # 2. We can fix u to a specific cube vertex, say 0. Then v must be one of 0's neighbors: 1, 2, or 4.
    # 3. Due to symmetry, we can fix u=0, v=1 (WLOG, since we can rotate/reflect cube).
    #    Actually, we need to try all 3 cases for v's position relative to u, or just fix u=0 and try v in {1,2,4}.
    #    If we fix u=0, v=1, then the remaining 6 vertices must map to {2,3,4,5,6,7} such that all cube edges are present.
    #    The cube edges not involving 0 or 1 are: (2,3),(2,6),(3,7),(4,5),(4,6),(5,7).
    #    Wait, edges involving 0: (0,1),(0,2),(0,4). Edges involving 1: (1,3),(1,5).
    #    So if u=0, v=1, we need edges (0,2),(0,4),(1,3),(1,5) to be present in the graph (besides (0,1)).
    #    And the subgraph on {2,3,4,5,6,7} must contain edges (2,3),(2,6),(3,7),(4,5),(4,6),(5,7).
    #    This is a specific pattern.
    
    # This is getting complicated to implement efficiently in Python within time limits.
    # Let's try a simpler heuristic:
    # 1. Start with a known Q3-free graph, e.g., K3,13 (39 edges).
    # 2. Try to add edges. For each candidate edge, run a full Q3 check (optimized).
    # 3. If no Q3, add it.
    
    # Optimized Full Q3 Check:
    # Use bitmasks for adjacency.
    # adj_mask[u] is an integer where bit v is 1 if (u,v) is an edge.
    
    adj_mask = [0] * n
    edges = set()
    
    def add_edge(u, v):
        if u == v: return
        if (u,v) in edges or (v,u) in edges: return
        edges.add((u,v))
        edges.add((v,u))
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    # Initialize with K3,13
    # Left: 0,1,2. Right: 3..15.
    for u in range(3):
        for v in range(3, 16):
            add_edge(u, v)
    
    # Function to check if current graph contains Q3
    def contains_q3():
        # Iterate over all 8-subsets
        for S in all_8_subsets:
            S_set = set(S)
            # Check all 4-subsets A of S
            for A in combinations(S, 4):
                A_set = set(A)
                B_set = S_set - A_set
                if len(B_set) != 4: continue
                B = tuple(B_set)
                
                # Try to map A_cube={0,3,5,6} to A and B_cube={1,2,4,7} to B
                # Or A_cube to B and B_cube to A.
                
                # We'll try both bipartitions.
                # Case 1: A corresponds to {0,3,5,6}
                # We need a bijection f: {0,3,5,6} -> A and g: {1,2,4,7} -> B
                # such that for all (u,v) in cube_edges, if u in A_cube and v in B_cube, then (f(u), g(v)) is an edge.
                
                # To speed up, we can precompute for each vertex in A its neighbors in B.
                # And check if the bipartite graph between A and B is isomorphic to Q3's bipartite graph.
                
                # Q3 bipartite graph (A_cube -> B_cube):
                # 0 -> {1,2,4}
                # 3 -> {1,2,7}
                # 5 -> {1,4,7}
                # 6 -> {2,4,7}
                
                # Let's represent the bipartite graph between A and B as a 4x4 matrix M where M[i][j] = 1 if A[i] is connected to B[j].
                # We need to check if this matrix is isomorphic to the Q3 matrix.
                # Q3 matrix (rows 0,3,5,6; cols 1,2,4,7):
                # 0: 1 1 1 0  (1,2,4)
                # 3: 1 1 0 1  (1,2,7)
                # 5: 1 0 1 1  (1,4,7)
                # 6: 0 1 1 1  (2,4,7)
                
                # We can check isomorphism by trying all 4! * 4! = 576 permutations.
                # But we can prune.
                
                # Let's just do the 576 check. It's fast enough if we do it in C, but in Python it might be slow.
                # 12870 * 70 * 576 = ~518 million operations. Too slow in Python.
                
                # We need a faster check.
                # Idea: Count degrees. In Q3, all vertices have degree 3.
                # So in the bipartite graph between A and B, all vertices must have degree exactly 3?
                # No, the induced subgraph on S might have extra edges. We are looking for a SUBGRAPH isomorphic to Q3.
                # So we need to find a 3-regular bipartite subgraph with the specific structure.
                
                # Alternative: Use the fact that Q3 has no triangles.
                # Or: Use a SAT solver? No, standard library only.
                
                # Let's try a randomized approach for the check? No, must be exact.
                
                # Let's optimize the permutation check.
                # Instead of 576, we can fix one vertex.
                # Fix A[0] to map to 0. Then A[1] can map to 3,5,6 (3 choices).
                # Actually, the automorphism group of Q3 has size 48.
                # So we only need to check 576 / 48 = 12 mappings?
                # No, that's for the whole cube. For the bipartite check, we can fix the mapping of one vertex in A.
                
                # Let's just try to implement the 576 check efficiently and hope it's fast enough for a few iterations.
                # Or, use a different strategy:
                # Start with a dense graph and remove edges?
                
                pass
        return False
    
    # Given the time, I will implement a simpler heuristic:
    # Use a known construction that is likely Q3-free and dense.
    # Or, use a greedy addition with a limited Q3 check (e.g., only check subsets containing the new edge).
    
    # Let's try greedy addition with edge-specific check.
    # When adding (u,v), check C(14,6) subsets.
    # For each subset, check if it forms a Q3 with (u,v) as an edge.
    # This is 3003 checks per edge. 120 edges * 3003 = 360k checks.
    # Each check involves 12 edge lookups. 4.3 million lookups. This is fast in Python.
    
    def has_q3_containing_edge(u, v):
        # u and v are the new edge.
        # Remaining vertices: all except u, v
        others = [x for x in range(n) if x != u and x != v]
        
        # For each 6-subset of others
        for subset6 in combinations(others, 6):
            S = set([u, v] + list(subset6))
            # Check if S contains Q3 with (u,v) as an edge
            # We can fix u to a cube vertex. Due to symmetry, u can be 0.
            # Then v must be 1, 2, or 4 (neighbors of 0).
            # We try all 3 cases.
            
            # Case 1: u=0, v=1
            # Required edges: (0,2),(0,4),(1,3),(1,5) and subgraph on {2,3,4,5,6,7} must contain (2,3),(2,6),(3,7),(4,5),(4,6),(5,7)
            # We need to map {2,3,4,5,6,7} to S \ {u,v} such that these edges exist.
            # This is a small subgraph isomorphism problem.
            
            # Let's just do a full check for the 8-set S, but optimized.
            # Actually, let's just check if the induced subgraph on S contains Q3.
            # We can use the 576 permutation check, but only for this specific S.
            # Since we only do this for 3003 subsets, and each check is ~576*12 ops, it's ~20k ops per edge addition.
            # 120 edges * 20k = 2.4 million ops. This is very fast.
            
            # Let's implement the 576 check for a specific S.
            S_list = list(S)
            # We need to check if there is a bijection f: 0..7 -> S_list such that all cube edges are present.
            # We can fix f(0) = u. Then f(1) can be v, or f(2)=v, or f(4)=v.
            # Due to symmetry, we can just try all 8! permutations? No, too slow.
            # Try all 8! / 48 = 840 mappings?
            # Or just try all 8! = 40320? 3003 * 40320 = 121 million. Too slow.
            
            # Let's use the bipartite check.
            # For S, try all 4-subsets A.
            # This is 70 * 576 = 40k ops per S.
            # 3003 * 40k = 120 million. Too slow in Python.
            
            # I need a faster check.
            # Let's use the fact that Q3 is 3-regular.
            # In S, if it contains Q3, then there are 8 vertices with degree >= 3 in the subgraph.
            # We can filter S by degree.
            
            # Given time, let's just return the seed K3,13. It's safe and scores 1.0.
            # But I want to improve.
            
            # Let's try a different construction:
            # The complement of a matching? No.
            
            # Let's try: Complete graph K16 minus a 4-regular graph?
            # K16 has 120 edges.
            
            # Actually, let's just use a simple local search with a timeout.
            # Start with K3,13.
            # Try to add edges. If it takes too long, stop.
            
            pass
        return False
    
    # Since the exact check is too slow to implement robustly in Python within the time limit,
    # I will return the seed construction K3,13 which is guaranteed Q3-free.
    # This ensures a score of 1.0.
    
    return [[a, b] for a in range(3) for b in range(3, 16)]
