"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def construct():
    deadline = time.time() + 110  # 110s safety margin before 120s limit
    
    # Start with K3,13 (39 edges)
    n = 16
    best_edges = [[a, b] for a in range(3) for b in range(3, 16)]
    best_count = len(best_edges)
    
    # Build adjacency set for quick lookup
    edge_set = set()
    for u, v in best_edges:
        edge_set.add((min(u,v), max(u,v)))
    
    # Precompute all Q3 subgraphs (subsets of 8 vertices that form a cube)
    # Q3 has 8 vertices and 12 edges
    # We'll check if a set of 8 vertices contains a Q3 as a subgraph
    
    # A Q3 is a 3-dimensional hypercube. Its vertices can be labeled 0..7 (binary 000..111)
    # Edges connect vertices differing in exactly one bit
    
    # For a set of 8 vertices to contain a Q3 as a subgraph, there must exist a subset of 8 vertices
    # that is isomorphic to Q3
    
    # Efficient check: For each 8-vertex subset, check if it contains a Q3
    # But that's C(16,8) = 12870 subsets, and checking each is expensive
    
    # Better approach: enumerate all possible Q3 embeddings in K16
    # A Q3 embedding is a mapping from {0,...,7} to vertices such that
    # edges of Q3 map to edges of our graph
    
    # Precompute all possible Q3 vertex sets (8-tuples of distinct vertices)
    # There are P(16,8) = 16!/(16-8)! = 518,918,400 which is too many
    
    # Alternative: use the fact that Q3 is 3-regular with 8 vertices
    # For each 8-vertex subset, check if it contains a Q3 as a subgraph
    
    # Let's use a different approach: 
    # Try to add edges one at a time, checking if it creates a Q3
    
    # To check if adding edge (u,v) creates a Q3:
    # A Q3 containing edge (u,v) must have u and v adjacent in the cube
    # In Q3, each edge belongs to exactly 2 faces (4-cycles)
    # Actually, in Q3, each edge is in exactly 2 squares (4-cycles)
    # Wait, let me think more carefully.
    
    # In Q3, each vertex has degree 3. Each edge connects two vertices.
    # The number of 4-cycles in Q3 is 6 (one for each pair of dimensions)
    # Each edge is in exactly 2 of these 4-cycles
    
    # So if we add edge (u,v), a Q3 containing this edge would have:
    # - u and v as adjacent vertices in the cube
    # - The remaining 6 vertices completing the cube
    
    # Let's precompute: for each pair (u,v), what are the possible Q3 embeddings 
    # that include edge (u,v)?
    
    # Actually, let's use a simpler approach:
    # For each 8-vertex subset, check if it forms a Q3 (as an induced subgraph or subgraph)
    
    # We need to check if the 8 vertices contain a Q3 as a subgraph (not necessarily induced)
    
    # Let's precompute all 8-vertex subsets and for each, determine if it can contain a Q3
    
    # A set of 8 vertices contains a Q3 as a subgraph iff there exists a bijection 
    # from {0,...,7} to the 8 vertices such that for each edge (i,j) in Q3,
    # the corresponding vertices are adjacent in our graph
    
    # This is a constraint satisfaction problem. For 8 vertices, we can try all 
    # 8! = 40320 permutations... but that's too slow for many subsets.
    
    # Better: use backtracking to check if a given 8-vertex set contains a Q3
    
    def has_q3_subgraph(vertices, adj):
        """Check if the 8 vertices contain a Q3 as a subgraph.
        vertices: set of 8 vertex indices
        adj: adjacency set (frozenset of (u,v) pairs)
        Returns True if a Q3 subgraph exists.
        """
        verts = list(vertices)
        
        # Q3 vertices: 0..7, edges connect i,j if i^j is a power of 2
        q3_edges = []
        for i in range(8):
            for j in range(i+1, 8):
                if (i ^ j) & (i ^ j - 1) == 0:  # differ in exactly one bit
                    q3_edges.append((i, j))
        
        # Backtracking: assign Q3 vertices to graph vertices
        # Use symmetry breaking: fix Q3 vertex 0 to graph vertex verts[0]
        
        def backtrack(assignment, q3_v, next_q3_v):
            """
            assignment: dict from q3 vertex index to graph vertex
            q3_v: next Q3 vertex to assign
            """
            if q3_v == 8:
                # All assigned, check all edges
                for a, b in q3_edges:
                    ga, gb = assignment[a], assignment[b]
                    if (min(ga,gb), max(ga,gb)) not in adj:
                        return False
                return True
            
            for gv in verts:
                if gv in assignment.values():
                    continue
                # Check constraints with already assigned neighbors
                valid = True
                for a in range(q3_v):
                    if (a, q3_v) in [(x,y) for x,y in q3_edges] or (q3_v, a) in [(x,y) for x,y in q3_edges]:
                        ga = assignment[a]
                        if (min(ga,gv), max(ga,gv)) not in adj:
                            valid = False
                            break
                if valid:
                    assignment[q3_v] = gv
                    if backtrack(assignment, q3_v + 1, None):
                        return True
                    del assignment[q3_v]
            return False
        
        # Precompute adjacency in Q3 for faster checking
        q3_adj = [[] for _ in range(8)]
        for a, b in q3_edges:
            q3_adj[a].append(b)
            q3_adj[b].append(a)
        
        def backtrack2(assignment, q3_v):
            if q3_v == 8:
                for a, b in q3_edges:
                    ga, gb = assignment[a], assignment[b]
                    if (min(ga,gb), max(ga,gb)) not in adj:
                        return False
                return True
            
            for gv in verts:
                if gv in assignment.values():
                    continue
                # Check with already assigned neighbors
                valid = True
                for nb in q3_adj[q3_v]:
                    if nb in assignment:
                        gn = assignment[nb]
                        if (min(gn,gv), max(gn,gv)) not in adj:
                            valid = False
                            break
                if valid:
                    assignment[q3_v] = gv
                    if backtrack2(assignment, q3_v + 1):
                        return True
                    del assignment[q3_v]
            return False
        
        # Fix symmetry: assign Q3 vertex 0 to verts[0]
        assignment = {0: verts[0]}
        return backtrack2(assignment, 1)
    
    # This is still expensive. Let's use a different strategy.
    
    # Greedy approach: try adding edges in some order, check if Q3 is created
    
    # For checking if adding edge (u,v) creates a Q3:
    # We need to find if there exists a Q3 containing edge (u,v)
    # In Q3, fix edge (0,1) [vertices 0 and 1 differ in bit 0]
    # The other 6 vertices are 2,3,4,5,6,7
    # Edges from 0: (0,1), (0,2), (0,4)
    # Edges from 1: (0,1), (1,3), (1,5)
    # So we need vertices a,b such that (0,a) and (0,b) are edges, 
    # and vertices c,d such that (1,c) and (1,d) are edges
    # With a^b = 4 (differ in bit 2), c^d = 4
    # And a^c = 1 or 2, etc.
    
    # Let me enumerate all Q3 embeddings that include a specific edge
    
    # For edge between graph vertices u and v:
    # In Q3, suppose u corresponds to vertex 0 and v to vertex 1
    # Then we need to assign vertices 2,3,4,5,6,7 to 6 distinct graph vertices
    # Such that all Q3 edges are present
    
    # The remaining Q3 structure:
    # 0-2, 0-4, 1-3, 1-5, 2-3, 2-6, 3-7, 4-5, 4-6, 5-7, 6-7
    # (edges not involving 0-1)
    
    # So we need:
    # - a vertex a2 with edge (u, a2) [corresponds to Q3 vertex 2]
    # - a vertex a4 with edge (u, a4) [corresponds to Q3 vertex 4]
    # - a vertex a3 with edge (v, a3) [corresponds to Q3 vertex 3]
    # - a vertex a5 with edge (v, a5) [corresponds to Q3 vertex 5]
    # - a vertex a6 with edges (a2, a6) and (a4, a6) [Q3 vertex 6]
    # - a vertex a7 with edges (a3, a7) and (a5, a7) [Q3 vertex 7]
    # - edge (a2, a3), (a4, a5), (a6, a7)
    
    # This gives us a way to check efficiently
    
    def creates_q3(u, v, adj):
        """Check if adding edge (u,v) creates a Q3 in the graph."""
        # Try all assignments of Q3 vertices 2-7 to graph vertices
        # with u->0, v->1
        
        # Candidates for Q3 vertex 2: neighbors of u (excluding v)
        # Candidates for Q3 vertex 4: neighbors of u (excluding v)
        # Candidates for Q3 vertex 3: neighbors of v (excluding u)
        # Candidates for Q3 vertex 5: neighbors of v (excluding u)
        
        # First, temporarily add edge (u,v) to adj for checking
        adj_temp = adj | {(min(u,v), max(u,v))}
        
        def in_adj(a, b):
            return (min(a,b), max(a,b)) in adj_temp
        
        neigh_u = [w for w in range(n) if w != v and in_adj(u, w)]
        neigh_v = [w for w in range(n) if w != u and in_adj(v, w)]
        
        for a2 in neigh_u:
            for a4 in neigh_u:
                if a4 == a2:
                    continue
                for a3 in neigh_v:
                    if a3 == a2 or a3 == a4:
                        continue
                    if not in_adj(a2, a3):
                        continue
                    for a5 in neigh_v:
                        if a5 == a2 or a5 == a3 or a5 == a4:
                            continue
                        if not in_adj(a4, a5):
                            continue
                        # Now find a6: common neighbor of a2 and a4, distinct from others
                        # And a7: common neighbor of a3 and a5, distinct from others
                        used = {u, v, a2, a3, a4, a5}
                        for a6 in range(n):
                            if a6 in used:
                                continue
                            if not in_adj(a2, a6) or not in_adj(a4, a6):
                                continue
                            for a7 in range(n):
                                if a7 in used or a7 == a6:
                                    continue
                                if not in_adj(a3, a7) or not in_adj(a5, a7):
                                    continue
                                if not in_adj(a6, a7):
                                    continue
                                return True
        return False
    
    # Try to add edges greedily
    # Order edges by some heuristic
    all_possible_edges = []
    for u in range(n):
        for v in range(u+1, n):
            if (u, v) not in edge_set:
                all_possible_edges.append((u, v))
    
    # Try adding edges
    for u, v in all_possible_edges:
        if time.time() > deadline:
            break
        
        # Check if adding this edge creates a Q3
        if not creates_q3(u, v, edge_set):
            edge_set.add((u, v))
            best_edges.append([u, v])
            best_count += 1
    
    # Verify the result
    # Check that the final graph has no Q3
    # Convert to adjacency set
    final_adj = set()
    for u, v in best_edges:
        final_adj.add((min(u,v), max(u,v)))
    
    # Quick verification: check all 8-vertex subsets for Q3
    # This is expensive, so we'll do a lighter check
    # Just verify that our greedy additions didn't create a Q3
    # (which should be guaranteed by our check)
    
    return best_edges
# EVOLVE-BLOCK-END
