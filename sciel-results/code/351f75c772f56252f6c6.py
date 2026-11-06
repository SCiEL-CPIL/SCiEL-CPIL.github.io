"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    n = 16
    # Total possible edges in K16
    all_edges = list(itertools.combinations(range(n), 2))
    edge_index = {e: i for i, e in enumerate(all_edges)}
    m = len(all_edges) # 120
    
    # Precompute all embeddings of Q3 into K16
    # Q3 is the cube graph. Vertices can be represented by 3-bit integers 0..7.
    # Edges in Q3 connect vertices differing by exactly one bit.
    q3_vertices = list(range(8))
    q3_edges = []
    for u in range(8):
        for v in range(u+1, 8):
            # Check if u and v differ by exactly one bit
            diff = u ^ v
            if diff & (diff - 1) == 0: # power of 2
                q3_edges.append((u, v))
    
    # An embedding is a bijection phi: V(Q3) -> V(G) such that for all (u,v) in E(Q3), (phi(u), phi(v)) in E(G)
    # Since G starts as K16, every injection is an embedding.
    # We need to count, for each edge e in G, how many embeddings contain e.
    
    # Optimization: Instead of iterating all 16P8 permutations (huge), 
    # we can exploit symmetry or use a smarter counting method.
    # However, 16P8 is about 5e8, which is too slow for Python if done naively.
    
    # Alternative: Randomized Greedy / Local Search
    # Start with K16.
    # While there exists a Q3 subgraph:
    #   Find one Q3 subgraph (randomly or systematically).
    #   Remove one edge from it.
    # This is simple but might be slow to converge or get stuck in local minima?
    # Actually, removing edges from K16 to make it Q3-free is equivalent to finding a large Q3-free subgraph.
    
    # Better approach: 
    # 1. Start with K16.
    # 2. Identify a Q3 subgraph.
    # 3. Remove an edge from that subgraph.
    # 4. Repeat.
    # The number of edges to remove is 120 - ex(16; Q3).
    # If ex(16; Q3) is around 40-50, we need to remove 70-80 edges.
    # Finding a Q3 in a dense graph is fast.
    
    # Even better: "Most Constrained Edge" heuristic
    # But counting embeddings is expensive.
    
    # Let's try the simple "Find Q3, remove random edge from it" approach first, 
    # but with a twist: we can track which edges are in *many* Q3s by sampling?
    
    # Actually, let's look at the structure.
    # A Q3 has 8 vertices and 12 edges.
    # If we pick a random set of 8 vertices, the probability they form a Q3 is low, 
    # but in K16, ANY 8 vertices induce a K8, which contains many Q3s.
    
    # Strategy: 
    # Maintain the current graph as a set of edges.
    # While True:
    #   Try to find a Q3 subgraph.
    #   If found, remove an edge from it.
    #   If not found, return the graph.
    
    # How to find a Q3 subgraph efficiently?
    # Pick 8 random vertices. Check if they contain a Q3.
    # A set of 8 vertices contains a Q3 if there is a bijection to 0..7 preserving adjacency.
    # In a complete graph on 8 vertices, there are many Q3s.
    # In a sparse graph, we need to check.
    
    # Let's implement a function to check if a specific set of 8 vertices contains a Q3.
    # Since the graph is dense initially, we can just check if the induced subgraph is connected enough?
    # No, we need exact isomorphism.
    
    # Efficient Check for Q3 in 8 vertices:
    # There are 8! = 40320 permutations of 8 vertices.
    # For each permutation, check if it maps to a Q3.
    # This is feasible for a single check.
    
    # Algorithm:
    # 1. G = K16
    # 2. While True:
    #    a. If time limit near, break.
    #    b. Pick a random subset S of 8 vertices from G.
    #    c. Check if G[S] contains a Q3.
    #       - If yes, pick a random edge e from that Q3.
    #       - Remove e from G.
    #    d. If no Q3 found in many random tries, assume Q3-free? 
    #       - Dangerous. We need certainty.
    
    # To ensure certainty, we must systematically search or have a robust random search.
    # Given 120s, we can afford to be thorough.
    
    # Let's refine:
    # The graph will become sparse. The number of Q3s will drop rapidly.
    # Once the graph is sparse, finding a Q3 is harder.
    # We can use a backtracking search to find a Q3, or just rely on random sampling if the graph is dense.
    # But as it gets sparse, random sampling might miss the last few Q3s.
    
    # Hybrid:
    # 1. Start K16.
    # 2. While True:
    #    - Check if graph is Q3-free using a deterministic method? 
    #      Checking if a 16-vertex graph is Q3-free is NP-hard? 
    #      Actually, checking for a specific 8-vertex subgraph is fixed-parameter tractable.
    #      We can enumerate all 8-subsets (C(16,8) = 12870) and check each.
    #      Checking one 8-subset for Q3:
    #        - Build adjacency matrix of induced subgraph.
    #        - Check if isomorphic to Q3.
    #        - Isomorphism check for 8 vertices: 
    #          - Try all 8! permutations? 40k checks.
    #          - 12870 * 40k = 500M operations. Too slow in Python if done every iteration.
    
    # Optimization:
    # We only need to check if a Q3 *exists*.
    # We can maintain a list of "candidate" Q3s?
    
    # Let's go with a simpler, faster heuristic that is likely to find a good solution:
    # 1. Start with a known good construction (e.g., K3,13 + triangle = 42 edges).
    # 2. Try to ADD edges back to reach K16, checking for Q3 violations.
    #    - This is "Growing" approach.
    #    - Start with E0 = {edges of K3,13 + triangle}.
    #    - For each non-edge e in K16 \ E0:
    #       - Temporarily add e.
    #       - Check if the new graph contains a Q3.
    #       - If yes, remove e (or keep it if we use local search?).
    #       - If no, keep e.
    #    - This greedy addition is O(N_edges * Cost_Check).
    #    - Cost_Check: Does adding one edge create a Q3?
    #      A new Q3 must use the new edge e.
    #      So we only need to check if there is a Q3 containing e.
    #      This reduces the search space significantly.
    #      To check if a Q3 contains edge (u,v):
    #        - The other 6 vertices of the Q3 must be chosen from the remaining 14.
    #        - In Q3, u and v are adjacent. 
    #        - The structure of Q3 around an edge:
    #          Let u=0, v=1 (differ in bit 0).
    #          The neighbors of 0 in Q3 are 1, 2, 4.
    #          The neighbors of 1 in Q3 are 0, 3, 5.
    #          So in the embedding, the neighbors of u in G must include the images of 2,4.
    #          This constrains the choice of the other 6 vertices.
    
    # Let's implement the "Add Edge Back" strategy.
    # Base graph: K3,13 with triangle on {0,1,2}. (42 edges)
    # This is Q3-free.
    # We try to add the remaining 120-42=78 edges.
    
    # Pre-compute the base edges
    base_edges = set()
    # Triangle on 0,1,2
    base_edges.add((0,1))
    base_edges.add((0,2))
    base_edges.add((1,2))
    # K3,13 between {0,1,2} and {3..15}
    for a in range(3):
        for b in range(3, 16):
            base_edges.add((a,b))
    
    current_edges = list(base_edges)
    current_edge_set = set(base_edges)
    
    # Non-edges to try adding
    non_edges = [e for e in all_edges if e not in current_edge_set]
    random.shuffle(non_edges)
    
    def contains_q3_with_edge(edge_set, new_edge):
        """
        Check if adding new_edge to edge_set creates a Q3.
        The Q3 must contain new_edge.
        """
        u, v = new_edge
        # In Q3, let's fix u and v to be adjacent vertices, say 0 and 1.
        # Q3 vertices: 0..7.
        # Edges: (0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)
        # Wait, standard Q3 edges:
        # 000-001 (0-1)
        # 000-010 (0-2)
        # 000-100 (0-4)
        # 001-011 (1-3)
        # 001-101 (1-5)
        # 010-011 (2-3)
        # 010-110 (2-6)
        # 100-101 (4-5)
        # 100-110 (4-6)
        # 101-111 (5-7)
        # 110-111 (6-7)
        
        # If we map Q3 vertex 0 to u and Q3 vertex 1 to v:
        # Then Q3 vertex 2 must be a neighbor of u in G.
        # Q3 vertex 4 must be a neighbor of u in G.
        # Q3 vertex 3 must be a neighbor of v in G.
        # Q3 vertex 5 must be a neighbor of v in G.
        # Also:
        # 2 is neighbor of 3.
        # 4 is neighbor of 5.
        # 3 is neighbor of 7.
        # 5 is neighbor of 7.
        # 6 is neighbor of 2, 4, 7.
        
        # This is getting complex. Let's just enumerate the 6 other vertices.
        # The other 6 vertices in Q3 are 2,3,4,5,6,7.
        # We need to map these to 6 distinct vertices from the remaining 14.
        # There are P(14,6) = 2,162,160 possibilities.
        # For each, check if the edges exist in edge_set U {new_edge}.
        # This is ~2M checks per edge addition.
        # 78 edges * 2M = 156M checks. Might be too slow in Python (150s limit).
        
        # Optimization:
        # We can prune.
        # u must be connected to 2 and 4.
        # v must be connected to 3 and 5.
        # Let N_u = neighbors of u in G (excluding v).
        # Let N_v = neighbors of v in G (excluding u).
        # We pick 2 from N_u, 4 from N_u.
        # We pick 3 from N_v, 5 from N_v.
        # Then 6 and 7 are determined by intersections?
        # 6 is neighbor of 2, 4, 7.
        # 7 is neighbor of 3, 5, 6.
        
        # Let's try a simpler check: 
        # Just use the 8-vertex isomorphism check on the induced subgraph of 
        # {u, v} U {6 randomly selected neighbors?}
        # No, the vertices are specific.
        
        # Let's stick to the random search on the full graph, but smarter.
        
        # Alternative: Use the fact that Q3 is bipartite.
        # Q3 has bipartition {0,3,5,6} and {1,2,4,7} (depending on labeling).
        # Actually, 0(000) is in part A. 1(001) is in part B.
        # Part A: 0, 3, 5, 6 (even parity)
        # Part B: 1, 2, 4, 7 (odd parity)
        
        # If we add edge (u,v), it must connect Part A to Part B in the embedding.
        # So u and v must be in different parts of the Q3 embedding.
        
        # This doesn't simplify the search much.
        
        # Let's go back to the "Remove Edge" strategy from K16.
        # It is robust.
        # To speed it up, we can use a "conflict count" heuristic.
        # But we don't have the counts.
        
        # Let's try a hybrid:
        # 1. Start with K16.
        # 2. Use a fast heuristic to find a Q3.
        #    - Pick random 8 vertices.
        #    - Check if they form a Q3 (using the 40k permutation check).
        #    - If yes, remove a random edge from that Q3.
        #    - If no, try again.
        # 3. To handle the "sparse" end where random sampling fails:
        #    - If we fail to find a Q3 in K random tries, do a systematic check?
        #    - Or just accept the risk? No, must be valid.
        
        # Systematic check for Q3-free:
        # Iterate all C(16,8) subsets.
        # For each, check isomorphism to Q3.
        # This is the bottleneck.
        
        # Can we cache the isomorphism results?
        # The graph changes, so no.
        
        # How about this:
        # We know K3,13 + triangle is 42 edges and Q3-free.
        # We can start there and try to ADD edges.
        # Adding an edge e creates a Q3 only if there is a Q3 containing e.
        # We can check this by looking for a Q3 in the graph G' = G U {e}.
        # Since G is Q3-free, any Q3 in G' must use e.
        # So we just need to check if there is a Q3 containing e.
        # As analyzed, this requires checking ~2M mappings.
        # But we can prune heavily.
        
        # Let's implement the "Add Edge" strategy with pruning.
        
        def check_q3_containing_edge(G, u, v):
            """
            Check if there is a Q3 in G containing edge (u,v).
            G is a set of edges.
            """
            # Neighbors of u and v in G
            # Note: u and v are connected in G.
            N_u = set()
            N_v = set()
            for a, b in G:
                if a == u:
                    N_u.add(b)
                elif b == u:
                    N_u.add(a)
                if a == v:
                    N_v.add(b)
                elif b == v:
                    N_v.add(a)
            
            # Remove each other from neighbor sets
            N_u.discard(v)
            N_v.discard(u)
            
            # In Q3, u (0) connects to 1, 2, 4.
            # v (1) connects to 0, 3, 5.
            # So in the embedding:
            # The neighbors of u in the Q3 (excluding v) are 2 and 4.
            # The neighbors of v in the Q3 (excluding u) are 3 and 5.
            
            # We need to choose two distinct vertices x, y from N_u to map to 2, 4.
            # We need to choose two distinct vertices z, w from N_v to map to 3, 5.
            # Then we need to find 6 and 7.
            # 6 is connected to 2, 4, 7.
            # 7 is connected to 3, 5, 6.
            
            # So 6 must be in N(x) U N(y) (neighbors of 2 and 4 in G)
            # And 7 must be in N(z) U N(w) (neighbors of 3 and 5 in G)
            # And 6 must be connected to 7.
            # And 6 must be connected to x and y.
            # And 7 must be connected to z and w.
            
            # Let's iterate over choices of x, y from N_u and z, w from N_v.
            # |N_u| and |N_v| are small in the final graph.
            
            list_N_u = list(N_u)
            list_N_v = list(N_v)
            
            for x, y in itertools.combinations(list_N_u, 2):
                for z, w in itertools.combinations(list_N_v, 2):
                    # Check if x, y, z, w are distinct (they are, since N_u and N_v are disjoint from u,v and each other? 
                    # Not necessarily disjoint from each other. x could be z? No, x is neighbor of u, z is neighbor of v. 
                    # In Q3, 2 is not neighbor of 1 (v). So x != z.
                    # Also x != w, y != z, y != w.
                    if x in (z, w) or y in (z, w):
                        continue
                    
                    # Find candidates for 6 and 7
                    # 6 must be neighbor of x and y.
                    # 7 must be neighbor of z and w.
                    # 6 and 7 must be connected.
                    
                    # Get neighbors of x, y, z, w
                    Nx = set()
                    Ny = set()
                    Nz = set()
                    Nw = set()
                    for a, b in G:
                        if a == x: Nx.add(b)
                        elif b == x: Nx.add(a)
                        if a == y: Ny.add(b)
                        elif b == y: Ny.add(a)
                        if a == z: Nz.add(b)
                        elif b == z: Nz.add(a)
                        if a == w: Nw.add(b)
                        elif b == w: Nw.add(a)
                    
                    # Candidates for 6: N(x) ∩ N(y)
                    cand_6 = Nx & Ny
                    # Candidates for 7: N(z) ∩ N(w)
                    cand_7 = Nz & Nw
                    
                    # Exclude u, v, x, y, z, w from candidates
                    exclude = {u, v, x, y, z, w}
                    cand_6 -= exclude
                    cand_7 -= exclude
                    
                    for six in cand_6:
                        for seven in cand_7:
                            if six == seven:
                                continue
                            # Check if 6-7 edge exists
                            if (six, seven) in G or (seven, six) in G:
                                # Check if 6 is connected to x, y (already in cand_6)
                                # Check if 7 is connected to z, w (already in cand_7)
                                # Check if 6 is connected to 7 (done)
                                # Are there any other constraints?
                                # Q3 edges:
                                # 0-1 (u-v) - present
                                # 0-2 (u-x) - present (x in N_u)
                                # 0-4 (u-y) - present (y in N_u)
                                # 1-3 (v-z) - present (z in N_v)
                                # 1-5 (v-w) - present (w in N_v)
                                # 2-3 (x-z) - MUST CHECK
                                # 2-6 (x-six) - present (six in Nx)
                                # 4-5 (y-w) - MUST CHECK
                                # 4-6 (y-six) - present (six in Ny)
                                # 5-7 (w-seven) - present (seven in Nw)
                                # 6-7 (six-seven) - present
                                # 3-7 (z-seven) - present (seven in Nz)
                                # Wait, I missed some.
                                # Let's list all 12 edges of Q3 and check them.
                                # Mapping: 0->u, 1->v, 2->x, 3->z, 4->y, 5->w, 6->six, 7->seven
                                # Edges:
                                # (0,1) -> (u,v) OK
                                # (0,2) -> (u,x) OK
                                # (0,4) -> (u,y) OK
                                # (1,3) -> (v,z) OK
                                # (1,5) -> (v,w) OK
                                # (2,3) -> (x,z) Check
                                # (2,6) -> (x,six) OK
                                # (3,7) -> (z,seven) OK
                                # (4,5) -> (y,w) Check
                                # (4,6) -> (y,six) OK
                                # (5,7) -> (w,seven) OK
                                # (6,7) -> (six,seven) OK
                                
                                if (x, z) in G or (z, x) in G:
                                    if (y, w) in G or (w, y) in G:
                                        return True
            return False
        
        # Add edges greedily
        for e in non_edges:
            if time.time() > deadline - 5:
                break
            u, v = e
            # Check if adding e creates a Q3
            if not check_q3_containing_edge(current_edge_set, u, v):
                current_edge_set.add(e)
                current_edges.append(e)
        
        return [[u, v] for u, v in current_edges]

# EVOLVE-BLOCK-END
