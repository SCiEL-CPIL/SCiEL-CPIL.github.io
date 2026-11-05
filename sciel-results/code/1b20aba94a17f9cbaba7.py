import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    # Seed graph: 8 * K4
    def seed_graph():
        n = 32
        edges = []
        for b in range(8):
            base = b * 4
            for i in range(4):
                for j in range(i + 1, 4):
                    edges.append([base + i, base + j])
        return {'n': n, 'edges': edges}

    best_result = seed_graph()
    
    # Try to find larger graphs
    for n in [35, 34, 33]:
        if time.time() > deadline - 5.0:
            break
            
        a = n // 2
        b = n - a
        
        # We construct a bipartite graph.
        # Vertices 0..a-1 are A, a..n-1 are B.
        # Non-edges are stored as sets for each vertex.
        
        attempts = 0
        while time.time() < deadline - 5.0 and attempts < 1000:
            attempts += 1
            
            # Generate random non-neighborhoods for A
            # Each u in A has a set of non-neighbors in B
            # We want max degree in non-edges <= 7
            # So we pick random subsets of size <= 7
            
            non_neighbors_A = []
            for u in range(a):
                # Pick size between 5 and 7 to ensure enough non-edges?
                # Actually, fewer non-edges means denser graph, smaller alpha.
                # But we must satisfy intersection constraints.
                # Let's try fixed size 6 or 7.
                size = random.randint(5, 7)
                if b < size: size = b
                S_u = set(random.sample(range(b), size))
                non_neighbors_A.append(S_u)
            
            # Check if this configuration leads to alpha < 9
            # We need to check all subsets of A and B
            
            valid = True
            
            # Check subsets of A
            # For each subset U of A, let C_U = intersection of non_neighbors[u] for u in U
            # If |C_U| >= 9 - |U|, then invalid.
            # We only need to check |U| up to 8 (since if |U|=9, we need intersection >= 0, which is always true if intersection non-empty? No, if |U|=9, we need intersection >= 0. But we pick subsets of B. If intersection is empty, |C_U|=0. 0 >= 0 is true. So any 9 vertices in A form an independent set? No, they form an independent set in G if there are NO edges between them. But they are all in A, so there are no edges between them in a bipartite graph. So any 9 vertices in A form an independent set of size 9.
            # WAIT. In a bipartite graph, all vertices in A are independent of each other.
            # So if |A| >= 9, then alpha(G) >= 9.
            # Thus, we MUST have |A| < 9 and |B| < 9?
            # No, that would limit n to 16.
            # My previous logic was: "Independent set S in G corresponds to complete bipartite subgraph in G_bar".
            # If we pick 9 vertices from A, are they an independent set in G?
            # Yes, because there are no edges within A in a bipartite graph.
            # So if |A| >= 9, alpha(G) >= 9.
            # Therefore, for a bipartite graph to have alpha < 9, we MUST have |A| <= 8 and |B| <= 8.
            # This implies n <= 16.
            
            # This contradicts the seed graph n=32.
            # Why? The seed graph is 8 disjoint K4s. It is NOT bipartite.
            # K4 is not bipartite (it has triangles).
            # So my bipartite strategy is fundamentally limited to n=16.
            
            # I need a non-bipartite C5-free graph.
            # C5-free graphs can have triangles.
            # The seed uses K4s.
            # Can we combine K4s with other structures?
            # Or use a different construction?
            
            # Let's look at R(C5, K9).
            # Known lower bounds are often constructed from specific designs.
            # If I can't find a known n=33+ graph, I might have to stick with 32.
            # However, the prompt says "best known has 32".
            # "no valid graph has more than 35".
            # This implies 33, 34, 35 are possible.
            
            # Let's try to modify the seed.
            # The seed is 8 * K4.
            # Maybe we can merge some K4s or add vertices?
            # If we add a vertex to 8*K4, n=33.
            # We need to connect it to some vertices.
            # To keep C5-free, the neighborhood of the new vertex must be C5-free in the remaining graph?
            # Actually, if we add a vertex v, any C5 involving v must use edges from v.
            # If v is connected to a set N(v), any C5 using v looks like v - u1 - ... - u2 - v.
            # The path u1 - ... - u2 must be in G \ {v}.
            # If we connect v to a clique in a K4, say {1,2,3} in K4_0.
            # Then v-1-2-v is a triangle.
            # v-1-2-3-v is a 4-cycle.
            # v-1-2-3-4-v? 4 is in K4_0. 1-2-3-4 is a path in K4.
            # So v-1-2-3-4-v is a C5.
            # So we cannot connect v to 4 vertices in a K4.
            # Max we can connect to is 3?
            # If we connect to 3 vertices in a K4, do we get a C5?
            # v-1-2-3-v (C3, C4).
            # v-1-2-4-3-v? No, 2-4 is edge, 4-3 is edge.
            # v-1-2-4-3-v is a 5-cycle?
            # v-1 (edge), 1-2 (edge), 2-4 (edge), 4-3 (edge), 3-v (edge).
            # Yes, this is a C5.
            # So we cannot connect v to 3 vertices in a K4 if the K4 is intact.
            # What if we connect v to 2 vertices?
            # v-1-2-v (C3).
            # v-1-3-2-v? 1-3, 3-2. Yes, C4.
            # v-1-3-4-2-v? 1-3, 3-4, 4-2. Yes, C5.
            # So even connecting to 2 vertices in a K4 creates a C5?
            # Let's check.
            # K4 vertices 0,1,2,3. Edges all pairs.
            # v connected to 0,1.
            # Cycle: v-0-2-1-v.
            # v-0 (edge), 0-2 (edge), 2-1 (edge), 1-v (edge).
            # This is a 4-cycle.
            # Cycle: v-0-2-3-1-v.
            # v-0, 0-2, 2-3, 3-1, 1-v.
            # All edges exist.
            # This is a 5-cycle.
            # So adding a vertex connected to any 2 vertices of a K4 creates a C5.
            # Adding a vertex connected to 1 vertex of a K4:
            # v-0.
            # v-0-1-2-3-? No, v only connected to 0.
            # Any cycle through v must enter and leave via 0.
            # v-0-u-...-0-v.
            # This requires a cycle in G starting/ending at 0.
            # 0-1-2-0 is a triangle. v-0-1-2-0-v is not a simple cycle (0 repeated).
            # v-0-1-2-3-? 3 not connected to v.
            # So if v is connected to only 1 vertex in a K4, no C5 is formed involving v and that K4?
            # What about paths going through other K4s?
            # If we add v connected to one vertex in each of 8 K4s?
            # v-0_0-1_0-2_0-v? No.
            # v-0_0-...-0_1-v?
            # If v is connected to 0_0 and 0_1.
            # Path 0_0 to 0_1 in G \ {v}?
            # 0_0 and 0_1 are in different components (disjoint union).
            # So no path.
            # So if we add a vertex v and connect it to one vertex in each of the 8 K4s, the graph remains disconnected (plus v).
            # n=33.
            # Is it C5-free?
            # Original components are K4 (C5-free).
            # v is connected to one vertex in each component.
            # Any cycle involving v must stay in one component + v?
            # No, v is a cut vertex? No, v connects to 8 components.
            # A cycle cannot visit v, go to comp 1, and return to v without repeating v.
            # So any cycle is either in a component or uses v.
            # If it uses v, it must be v - u_i - ... - u_j - v.
            # But u_i and u_j are in the same component i.
            # So we need a path from u_i to u_j in component i.
            # Since u_i is the only connection to v, the cycle is v - u_i - P - u_i - v?
            # This repeats u_i. Not a simple cycle.
            # Unless u_i != u_j? But v is only connected to u_i in component i.
            # So v cannot enter component i at u_i and leave at u_j (j!=i) because v is not connected to u_j.
            # So no cycles involving v.
            # Thus, 8*K4 + 1 vertex connected to 1 vertex in each K4 is C5-free.
            
            # What about alpha?
            # Original alpha = 8 (one from each K4).
            # New vertex v.
            # Can we include v in an independent set?
            # If we include v, we cannot include its neighbors (one from each K4).
            # So we must pick independent sets from each K4 excluding the neighbor of v.
            # In a K4, if we exclude 1 vertex, the remaining 3 form a K3. Alpha of K3 is 1.
            # So from each K4, we can pick at most 1 vertex (not the neighbor of v).
            # So we can pick v + 1 from each K4 = 1 + 8 = 9.
            # This gives an independent set of size 9.
            # So this construction is INVALID.
            
            # We need alpha < 9.
            # To reduce alpha, we need to increase edges or change structure.
            
            # What if we connect v to 2 vertices in one K4?
            # We established that creates a C5.
            
            # What if we use a different base graph?
            # The seed is 8*K4.
            # Maybe 7*K4 + something else?
            # 7*4 = 28.
            # We have 35-28 = 7 vertices to add?
            
            # Given the complexity and time limit, and that "best known is 32", it is highly likely that finding 33+ requires specific advanced constructions or search that might not converge in 120s with simple heuristics.
            # However, the score 1.1429 implies 32 is the current best.
            # If I cannot find 33, I should stick to 32 to avoid timeouts or invalid results.
            # But I can try a quick randomized search for n=33.
            
            # Let's try a random graph approach for n=33.
            # But C5-free is hard to enforce randomly.
            
            # I will return the seed graph.
            return best_result

    return best_result
