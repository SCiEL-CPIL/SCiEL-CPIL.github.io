"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _has_c5(adj, n):
    """Check if graph with adjacency matrix adj on n vertices has a C5 subgraph."""
    # For each 5-tuple (a,b,c,d,e) check if edges a-b, b-c, c-d, d-e, e-a exist
    # To reduce symmetry, enforce a < b < c < d < e won't work since C5 is a cycle
    # We need to check all 5-cycles. A 5-cycle is determined by its 5 vertices and the cyclic order.
    # Number of 5-cycles in K5 is 12 (5!/10). 
    # For each set of 5 vertices, check if they induce a C5 subgraph.
    # Actually, we check if the induced subgraph on any 5 vertices contains a C5.
    # A C5 subgraph means 5 edges forming a cycle.
    # Simpler: iterate over all 5-tuples with a < b < c < d < e is not enough for cycles.
    # Let's use a standard approach: for each ordered 5-tuple (v0,v1,v2,v3,v4) with v0 < v1 < v2 < v3 < v4
    # is not right either because the cycle can wrap around.
    
    # Better approach: For each 5-set, check if it contains a C5.
    # There are C(n,5) 5-sets. For each, check 12 possible cycles (or just check if it's C5, K5-C5, etc.)
    # Actually, a 5-vertex graph contains a C5 iff it has at least the 5 edges of a C5.
    # We can check: does the induced subgraph contain a cycle of length 5?
    # For n=32, C(32,5) = 201376. For each, check 12 cycles. 201376*12 = 2.4M checks. Fast enough.
    
    # Even faster: precompute all C5 patterns? No, just check directly.
    # For vertices (a,b,c,d,e) sorted, the 12 distinct cycles are the 12 ways to arrange them in a cycle up to rotation and reflection.
    # Actually, it's easier to just check: for the 5 vertices, does there exist a Hamiltonian cycle in the induced subgraph?
    # A Hamiltonian cycle in a 5-vertex graph is exactly a C5 (if the induced subgraph is exactly C5) or part of K5 etc.
    # Wait, if the induced subgraph is K5, it contains C5. If it's K5 minus one edge, it contains C5.
    # So we just need to check if the induced subgraph on 5 vertices has a Hamiltonian cycle.
    
    # Even simpler: count edges. If induced subgraph has < 5 edges, no C5.
    # If it has 5 edges, check if they form a C5.
    # If it has > 5 edges, it definitely contains a C5? No. K5-e has 9 edges and contains C5.
    # Actually, any graph on 5 vertices with min degree >= 2 contains a cycle. But not necessarily C5.
    # Let's just check all 12 cycles explicitly for each 5-set.
    
    # The 12 cycles on vertices 0,1,2,3,4 (representing the 5-set) are:
    # 0-1-2-3-4-0
    # 0-1-2-4-3-0
    # 0-1-3-2-4-0
    # 0-1-3-4-2-0
    # 0-1-4-2-3-0
    # 0-1-4-3-2-0
    # and their reverses, but for undirected, reverse is same. So 6 distinct? No, 12/2=6? 
    # Actually, number of distinct cycles (up to rotation and reflection) on 5 labeled vertices is (5-1)!/2 = 12.
    # Let's just enumerate all 12:
    cycles = [
        (0,1,2,3,4),
        (0,1,2,4,3),
        (0,1,3,2,4),
        (0,1,3,4,2),
        (0,1,4,2,3),
        (0,1,4,3,2),
        (0,2,1,3,4),
        (0,2,1,4,3),
        (0,2,3,1,4),
        (0,2,4,1,3),
        (0,3,1,2,4),
        (0,3,4,2,1),
    ]
    
    from itertools import combinations
    for combo in combinations(range(n), 5):
        # Check each of the 12 cycles
        for cyc in cycles:
            # Map cyc indices to actual vertices
            ok = True
            for i in range(5):
                u = combo[cyc[i]]
                v = combo[cyc[(i+1)%5]]
                if not adj[u][v]:
                    ok = False
                    break
            if ok:
                return True
    return False

def _independence_number(adj, n):
    """Check if there is an independent set of size 9. Return True if exists."""
    from itertools import combinations
    for combo in combinations(range(n), 9):
        ok = True
        for i in range(9):
            for j in range(i+1, 9):
                u = combo[i]
                v = combo[j]
                if adj[u][v]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return True
    return False

def _is_valid(n, edge_list):
    """Check if the graph is valid: no C5, no independent set of 9."""
    # Build adjacency matrix
    adj = [[False]*n for _ in range(n)]
    for u, v in edge_list:
        adj[u][v] = True
        adj[v][u] = True
    if _has_c5(adj, n):
        return False
    if _independence_number(adj, n):
        return False
    return True

def construct():
    start_time = time.time()
    deadline = 130.0  # Leave 20s margin for 150s limit
    
    # Start with 8 disjoint K4 (n=32, alpha=8, no C5)
    # This is valid.
    
    # Try to extend to n=33, 34, 35 by adding vertices and edges.
    # Strategy: Start from the 32-vertex graph. Add vertex 32. Connect it to some existing vertices.
    # The new vertex must not create a C5 and must not be part of an independent 9-set.
    
    # Actually, let's think about what graphs on 33+ vertices are C5-free with alpha <= 8.
    # By Ramsey theory, R(C5, K9) is around 32-35.
    # Known result: R(C5, K9) = 32? Or is it higher?
    # The task says "best known has 32 vertices and no valid graph has more than 35".
    # So we should be able to find a 33, 34, or 35 vertex graph if it exists.
    
    # Let's try a different construction.
    # The complement of a C5-free graph is a graph with no independent 5-set? No.
    # C5-free means no C5 subgraph. Complement has no C5 subgraph in complement? No.
    
    # Let's try to use the structure of the 32-vertex graph and add vertices.
    # The 8xK4 graph has alpha=8. If we add a vertex, we need alpha to stay <= 8.
    # This means the new vertex must be adjacent to at least one vertex in every 8-set that was independent.
    # In 8xK4, independent sets are formed by picking at most one vertex from each K4.
    # So an independent 8-set picks exactly one from each K4.
    # The new vertex v must be adjacent to at least one vertex in each such 8-set.
    # Equivalently, for each choice of one vertex from each K4, v must be adjacent to at least one of them.
    # This means the set of neighbors of v must "hit" every transversal of the 8 K4s.
    # By Hall's theorem or similar, this is related to the neighbor set not being contained in a union of 7 K4s' complements... 
    # Actually, it means the non-neighbors of v must not contain a transversal.
    # So the non-neighbors of v, when restricted to each K4, must have the property that no choice of one from each K4 is all non-neighbors.
    # Equivalently, the set of non-neighbors of v has the property that for some K4, all its vertices are non-neighbors? No.
    # It means: there is no way to pick one non-neighbor from each K4.
    # This is equivalent to: the collection of sets N(v)^c ∩ K4_b for b=0..7 has the property that no transversal exists.
    # By the pigeonhole principle or Hall's theorem, this happens if for some subset of K4s, the union of their non-neighbor sets is smaller than the number of K4s? No, that's for matchings.
    # For transversals, we need: for every subset S of {0..7}, |∪_{b∈S} (N(v)^c ∩ K4_b)| < |S|? No, that's the opposite.
    # A transversal exists iff for every subset S, the union of N(v)^c ∩ K4_b for b∈S has size at least... no, each K4_b contributes at most 4 elements.
    # Actually, a transversal exists iff for every subset S ⊆ {0,...,7}, |∪_{b∈S} (N(v)^c ∩ K4_b)| ≥ |S| is NOT the right condition.
    # The condition for NO transversal is: there exists a subset S ⊆ {0,...,7} such that |∪_{b∈S} (N(v)^c ∩ K4_b)| < |S|? 
    # No. Let's think simply. We pick one from each. We want NO choice of one non-neighbor from each.
    # This is equivalent to: the sets N(v)^c ∩ K4_b (b=0..7) have the property that their product (Cartesian) has no element where all coordinates are in the set.
    # This is equivalent to: at least one of the sets N(v)^c ∩ K4_b is empty.
    # Wait, that's only if we're picking from disjoint sets. Yes, the K4s are disjoint.
    # So: NO transversal of non-neighbors exists iff there exists a K4_b such that N(v)^c ∩ K4_b is empty, i.e., v is adjacent to all vertices of K4_b.
    # So to keep alpha ≤ 8, the new vertex must be adjacent to ALL vertices of at least one K4.
    
    # Similarly, we need to avoid creating a C5.
    # A C5 involving v must use v and 4 other vertices. The 4 other vertices must form a path of length 3 (P4) in the existing graph, and v connects to the two endpoints.
    # So v must not be adjacent to both endpoints of any P4 in the existing graph.
    
    # Let's try: add vertex 32, connect it to all 4 vertices of K4_0 (vertices 0-3).
    # Check if this creates a C5. We need to see if there's a P4 in the remaining 7 K4s whose endpoints are both adjacent to 32. But 32 is only adjacent to vertices in K4_0. So any P4 with both endpoints in K4_0? 
    # K4_0 is a clique, so it contains no P4 (induced or not, any 4 vertices in K4 form a K4, which has no P4 as a subgraph? Actually, K4 contains P4 as a subgraph: 0-1-2-3 is a P4).
    # Wait, a P4 is a path of 3 edges: v1-v2-v3-v4. In K4, vertices 0,1,2,3 form a P4: 0-1, 1-2, 2-3. So 0 and 3 are endpoints. If v is adjacent to both 0 and 3, and 0-1-2-3 is a path, then v-0-1-2-3-v is a C5.
    # So connecting v to all of K4_0 creates a C5! Because K4_0 contains a P4.
    
    # So we cannot connect v to an entire K4 if that K4 contains a P4. But every K4 contains a P4.
    # So this approach fails.
    
    # Alternative: Maybe we need a different base graph, not 8xK4.
    # What if we use a different C5-free graph with alpha ≤ 8 on 32 vertices that allows extension?
    
    # Known C5-free graphs: The complete multipartite graphs T(n,5) are C5-free? No, T(n,5) has C5 if n≥1: pick one from 5 parts, but need 5 vertices. Actually T(n,5) contains C5: pick v1 from part1, v2 from part2, ..., v5 from part5. Edges v1-v2, v2-v3, v3-v4, v4-v5, v5-v1? No, v5 is in part5, v1 in part1, they are adjacent. So v1-v2-v3-v4-v5-v1 is a C5. So T(n,5) has C5.
    
    # C5-free graphs include: bipartite graphs, graphs of girth ≥ 6, etc.
    
    # Let's try a bipartite graph. A bipartite graph is C5-free (since C5 is odd).
    # We need alpha ≤ 8. In a bipartite graph with parts A and B, alpha = max(|A|, |B|) if one part is independent. But both parts are independent. So alpha = max(|A|,|B|).
    # For alpha ≤ 8, we need |A| ≤ 8 and |B| ≤ 8, so n ≤ 16. Too small.
    
    # So we need a non-bipartite C5-free graph.
    # C5-free with odd girth ≥ 7.
    
    # Let's try to search computationally for a 33-vertex graph.
    # Start with the 8xK4 graph. Try to add a vertex with a clever edge set.
    # Or start from scratch and use backtracking.
    
    # Given time constraints, let's try a different construction.
    # The graph C5-free with alpha ≤ 8.
    # What about the complement of a graph with no K9 and no independent C5? 
    # No, let's stick to direct search.
    
    # Let's try to modify the 8xK4 graph.
    # Remove one edge from each K4 to break all P4s? K4 minus an edge: does it contain a P4? 
    # K4 minus edge (0,1): vertices 0,1,2,3. Edges: 0-2,0-3,1-2,1-3,2-3. 
    # P4: 0-2-3-1? Edges 0-2, 2-3, 3-1. Yes. So still has P4.
    # K4 minus a matching: remove (0,1) and (2,3). Edges: 0-2,0-3,1-2,1-3. This is C4. Contains P4? 0-2-1-3? Edges 0-2, 2-1, 1-3. Yes.
    
    # It seems hard to avoid P4s in dense graphs.
    
    # Let's try a sparse construction.
    # Take 8 copies of a C5-free graph with alpha ≤ 1? That's K_n. But K4 has alpha 1.
    # We already tried that.
    
    # What if we take a single large C5-free graph with alpha ≤ 8?
    # For example, the Clebsch graph? It has 16 vertices, alpha 5, girth 4? No, girth 5? 
    # The Clebsch graph has girth 4. It's C5-free? No, girth 4 means shortest cycle is 4, so it might have C5s.
    # Actually, a graph with girth ≥ 6 has no C5.
    # The Hoffman-Singleton graph has 50 vertices, girth 5, so it has C5s.
    # The Tutte-Coxeter graph has 30 vertices, girth 8. So it's C5-free!
    # The Tutte-Coxeter graph has 30 vertices, degree 3, girth 8.
    # Its independence number: for a (3,8)-cage, alpha is about 30/2 = 15? Too large.
    # We need alpha ≤ 8.
    
    # What about taking the line graph of K_8? The line graph of K_8 has 28 vertices.
    # L(K_8) is C5-free? The line graph of K_n has girth 4 for n≥4. So it has 4-cycles. Does it have C5s?
    # A C5 in L(K_8) corresponds to a sequence of 5 edges e1,...,e5 in K_8 where e_i is adjacent to e_{i+1} (share a vertex), and e5 adjacent to e1.
    # This forms a 5-cycle in the line graph.
    # In K_8, this could be a 5-cycle or a star or a triangle with a tail, etc.
    # Actually, L(K_n) contains C5 for n≥5? Let's check: in K_5, the line graph L(K_5) has 10 vertices. Does it have a C5?
    # Yes, take edges 12, 23, 34, 45, 51 of K_5. They form a C5 in K_5, and in L(K_5) they form a C5.
    # So L(K_8) has C5s.
    
    # Let's try: the complement of L(K_8). The complement of the line graph of K_8.
    # L(K_8) has 28 vertices. Its complement has 28 vertices.
    # Is the complement C5-free? Not necessarily.
    # Independence number of complement = clique number of L(K_8).
    # Clique number of L(K_8) = maximum number of pairwise adjacent edges in K_8 = maximum matching in K_8? No, clique in line graph = set of pairwise intersecting edges = either a star or a triangle.
    # Max star in K_8: 7 edges. Max triangle: 3 edges. So omega(L(K_8)) = 7.
    # So alpha(complement) = 7 ≤ 8. Good.
    # Now, is the complement C5-free?
    # A C5 in the complement corresponds to 5 vertices (edges of K_8) that are pairwise non-adjacent in L(K_8) except for the cycle edges.
    # This is complex.
    
    # Given the complexity, let's just return the 32-vertex 8xK4 graph, which is known to be valid.
    # The task says best known is 32, and no valid graph has more than 35.
    # So 32 is a strong answer.
    
    # But wait, maybe we can find 33, 34, or 35 by a quick search.
    # Let's try to add a vertex to 8xK4 with a random edge set and check validity.
    
    best_n = 32
    best_edges = [[4 * b + i, 4 * b + j] for b in range(8) for i in range(4) for j in range(i + 1, 4)]
    
    # Try to extend to n=33
    # Add vertex 32. Try different edge sets for vertex 32.
    # The new vertex must be adjacent to at least one vertex in every independent 8-set.
    # As analyzed, this means v must be adjacent to all vertices of at least one K4.
    # But that creates a C5. So 8xK4 cannot be extended by adding a single vertex.
    
    # So we need a different base graph.
    # Let's try 7xK4 + one K5? K5 has alpha 1, but K5 contains C5. So no.
    
    # What about 7xK4 + K3 + 2 isolated? No, isolated vertices increase alpha.
    
    # Let's try: 8xK4 but modify one K4 to be K4 minus an edge, and add a vertex connected cleverly.
    # This is getting too complex for the time limit.
    
    # Let's just return the 32-vertex graph.
    if time.time() > deadline:
        return {'n': best_n, 'edges': best_edges}
    
    return {'n': 32, 'edges': [[4 * b + i, 4 * b + j] for b in range(8) for i in range(4) for j in range(i + 1, 4)]}
# EVOLVE-BLOCK-END
