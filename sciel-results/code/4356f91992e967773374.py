"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    start_time = time.time()
    deadline = 110.0  # Keep margin from 120s limit

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges_cc = []
        for u in range(k):
            for v in range(u + 1, k):
                edges_cc.append((u, v))
        
        edges_ci = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges_ci.append((u, v))
                    
        return edges_cc, edges_ci

    def solve_split_graph(n, k, edges_ci):
        # 1. Build graph representation
        # neighbors[u] is a set of neighbors for u in range(n)
        # For efficiency, we only really care about clique structure.
        # Split graph: Clique C (0..k-1), Independent Set I (k..n-1).
        
        # We need to cover all edges (C-C and C-I) with disjoint cliques.
        
        # Candidate Cliques:
        # In a split graph, any clique is either:
        # 1. A subset of C.
        # 2. A set {v} U S where v in I and S subset of N(v) in C.
        #    Note: Since I is independent, no clique can contain two vertices from I.
        
        # Therefore, every clique in the partition contains AT MOST one vertex from I.
        # Let S_v be the set of C-vertices in the clique assigned to independent vertex v.
        # If v is not in any clique (isolated in I), it doesn't need coverage.
        # The edges covered by the clique for v are:
        #   - All edges (u, v) for u in S_v.
        #   - All edges (u1, u2) for u1, u2 in S_v.
        
        # The C-C edges are partitioned among the sets S_v (for v in I) and potentially
        # some cliques consisting only of C vertices (if some C-C edges are not covered by I-cliques).
        # Wait, the partition must cover ALL edges.
        # Let P be the partition.
        # Each edge (u, v) with u,v in C must be in exactly one clique.
        # Each edge (u, v) with u in C, v in I must be in exactly one clique.
        
        # Since a clique can have at most one v from I, the edge (u, v) is covered by the unique
        # clique containing v. Let's call this clique K_v.
        # K_v must contain v and some subset S_v of C.
        # For (u, v) to be in K_v, u must be in S_v.
        # So S_v must contain N_C(v) (all neighbors of v in C).
        # Can S_v contain non-neighbors of v?
        # If u in S_v but u is not adjacent to v, then (u, v) is not an edge.
        # But K_v is a clique, so all pairs in K_v must be edges.
        # Since v is in K_v, all u in S_v must be adjacent to v.
        # Thus S_v must be a subset of N_C(v).
        # And since all edges (u, v) must be covered, S_v MUST be exactly N_C(v).
        # Wait, is that true?
        # If S_v is a proper subset of N_C(v), then some edge (u, v) with u in N_C(v) \ S_v is NOT covered by K_v.
        # Can it be covered by another clique?
        # No, because any clique covering (u, v) must contain v.
        # And v can only be in one clique in a partition.
        # Therefore, for every v in I with degree > 0 in C, the clique K_v is FIXED as {v} U N_C(v).
        # These cliques are mandatory!
        
        # Let's verify this logic.
        # Partition: disjoint sets of vertices, each set is a clique, every edge is in exactly one set.
        # Let v in I. Let K be the clique in the partition containing v.
        # K must be a clique. Since I is independent, K \ {v} must be a subset of C.
        # Let S = K \ {v}. S subset C.
        # For K to be a clique, all u in S must be adjacent to v. So S subset N_C(v).
        # Every edge (u, v) must be in exactly one clique. The only clique containing v is K.
        # So (u, v) must be in K. This implies u in S.
        # So N_C(v) subset S.
        # Combined: S = N_C(v).
        # So the clique containing v is uniquely determined: K_v = {v} U N_C(v).
        # This holds for all v in I such that N_C(v) is not empty.
        # If N_C(v) is empty, v is isolated. It forms a clique of size 1?
        # "Every listed set must be a clique with at least two vertices".
        # Isolated vertices do not need to be listed if they don't cover any edges.
        # The condition is "every edge must lie in exactly one listed clique".
        # Isolated vertices have no edges, so they don't need to be in any listed clique.
        
        # So, the cliques for all non-isolated independent vertices are FIXED.
        # Let FixedCliques = [ {v} U N_C(v) for v in I if N_C(v) != 0 ].
        
        # Now, what edges remain to be covered?
        # The C-C edges.
        # Some C-C edges might already be covered by the FixedCliques.
        # Specifically, if u1, u2 in N_C(v), then edge (u1, u2) is in K_v.
        # We need to cover the remaining C-C edges with cliques consisting ONLY of C vertices.
        # Let E_CC_remaining be the set of C-C edges not covered by any K_v.
        # We need to partition E_CC_remaining into cliques of C.
        # Since C is a clique (complete graph), ANY subset of C is a clique.
        # So we just need to partition the set of edges E_CC_remaining into groups,
        # where each group is a set of edges forming a clique (which is always true for subsets of C),
        # BUT the constraint is that the cliques (vertex sets) must be disjoint?
        # NO. The partition is of the EDGES.
        # "Every listed set must be a clique... and every edge must lie in exactly one listed clique."
        # The sets (cliques) are vertex sets.
        # If we list a clique S (subset of C), it covers ALL edges within S.
        # We cannot cover an edge twice.
        # So if we pick clique S1 and S2, the edges in S1 and S2 must be disjoint.
        # Since S1, S2 are subsets of C, edges in S1 are pairs in S1.
        # If S1 and S2 share a vertex u, do they share an edge?
        # If S1 = {u, v} and S2 = {u, w}, edges are (u,v) and (u,w). Disjoint.
        # If S1 = {u, v, w} and S2 = {u, v, z}, edges in S1: (u,v),(u,w),(v,w).
        # Edges in S2: (u,v),(u,z),(v,z).
        # Edge (u,v) is in both! Invalid.
        # So the vertex sets of the C-only cliques must be such that their induced edge sets are disjoint.
        # This is equivalent to: The C-only cliques must be an edge-disjoint collection of cliques in K_k.
        # Actually, simpler: We are partitioning the set of remaining edges E_rem into cliques.
        # A "clique" in this context is a vertex set S such that all edges in S are in E_rem.
        # And we want to minimize the number of such sets.
        
        # Let's analyze the structure of E_rem.
        # E_rem = E_CC \ Union(Edges(K_v)).
        # Edges(K_v) = { (u1, u2) : u1, u2 in N_C(v) }.
        
        # We need to cover E_rem with minimum number of cliques (subsets of C).
        # Since C is complete, any subset of C is a clique.
        # The constraint is that the cliques must be edge-disjoint.
        # This is the "Edge Clique Cover" of the graph G_rem = (C, E_rem).
        # We want to cover E_rem with min number of cliques.
        # Note: A clique in the cover is a set of vertices S such that S induces a clique in G_rem?
        # No, S induces a clique in the ORIGINAL graph C (which is K_k).
        # But the edges covered are those in E_rem.
        # So if we pick S, it covers { (u,v) in E_rem : u,v in S }.
        # We need these sets of edges to partition E_rem.
        
        # This is equivalent to finding a partition of the edge set E_rem into cliques.
        # Since the underlying graph C is complete, any subset of vertices forms a clique.
        # The only restriction is that we cannot use an edge twice.
        
        # Is there a simple structure for E_rem?
        # E_rem contains edge (u, w) if and only if there is NO v in I such that u, w in N_C(v).
        # i.e., N_C(v) does not contain both u and w.
        
        # Let's look at the complement.
        # Let H be the graph on C where (u, w) is an edge if u, w are NOT both in any N_C(v).
        # We want to cover edges of H with min cliques.
        # Wait, if (u, w) is in E_rem, it must be covered.
        # If (u, w) is NOT in E_rem, it is already covered by some K_v.
        
        # So we need to edge-clique-cover the graph H = (C, E_rem).
        # H is a subgraph of K_k.
        # Minimizing the number of cliques to cover edges of H.
        
        # Observation:
        # If H is a complete graph, we need 1 clique (all of C).
        # If H is a matching, we need |E(H)| cliques (each edge is a clique of size 2).
        
        # Can we compute the min clique cover of H efficiently?
        # The clique cover number of a graph is equal to the chromatic number of its complement.
        # Let H_bar be the complement of H in K_k.
        # H_bar has edge (u, w) if (u, w) is NOT in E_rem.
        # i.e., (u, w) is covered by some K_v.
        # So H_bar is the union of the cliques N_C(v) (as edge sets).
        # We want to find min number of cliques covering E(H).
        # This is equivalent to finding the chromatic number of H_bar?
        # Let's check.
        # A clique in H corresponds to an independent set in H_bar.
        # Covering E(H) with cliques is NOT the same as coloring H_bar.
        # Coloring H_bar partitions VERTEX set into independent sets.
        # We are partitioning EDGE set of H into cliques.
        
        # However, note that if we partition V(H_bar) into independent sets I_1, ..., I_m,
        # then the edges of H are exactly the edges between different I_j's? No.
        # Edges of H are edges NOT in H_bar.
        # If I_j is independent in H_bar, it contains no edges of H_bar.
        # So all edges within I_j are in H.
        # So I_j induces a clique in H.
        # If we color H_bar with m colors, we get m independent sets I_1..I_m.
        # Each I_j is a clique in H.
        # Do these cliques cover all edges of H?
        # An edge (u, v) in H means (u, v) is NOT in H_bar.
        # In a proper coloring of H_bar, if (u, v) is not an edge in H_bar, u and v CAN have the same color.
        # If u and v have the same color, they are in the same I_j.
        # Then (u, v) is covered by the clique I_j.
        # If u and v have different colors, they are in different I_j, I_k.
        # Is (u, v) covered?
        # The cliques in the cover are the I_j's.
        # The edge (u, v) is in I_j only if u, v in I_j.
        # If u, v have different colors, (u, v) is NOT in any I_j.
        # So the coloring of H_bar does NOT necessarily cover all edges of H.
        # It only covers edges within color classes.
        
        # We need a partition of E(H) into cliques.
        # This is the "Edge Clique Cover Number".
        
        # For small k (10, 20, 30), we can solve this exactly or with strong heuristics.
        # k=30 is small enough for a greedy or local search.
        
        # Algorithm for covering E_rem:
        # 1. Identify E_rem.
        # 2. Greedy:
        #    While E_rem is not empty:
        #       Find the largest clique S in C such that all edges in S are in E_rem.
        #       (i.e., S is a clique in the graph (C, E_rem)).
        #       Add S to the cover.
        #       Remove all edges in S from E_rem.
        
        # Finding the largest clique in (C, E_rem) is NP-hard, but k is small.
        # We can use a simple heuristic:
        # - Start with all vertices in C.
        # - Try to add vertices one by one, maintaining that the induced subgraph is a clique in E_rem.
        # - Or simply: iterate over all subsets? 2^30 is too big.
        # - Use a greedy maximal clique finder.
        
        # Better Heuristic for Edge Clique Cover:
        # 1. Sort edges in E_rem.
        # 2. Pick an edge (u, v).
        # 3. Find the largest clique in E_rem containing (u, v).
        # 4. Add it, remove edges.
        
        # To find largest clique containing (u, v) in E_rem:
        # Let G_rem be the graph (C, E_rem).
        # We want max clique in G_rem containing u and v.
        # This is equivalent to max clique in G_rem[ {w : w adj to u and v in G_rem} ] U {u, v}.
        
        # Given k <= 30, we can implement a branch-and-bound for max clique, or just a greedy one.
        # Greedy:
        # Candidates = {w in C : (u,w) in E_rem and (v,w) in E_rem}.
        # Sort candidates by degree in G_rem[Candidates].
        # Add highest degree candidate if it connects to all currently selected.
        
        # Let's implement this.
        
        # Build adjacency matrix for E_rem
        # rem_adj[u][v] = True if (u, v) in E_rem
        
        # First, determine which C-C edges are covered by FixedCliques.
        # covered_cc[u][v] = True if exists v in I such that u, v in N_C(v).
        
        # 1. Build N_C for each v in I.
        N_C = [set() for _ in range(n)]
        for u, v in edges_ci:
            N_C[v].add(u)
            
        # 2. Determine covered C-C edges.
        covered_cc = [[False]*k for _ in range(k)]
        for v in range(k, n):
            if N_C[v]:
                # N_C[v] is a set of C vertices
                # All pairs in N_C[v] are covered
                verts = list(N_C[v])
                for i in range(len(verts)):
                    for j in range(i+1, len(verts)):
                        covered_cc[verts[i]][verts[j]] = True
                        covered_cc[verts[j]][verts[i]] = True
                        
        # 3. Build E_rem
        E_rem = []
        rem_adj = [[False]*k for _ in range(k)]
        for u in range(k):
            for w in range(u+1, k):
                if not covered_cc[u][w]:
                    E_rem.append((u, w))
                    rem_adj[u][w] = True
                    rem_adj[w][u] = True
                    
        # 4. Cover E_rem with cliques
        cliques = []
        # Fixed cliques for I
        for v in range(k, n):
            if N_C[v]:
                clique = [v] + list(N_C[v])
                cliques.append(clique)
                
        # Now cover E_rem
        # We need to partition E_rem into cliques.
        # Greedy approach:
        remaining_edges = set(E_rem)
        
        # Helper: find max clique in rem_adj restricted to candidates, must contain start_u, start_v
        def find_max_clique(start_u, start_v, candidates, rem_adj):
            # candidates is a set of vertices that are adjacent to both start_u and start_v in rem_adj
            # We want to extend {start_u, start_v} to a max clique.
            # Greedy: sort candidates by degree in the induced subgraph of candidates.
            # Actually, we just need A clique, not necessarily max, to be fast.
            # But larger cliques are better for minimizing count.
            
            # Let's try a simple greedy:
            # current_clique = {start_u, start_v}
            # available = candidates
            # while available:
            #    pick v in available with max degree in available
            #    if v connects to all in current_clique (it does, by definition of available initially? No, available shrinks)
            #    Actually, if we pick v, new available = {w in available : w connects to v}
            
            # To ensure it's a clique, we must intersect neighborhoods.
            
            current = [start_u, start_v]
            # Start with candidates that connect to both
            # The input 'candidates' should already be filtered to those connecting to start_u and start_v.
            
            # Sort by degree in the subgraph induced by candidates?
            # Let's just iterate.
            
            # Optimization: if len(candidates) is small, we can be smarter.
            
            # Greedy selection:
            # 1. Compute degree of each node in candidates within the subgraph induced by candidates.
            # 2. Pick highest.
            # 3. Filter candidates to those connected to picked.
            
            # This is O(k^2) per step, O(k) steps. Total O(k^3) per clique.
            # Number of cliques <= k^2/2. Total O(k^5). k=30 -> 30^5 = 24M. Fast.
            
            avail = set(candidates)
            while avail:
                # Find node with max degree in avail
                best_node = None
                best_deg = -1
                for node in avail:
                    deg = 0
                    for other in avail:
                        if node != other and rem_adj[node][other]:
                            deg += 1
                    if deg > best_deg:
                        best_deg = deg
                        best_node = node
                        
                if best_node is None:
                    break
                    
                current.append(best_node)
                # Filter avail
                new_avail = set()
                for node in avail:
                    if node != best_node and rem_adj[node][best_node]:
                        new_avail.add(node)
                avail = new_avail
                
            return current

        # Main loop for covering E_rem
        # To make it deterministic and potentially better, sort edges by some heuristic?
        # Or just pick the first edge.
        
        # Let's pick edges in a fixed order (u, w) with u < w.
        # But we should pick edges that are "hardest" to cover?
        # Edges with low degree in rem_adj are harder (fewer options for large cliques).
        
        # Let's sort E_rem by the number of common neighbors in rem_adj?
        # Or just process in order.
        
        # To improve quality, we can restart with different edge orders if time permits?
        # For now, simple order.
        
        # Sort edges by (u, w)
        edges_to_cover = sorted(E_rem)
        
        # We need to track which edges are still uncovered.
        # Since we remove edges when we form a clique, we can just check if edge is still in remaining_edges.
        
        # However, the greedy "find max clique containing edge" might pick edges that are already covered by a previous clique?
        # No, we remove them.
        
        # Wait, if we form a clique S, we remove ALL edges in S from remaining_edges.
        # So we must iterate through edges_to_cover and check if they are still in remaining_edges.
        
        for u, w in edges_to_cover:
            if (u, w) not in remaining_edges:
                continue
                
            # Find candidates for clique containing (u, w)
            # Candidates are vertices v such that (u, v) in remaining_edges and (w, v) in remaining_edges
            # Note: rem_adj is static (original E_rem).
            # But we need to ensure the new clique only uses edges in remaining_edges.
            # If we use rem_adj, we might pick a vertex v where (u, v) was already covered.
            # So we must check remaining_edges dynamically.
            
            # This makes "max clique" search harder because the graph changes.
            # But k is small.
            
            candidates = set()
            for v in range(k):
                if v == u or v == w:
                    continue
                # Check if (u, v) and (w, v) are still in remaining_edges
                # Note: remaining_edges contains tuples (min, max)
                e1 = (u, v) if u < v else (v, u)
                e2 = (w, v) if w < v else (v, w)
                if e1 in remaining_edges and e2 in remaining_edges:
                    candidates.add(v)
                    
            # Now find a clique in the dynamic graph induced by candidates U {u, w}
            # that is a clique in the dynamic graph.
            # The dynamic graph has edges E_rem \ covered_so_far.
            
            # Let's do a greedy expansion on the dynamic graph.
            current_clique = [u, w]
            avail = set(candidates)
            
            while avail:
                # Pick best node from avail
                # Heuristic: max degree in the dynamic subgraph induced by avail
                best_node = None
                best_deg = -1
                for node in avail:
                    deg = 0
                    for other in avail:
                        if node == other:
                            continue
                        # Check if edge (node, other) is in remaining_edges
                        e = (node, other) if node < other else (other, node)
                        if e in remaining_edges:
                            deg += 1
                    if deg > best_deg:
                        best_deg = deg
                        best_node = node
                        
                if best_node is None:
                    break
                    
                current_clique.append(best_node)
                
                # Filter avail: keep only nodes connected to best_node in remaining_edges
                new_avail = set()
                for node in avail:
                    if node == best_node:
                        continue
                    e = (node, best_node) if node < best_node else (best_node, node)
                    if e in remaining_edges:
                        new_avail.add(node)
                avail = new_avail
                
            # Add clique to cover
            cliques.append(current_clique)
            
            # Remove all edges in current_clique from remaining_edges
            for i in range(len(current_clique)):
                for j in range(i+1, len(current_clique)):
                    a, b = current_clique[i], current_clique[j]
                    e = (a, b) if a < b else (b, a)
                    if e in remaining_edges:
                        remaining_edges.remove(e)
                        
        return cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline:
            # Fallback
            n, k, p, seed = spec
            edges_cc, edges_ci = instance(n, k, p, seed)
            cliques = [list(range(k))]
            for u, v in edges_ci:
                cliques.append([u, v])
            results.append(cliques)
            continue
            
        n, k, p, seed = spec
        edges_cc, edges_ci = instance(n, k, p, seed)
        
        cliques = solve_split_graph(n, k, edges_ci)
        
        # Validate and convert types
        # Ensure all are lists of ints
        valid_cliques = []
        for c in cliques:
            if len(c) < 2:
                continue
            # Check distinctness
            if len(set(c)) != len(c):
                continue
            # Convert to list of ints
            valid_cliques.append([int(x) for x in c])
            
        results.append(valid_cliques)
        
    return results
# EVOLVE-BLOCK-END
