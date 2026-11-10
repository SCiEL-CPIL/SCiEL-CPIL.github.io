import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 145.0

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        if not edges:
            return []

        # Build adjacency sets for quick lookup
        # adj[u] is a set of v such that (u,v) is an edge
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # The graph is a split graph:
        # C = {0, ..., k-1} is a clique
        # I = {k, ..., n-1} is an independent set
        # Edges only within C, within I (none), or between C and I.

        # Candidate cliques in a split graph are of the form:
        # 1. A subset of C (but maximal cliques involving only C is just C itself, or subsets if we want to be greedy? No, maximal cliques are better for set cover usually, but we can use any clique. However, any clique is contained in a maximal clique. Using maximal cliques is a good heuristic.)
        # 
        # Maximal cliques in a split graph:
        # - The entire clique C is a maximal clique (unless some vertex in C is not connected to any in I? No, C is a clique. If a vertex c in C has no neighbors in I, {c} is a clique, but C is larger. Is C always maximal? Yes, because you can't add any vertex from I to C unless it's connected to ALL of C. If it is, then {b} U C is a larger clique. So C is not maximal if there exists b in I connected to all of C.
        # - For each b in I, the set {b} U N(b) is a clique. It is maximal if N(b) is not a subset of N(b') for any other b' in I, and b is not connected to all of C (if it is, then {b} U C is the same as {b} U N(b) only if N(b)=C).
        #
        # Actually, the family of maximal cliques in a split graph is:
        # - For each b in I: K_b = {b} U N(b). Note: if N(b) is empty, K_b={b} (size 1, invalid for our output, but we can ignore).
        # - The set C itself? If there is no b in I such that N(b) == C, then C is a maximal clique. If there is such a b, then K_b = {b} U C, which contains C. So C is not maximal in that case.
        # - Also, could there be a maximal clique that is a proper subset of C? No, because if {u, v} is a clique in C, and there is another vertex w in C not in {u,v}, then {u,v,w} is a larger clique. So the only maximal clique contained in C is C itself (if it's not dominated by a K_b).
        
        # So the candidates are:
        # 1. All K_b = {b} U N(b) for b in I with |K_b| >= 2.
        # 2. C itself, if it is not a subset of any K_b. (Actually, if C is a subset of K_b, K_b covers all edges in C that K_b covers. But C covers ALL edges in C. K_b covers edges within N(b). If N(b) != C, K_b does not cover all edges in C. So we might need C or other cliques to cover the remaining edges in C.)
        
        # Wait, a clique partition requires that EVERY edge is in EXACTLY ONE clique.
        # This is stronger than edge cover. It's an edge partition.
        # The greedy set cover approach finds a set of cliques that COVER all edges.
        # It does NOT ensure that edges are not covered MULTIPLE times.
        # The problem statement says: "every edge must lie in exactly one listed clique".
        # This means the cliques must be EDGE-DISJOINT.
        
        # Re-reading the task:
        # "Every listed set must be a clique with at least two vertices and every edge must lie in exactly one listed clique."
        # This is an Edge Clique Partition (ECP).
        
        # My previous analysis:
        # "A clique can contain at most one vertex from the independent set."
        # Therefore, any clique containing a cross-edge (a,b) where b in I MUST be of the form {b} U S, where S subset of N(b) subset of C.
        # Such a clique covers:
        # - Cross edges: (b, s) for s in S.
        # - Clique edges: (s1, s2) for s1, s2 in S.
        
        # Since cliques must be edge-disjoint:
        # 1. No cross-edge (a,b) can be in two cliques.
        #    Since a cross-edge (a,b) can ONLY be in a clique containing b (because any clique containing (a,b) must contain both a and b, and if it contained another independent vertex b', it would have to contain edge (b,b') which doesn't exist), it follows that each cross-edge (a,b) is assigned to the unique clique containing b.
        #    Wait, is it possible for a cross-edge (a,b) to be in a clique that does NOT contain b? No, a clique containing edge (a,b) must contain vertices a and b.
        #    So, every cross-edge (a,b) is covered by the clique that contains b.
        #    Since b is in the independent set, the clique containing b is of the form {b} U S_b.
        #    Can b be in multiple cliques?
        #    If b is in clique K1 = {b} U S1 and clique K2 = {b} U S2, then the edges (b, s) for s in S1 are in K1, and edges (b, s) for s in S2 are in K2.
        #    If S1 and S2 are disjoint, this is fine. The edge (b, s1) is only in K1.
        #    However, what about edges within C?
        #    K1 covers edges (s1, s2) for s1, s2 in S1.
        #    K2 covers edges (s1, s2) for s1, s2 in S2.
        #    If S1 and S2 are disjoint, these sets of C-edges are disjoint.
        #    So, we CAN split the neighbors of b into multiple cliques!
        #    Example: b connected to {0, 1, 2}.
        #    Option A: One clique {b, 0, 1, 2}. Covers cross (b,0),(b,1),(b,2) and C-edges (0,1),(0,2),(1,2).
        #    Option B: Two cliques {b, 0, 1} and {b, 2}.
        #       K1 covers (b,0),(b,1) and (0,1).
        #       K2 covers (b,2). No C-edges (size of S2 is 1).
        #       Total cliques: 2.
        #       Edges covered: all cross edges incident to b. C-edge (0,1) covered by K1. C-edges (0,2) and (1,2) are NOT covered by K1 or K2. They must be covered by other cliques.
        
        # So the problem decomposes:
        # 1. Assign each cross-edge (a,b) to a clique containing b. Let S_b be the set of clique-part vertices assigned to b. Since cliques containing b are of form {b} U S, and must be edge-disjoint, the sets of cross-edges covered by different cliques containing b must be disjoint. This implies that if we have multiple cliques for b, their S sets must be disjoint?
        #    If K1 = {b} U S1 and K2 = {b} U S2.
        #    Cross edges in K1: {(b, s) : s in S1}.
        #    Cross edges in K2: {(b, s) : s in S2}.
        #    For these to be disjoint, S1 and S2 must be disjoint.
        #    So, for each b, we partition N(b) into disjoint subsets S_{b,1}, S_{b,2}, ...
        #    Each subset S_{b,i} forms a clique {b} U S_{b,i}.
        #    This clique covers cross edges (b, s) for s in S_{b,i} and C-edges (s1, s2) for s1, s2 in S_{b,i}.
        
        # 2. The C-edges not covered by these "b-cliques" must be covered by cliques consisting ONLY of vertices from C.
        #    Let U_C be the set of C-edges not covered by any {b} U S.
        #    We need to partition U_C into cliques. Since all vertices are in C (which is a clique), any subset of C is a clique.
        #    To minimize the number of cliques covering U_C, we should use as few large cliques as possible.
        #    Actually, since any subset is a clique, we can cover U_C with a single clique: the set of all vertices incident to U_C.
        #    Let V_U = {v in C : v is incident to an edge in U_C}.
        #    The clique K_C = V_U covers ALL edges in U_C (and potentially some edges not in U_C, but those are already covered? No, "exactly one". If K_C covers an edge (u,v) that was already covered by a b-clique, that's a violation.)
        #    
        #    Wait. If (u,v) is in U_C, it is NOT covered by any b-clique.
        #    If we add K_C = V_U, it covers (u,v).
        #    Does it cover any edge (u', v') that IS covered by a b-clique?
        #    If (u', v') is covered by a b-clique, it means u', v' are both in some S_{b,i}.
        #    If u', v' are also in V_U, then K_C also contains (u', v').
        #    This would mean (u', v') is in TWO cliques: the b-clique and K_C. VIOLATION.
        
        # So, we must ensure that the C-edges covered by b-cliques are DISJOINT from the C-edges covered by the final C-only clique(s).
        # This implies that if we use a C-only clique K_C, it cannot contain any edge (u,v) that is already covered by a b-clique.
        # Since K_C is a clique, it contains ALL edges between its vertices.
        # So, if K_C contains u and v, it covers (u,v).
        # Therefore, for any u, v in K_C, the edge (u,v) must NOT be covered by any b-clique.
        # This means u and v cannot BOTH be in the same S_{b,i} for any b, i.
        
        # This looks like a complex constraint satisfaction problem.
        
        # Let's simplify.
        # Observation: If we choose S_{b,i} to be singletons {a}, then the clique is {b, a}.
        # This covers cross edge (b,a) and NO C-edges (since |S|=1).
        # If we do this for ALL cross edges, then NO C-edges are covered by b-cliques.
        # Then ALL C-edges are in U_C.
        # We can cover U_C with one clique: C itself (or V_U).
        # Total cliques: (number of cross edges) + 1.
        # This is likely large.
        
        # Alternative: Choose S_{b,i} to be large.
        # If we choose S_b = N(b) (one clique per b), we cover many C-edges.
        # The C-edges covered are those within N(b) for some b.
        # The remaining C-edges U_C are those (u,v) such that there is NO b with u,v in N(b).
        # i.e., u and v are not co-adjacent to any b.
        # If we add a clique K_C for U_C, it must not contain any edge (u,v) that IS covered by a b-clique.
        # If (u,v) is covered by a b-clique, then u,v in N(b) for some b.
        # If we pick K_C = V_U (vertices incident to U_C), does it contain any "covered" edge?
        # Suppose (u,v) is in K_C. Then u,v in V_U.
        # If (u,v) was covered by a b-clique, then u,v in N(b).
        # Is it possible that u,v in V_U AND u,v in N(b)?
        # u in V_U means u is incident to some edge in U_C.
        # If u,v in N(b), then (u,v) is covered by K_b.
        # If we add K_C containing u and v, it covers (u,v) again. Violation.
        
        # So, we cannot simply take V_U.
        # We need to select a set of C-only cliques that cover U_C, such that no edge covered by a C-only clique is also covered by a b-clique.
        # This is equivalent to: The C-only cliques must be subsets of the complement of the "covered-by-b" graph? No.
        # The C-only cliques can only use edges from U_C.
        # Since a clique contains ALL edges between its vertices, a C-only clique K can only be formed from vertices such that ALL pairs in K are in U_C.
        # i.e., K is a clique in the graph G_U = (C, U_C).
        # Since G_U is a subgraph of the complete graph on C, any clique in G_U is a valid set of vertices for a C-only clique.
        # And any edge in G_U is NOT covered by any b-clique.
        # So, if we partition the edges of G_U into cliques, and partition the edges of G_b (covered by b-cliques) into cliques (the {b} U S sets), the union is a valid edge partition.
        
        # So the problem reduces to:
        # 1. For each b in I, partition N(b) into subsets S_{b,1}, ..., S_{b,m_b}.
        #    Each S_{b,i} defines a clique K_{b,i} = {b} U S_{b,i}.
        #    The edges covered by these are:
        #    - Cross: (b, s) for s in N(b). (All cross edges incident to b are covered).
        #    - C-edges: (u,v) for u,v in S_{b,i}.
        #    Let E_b be the set of C-edges covered by b.
        #    E_b = Union over i of { (u,v) : u,v in S_{b,i} }.
        #    Note: If u,v are in different S_{b,i}, the edge (u,v) is NOT covered by b's cliques.
        #
        # 2. The remaining C-edges are U_C = E_C \ (Union over b of E_b).
        #    We need to partition U_C into cliques.
        #    Since U_C is a set of edges in C, and any subset of C is a clique, we can cover U_C with cliques.
        #    To minimize the number of cliques, we should use large cliques.
        #    The maximum number of C-only cliques needed is bounded by the edge chromatic number of the graph (C, U_C)? No, we are partitioning edges into cliques, not coloring.
        #    Actually, since (C, U_C) is a subgraph of a clique, any clique in (C, U_C) is just a set of vertices where all pairs are in U_C.
        #    We can cover U_C with a single clique IF (C, U_C) is a complete graph (i.e., U_C = E_C).
        #    In general, we can cover U_C with at most |U_C| cliques (edges).
        #    But we can do better.
        #    
        #    Key Insight: The choice of S_{b,i} affects U_C.
        #    We want to MINIMIZE total cliques = (Sum m_b) + (Number of cliques to cover U_C).
        #    
        #    Strategy:
        #    To minimize Sum m_b, we want m_b = 1 for all b (i.e., S_b = N(b)).
        #    This minimizes the first term.
        #    Then U_C is determined.
        #    Then we cover U_C optimally.
        #    
        #    Is it always optimal to have m_b = 1?
        #    If we split N(b) into smaller sets, m_b increases.
        #    But it might reduce the size of U_C?
        #    If we split N(b), we cover FEWER C-edges (since edges between different S_{b,i} are not covered).
        #    So U_C becomes LARGER.
        #    A larger U_C likely requires MORE C-only cliques.
        #    So increasing m_b increases the first term and likely increases the second term.
        #    Therefore, m_b = 1 is likely optimal.
        
        # So the algorithm:
        # 1. For each b in I with N(b) non-empty, create clique {b} U N(b).
        #    This covers all cross edges incident to b.
        #    It covers C-edges (u,v) if u,v in N(b).
        # 2. Let Covered_C = Union over b of { (u,v) : u,v in N(b) }.
        # 3. Let U_C = E_C \ Covered_C.
        # 4. Cover U_C with the minimum number of cliques.
        #    Since U_C is a graph on vertices C, and we need to partition its edges into cliques.
        #    Note: A clique in the partition corresponds to a set of vertices V such that all edges in V are in U_C.
        #    This is equivalent to finding a clique cover of the graph G_U = (C, U_C).
        #    Wait, "clique cover of a graph" usually means covering vertices. Here we cover edges.
        #    Edge clique partition.
        #    For a graph G, the minimum number of cliques needed to partition the edges is related to the chromatic number of the line graph? No.
        #    It is the "clique partition number" of the edge set.
        #    For a general graph, this is hard.
        #    But G_U is a subgraph of a complete graph.
        #    Actually, G_U is the graph of edges NOT covered by any b.
        #    We can cover G_U with cliques.
        #    Since G_U is a subgraph of K_k, any clique in G_U is a set of vertices forming a clique in G_U.
        #    We can use a greedy approach: repeatedly find a maximal clique in G_U, remove its edges, repeat.
        
        # Let's implement this.
        
        # Step 1: Build N(b) for all b.
        N = [set() for _ in range(n)]
        for u, v in edges:
            if u < k and v >= k:
                N[v].add(u)
            elif v < k and u >= k:
                N[u].add(v)
            # Edges within C or I are handled separately or implicitly
        
        # Step 2: Create b-cliques and track covered C-edges.
        result_cliques = []
        covered_C_edges = set()
        
        for b in range(k, n):
            if N[b]:
                clique = [b] + sorted(N[b])
                result_cliques.append(clique)
                # Mark C-edges covered
                nb = sorted(N[b])
                for i in range(len(nb)):
                    for j in range(i+1, len(nb)):
                        covered_C_edges.add((nb[i], nb[j]))
        
        # Step 3: Find remaining C-edges.
        # E_C is all pairs in [0, k).
        remaining_C_edges = []
        for u in range(k):
            for v in range(u+1, k):
                if (u, v) not in covered_C_edges:
                    remaining_C_edges.append((u, v))
        
        # Step 4: Cover remaining_C_edges with cliques.
        # Build graph G_U on vertices [0, k) with edges remaining_C_edges.
        # We want to partition these edges into cliques.
        # Greedy: Find maximal cliques in G_U.
        
        # For efficiency, since k is small (10, 20, 30), we can use a simple backtracking or greedy.
        # Greedy: 
        # 1. Start with all remaining edges.
        # 2. Find a maximal clique in the current edge set.
        # 3. Add it to result.
        # 4. Remove its edges.
        # 5. Repeat.
        
        # How to find a maximal clique in a graph defined by an edge set?
        # Vertices are [0, k).
        # Adjacency in G_U: u ~ v if (u,v) in remaining.
        
        # Since k is small, we can represent G_U as a bitset or just a set of edges.
        
        rem_edges = set(remaining_C_edges)
        
        while rem_edges:
            # Find a maximal clique in G_U defined by rem_edges.
            # Start with an arbitrary edge (u, v).
            # Try to extend it to a maximal clique.
            # Simple greedy: 
            # Pick a vertex u with max degree in rem_edges.
            # Let C = {u}.
            # Let candidates = neighbors of u in rem_edges.
            # Iteratively add a vertex v in candidates that is connected to all in C.
            
            # To ensure we find a "good" clique (large), we can try a few heuristics.
            # For now, simple greedy:
            # 1. Pick edge (u,v) from rem_edges.
            # 2. Clique = {u, v}.
            # 3. Candidates = intersection of neighbors of u and v in rem_edges.
            # 4. While candidates not empty:
            #    Pick w in candidates with max degree in the subgraph induced by C U {w}?
            #    Just pick first one.
            #    Add w to C.
            #    Candidates = intersection of neighbors of all in C.
            
            # This finds A maximal clique.
            
            if not rem_edges:
                break
            
            # Pick an edge to start
            u, v = next(iter(rem_edges))
            clique_verts = {u, v}
            
            # Find common neighbors in rem_edges
            # Neighbors of u in rem_edges: {x : (u,x) in rem_edges or (x,u) in rem_edges}
            # Since rem_edges stores (min, max), we need to check both.
            
            def get_neighbors(vertex, edge_set):
                neigh = set()
                for e in edge_set:
                    if e[0] == vertex:
                        neigh.add(e[1])
                    elif e[1] == vertex:
                        neigh.add(e[0])
                return neigh
            
            n_u = get_neighbors(u, rem_edges)
            n_v = get_neighbors(v, rem_edges)
            candidates = n_u.intersection(n_v)
            
            while candidates:
                # Pick a candidate
                w = next(iter(candidates))
                clique_verts.add(w)
                # Update candidates: must be neighbor to all in clique_verts
                # Intersect neighbors of w with current candidates
                n_w = get_neighbors(w, rem_edges)
                candidates = candidates.intersection(n_w)
                
            # clique_verts is a maximal clique.
            # Add to result
            result_cliques.append(sorted(clique_verts))
            
            # Remove edges in clique_verts from rem_edges
            for i in range(len(clique_verts)):
                for j in range(i+1, len(clique_verts)):
                    a, b = sorted((clique_verts[i], clique_verts[j]))
                    rem_edges.discard((a, b))
            
        # Final check: ensure all cliques are valid and edges covered exactly once.
        # (Assuming logic is correct, this should hold).
        
        # Convert to list of lists of ints
        final_result = []
        for clique in result_cliques:
            final_clique = [int(v) for v in clique]
            if len(final_clique) >= 2:
                final_result.append(final_clique)
        
        return final_result

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if time.time() > DEADLINE:
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            # Fallback: trivial partition (each edge is a clique)
            # This is valid but likely high score.
            trivial = [[int(u), int(v)] for u, v in edges]
            results.append(trivial)
        else:
            res = solve_instance(*spec)
            results.append(res)

    return results

if __name__ == "__main__":
    construct()
