"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    def instance(n, k, p, seed):
        rng = Random(seed)
        # Generate internal edges of the clique C
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        # Generate cross edges between C and I
        cross_edges = [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return edges, cross_edges

    def solve_one(n, k, p, seed, deadline):
        internal_edges, cross_edges = instance(n, k, p, seed)
        
        # Build adjacency sets
        # C vertices: 0..k-1, I vertices: k..n-1
        # adj_C[u] = set of v in C such that (u,v) is an edge
        adj_C = [set() for _ in range(k)]
        for u, v in internal_edges:
            adj_C[u].add(v)
            adj_C[v].add(u)
            
        # adj_I[v] = set of u in C such that (u,v) is an edge
        adj_I = [set() for _ in range(n)]
        for u, v in cross_edges:
            adj_I[v].add(u)
            # Note: we don't add to adj_C for cross edges because C is a clique,
            # but for the "remaining edges" logic, we track which C-C edges are covered.
            
        # We need to cover:
        # 1. All edges in C (internal_edges)
        # 2. All edges between C and I (cross_edges)
        
        # Strategy:
        # We will form cliques.
        # A clique can contain vertices from C and I.
        # Since I is an independent set, a clique can contain at most one vertex from I.
        # So each clique is either:
        #   a) A subset of C (size >= 2)
        #   b) {v} U S where v in I, S subset of C, and S U {v} is a clique.
        #      This requires S to be a clique in C AND every u in S to be adjacent to v in I.
        #      i.e., S subset of N_C(v) (neighbors of v in C).
        
        # Let's maintain the set of uncovered edges.
        # uncovered_CC: set of (u, v) with u < v, both in C
        # uncovered_CI: for each v in I, set of u in C such that (u, v) is uncovered.
        
        uncovered_CC = set(internal_edges)
        uncovered_CI = {}
        for v in range(k, n):
            uncovered_CI[v] = set(adj_I[v])
            
        cliques = []
        
        # Helper: find a large clique in a set of C-vertices that are mutually adjacent in C
        # and also adjacent to a specific v in I (if v is not None).
        # Actually, if we are covering edges for v in I, we need S subset of N_C(v).
        # And S must be a clique in C.
        # If v is None, we just need S to be a clique in C (to cover CC edges).
        
        def max_clique_in_C(candidates, required_neighbors=None):
            """
            Find a large clique in C from 'candidates'.
            If required_neighbors is not None, the clique must be a subset of required_neighbors.
            candidates is a set of ints (vertices in C).
            """
            if not candidates:
                return []
            
            # Filter candidates
            if required_neighbors is not None:
                candidates = candidates & required_neighbors
            
            if not candidates:
                return []
            
            # Greedy Max Clique on C
            # Since C is a clique in the original graph, any subset is a clique?
            # NO. The graph is a split graph. C is a clique in the split graph definition?
            # "a clique on the first k vertices" -> Yes, C is a clique.
            # So ANY subset of C is a clique in G.
            # Wait, if C is a clique, then any subset S of C is a clique.
            # Then the constraint "S U {v} is a clique" just means S subset of N_C(v).
            # And "S is a clique" is automatically satisfied for any S subset of C.
            
            # So, for a vertex v in I, we can pick ANY subset S of N_C(v) to form a clique {v} U S.
            # To maximize edges covered, we should pick S = N_C(v) \ (already covered neighbors)?
            # No, we need to cover edges (u, v) for u in N_C(v).
            # If we pick S, we cover edges (u, v) for all u in S.
            # We also cover edges (u, w) for u, w in S.
            
            # So the problem reduces to:
            # We have a hypergraph or just sets.
            # For each v in I, we have a set N_v of neighbors in C.
            # We can choose a subset S_v of N_v.
            # This covers:
            #   - All edges (u, v) for u in S_v.
            #   - All edges (u, w) for u, w in S_v.
            # We want to cover all edges in CC and all edges in CI.
            # Minimize number of cliques.
            # Note: We can also use cliques entirely in C to cover CC edges.
            
            # Since C is a clique, the edges in CC form a complete graph K_k.
            # We need to cover all edges of K_k using cliques.
            # A clique in C of size s covers s(s-1)/2 edges.
            # But we can also cover CC edges using cliques that include a vertex from I.
            # If we use {v} U S_v, it covers all edges within S_v.
            
            # Let's try a greedy approach:
            # 1. For each v in I, pick S_v = N_v.
            #    This covers all CI edges incident to v.
            #    It also covers all CC edges within N_v.
            # 2. After processing all v in I, some CC edges might remain uncovered.
            #    These are edges (u, w) such that there is no v with u, w in N_v.
            #    We cover these with additional cliques in C.
            
            # Is this optimal?
            # Using {v} U N_v is one clique. It covers all edges incident to v.
            # If we split N_v into smaller subsets, we use more cliques.
            # So it's best to take the largest possible S_v for each v.
            # The largest possible S_v is N_v itself.
            
            # So, simple strategy:
            # For each v in I:
            #   S_v = N_v
            #   If S_v is non-empty:
            #      Add clique [v] + list(S_v)
            #      Remove all edges (u, v) for u in S_v from uncovered_CI[v]
            #      Remove all edges (u, w) for u, w in S_v from uncovered_CC
            #
            # Then, cover remaining uncovered_CC edges.
            # Since C is a clique, the remaining uncovered_CC edges form a graph H on C.
            # We need to cover edges of H with cliques.
            # Since H is a subgraph of a clique, we can just find maximal cliques in H.
            # Or, since we just need to cover edges, we can pair them up?
            # No, a clique of size s covers s choose 2 edges.
            # To minimize number of cliques, we want large cliques.
            # In a general graph H, finding min clique cover is hard.
            # But H is a subgraph of K_k.
            # Actually, we can just take the whole set of vertices incident to remaining edges?
            # No, they might not form a clique.
            # But wait, if we just take the set of vertices V_rem incident to remaining edges,
            # is V_rem a clique? No.
            # But we can decompose H into cliques.
            # Greedy: find a maximal clique in H, remove its edges, repeat.
            
            pass # This helper is not needed if we use the simple strategy directly.

        # Simple Strategy Implementation:
        
        # Step 1: Process I vertices
        for v in range(k, n):
            if time.time() > deadline:
                break # Fallback later if needed
                
            N_v = set(uncovered_CI[v])
            if not N_v:
                continue
                
            # The clique is {v} U N_v
            # Check if N_v has at least 1 vertex? 
            # Task says "at least two vertices".
            # If N_v is empty, no clique.
            # If N_v has 1 vertex u, clique is [v, u]. Size 2. OK.
            
            clique = [v] + list(N_v)
            cliques.append(clique)
            
            # Update uncovered edges
            # Remove CI edges
            uncovered_CI[v] = set()
            
            # Remove CC edges within N_v
            N_v_list = list(N_v)
            for i in range(len(N_v_list)):
                for j in range(i + 1, len(N_v_list)):
                    u, w = N_v_list[i], N_v_list[j]
                    e = (min(u, w), max(u, w))
                    uncovered_CC.discard(e)
        
        # Step 2: Cover remaining CC edges
        if uncovered_CC:
            # Find vertices involved
            V_rem = set()
            for u, w in uncovered_CC:
                V_rem.add(u)
                V_rem.add(w)
            
            # We need to cover edges in uncovered_CC using cliques.
            # Greedy:
            # While uncovered_CC is not empty:
            #   Pick a vertex u in V_rem with max degree in uncovered_CC
            #   Find neighbors of u in uncovered_CC: N_u
            #   Clique S = {u} U N_u
            #   Add S to cliques
            #   Remove edges within S from uncovered_CC
            
            # To speed up, we can just take all V_rem as one clique?
            # No, V_rem might not be a clique.
            # But we can find a maximal clique in the graph induced by V_rem with edges uncovered_CC.
            
            # Since k is small (10, 20, 30), we can do this.
            
            # Build adjacency for remaining CC graph
            adj_rem = {u: set() for u in V_rem}
            for u, w in uncovered_CC:
                adj_rem[u].add(w)
                adj_rem[w].add(u)
                
            while uncovered_CC:
                if time.time() > deadline:
                    # Fallback: add edges as pairs?
                    # Just add all remaining edges as cliques of size 2?
                    # Or just break and hope?
                    # Better: add a clique for each remaining edge.
                    for u, w in uncovered_CC:
                        cliques.append([u, w])
                    uncovered_CC = set()
                    break
                    
                # Pick vertex with max degree
                u_max = max(V_rem, key=lambda x: len(adj_rem[x]))
                if not adj_rem[u_max]:
                    # Should not happen if uncovered_CC is not empty and V_rem is updated
                    break
                    
                S = {u_max} | adj_rem[u_max]
                cliques.append(list(S))
                
                # Remove edges within S
                S_list = list(S)
                for i in range(len(S_list)):
                    for j in range(i + 1, len(S_list)):
                        a, b = S_list[i], S_list[j]
                        e = (min(a, b), max(a, b))
                        if e in uncovered_CC:
                            uncovered_CC.discard(e)
                            
                # Update adj_rem and V_rem
                for s in S:
                    for t in list(adj_rem[s]):
                        if t in S:
                            adj_rem[s].discard(t)
                            adj_rem[t].discard(s)
                    if not adj_rem[s]:
                        V_rem.discard(s)
                        
        return cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    for n, k, p, seed in specs:
        if time.time() > deadline:
            # Emergency fallback: just return empty or simple pairs?
            # Must be valid.
            # If we run out of time, we can't compute.
            # But 140s is generous.
            # Let's just compute the simple solution which is very fast.
            pass
            
        cliques = solve_one(n, k, p, seed, deadline)
        
        # Convert to list of lists of ints
        # Ensure types are correct
        clean_cliques = []
        for c in cliques:
            clean_cliques.append([int(x) for x in c])
            
        results.append(clean_cliques)
        
    return results
# EVOLVE-BLOCK-END
