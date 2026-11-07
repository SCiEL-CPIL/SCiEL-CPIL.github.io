"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    import time
    from itertools import combinations
    
    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)
    
    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Build adjacency: for each vertex, set of neighbors
        adj = [set() for _ in range(n)]
        edge_set = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((min(u,v), max(u,v)))
        
        # We need to cover every edge exactly once with cliques of size >= 2
        # Strategy: Greedy - pick edges in order, for each uncovered edge, 
        # extend to maximal clique containing that edge, remove all its edges
        
        remaining_edges = set(edge_set)
        cliques = []
        
        # Sort edges to process: prioritize edges that are "harder" to cover
        # Edges within K are in a dense region; cross edges may be sparse
        # Process cross edges first (they can only be in cliques with K vertices),
        # then K-internal edges
        
        cross_edges = [(u,v) for u,v in remaining_edges if u < k and v >= k]
        k_edges = [(u,v) for u,v in remaining_edges if u < k and v < k]
        
        # For cross edges: each such edge (u,v) where u in K, v in I
        # can only be in a clique containing u and v, plus other K vertices
        # that are adjacent to v
        
        # Greedy: for each cross edge, form the maximal clique
        # containing it: {u, v} + {w in K : w != u, w adjacent to v}
        # But we need to be careful about edge overlap
        
        # Better approach: group by I-vertex
        # For each I-vertex v, the edges from v go to some subset S_v of K
        # These edges can be covered by cliques: for each subset of S_v that forms
        # a clique with v... but since K is a clique, any subset of S_v plus v is a clique
        
        # The edges from v to S_v need to be partitioned into cliques.
        # Each clique containing v and some subset T of S_v covers edges (v,w) for w in T
        # and edges (w1,w2) for w1,w2 in T.
        
        # So the problem decomposes: for each I-vertex v, partition S_v into subsets
        # such that the edges within each subset are also covered (by the clique itself).
        # But edges within K are shared across multiple I-vertices' cliques!
        
        # This is the key constraint: an edge (a,b) within K can be covered by:
        # - A clique containing only K vertices
        # - A clique containing v (I-vertex) and {a,b} subset of S_v
        # But NOT by two different cliques.
        
        # So we need to decide for each K-edge whether it's covered by a K-only clique
        # or by a clique that includes some I-vertex.
        
        # Greedy strategy: 
        # 1. For each I-vertex v, we MUST cover edges (v, w) for w in S_v
        # 2. Each such edge must be in a clique with v and w, possibly with other K vertices
        # 3. To minimize cliques, we want large cliques
        # 4. But we also need to cover K-edges
        
        # Let's try: for each I-vertex v, create ONE clique: {v} union S_v
        # This covers all edges (v,w) for w in S_v, AND all edges (w1,w2) for w1,w2 in S_v
        # Then remaining K-edges are those not in any S_v (i.e., K-edges where both endpoints
        # are not in S_v for any v, or more precisely, K-edges not covered by any I-vertex clique)
        
        # Wait, but if we make {v} union S_v for each v, then a K-edge (a,b) is covered
        # by clique of v if both a and b are in S_v. It could be covered by multiple v's cliques!
        # That violates "exactly once".
        
        # So we need to assign each K-edge to exactly one covering clique.
        
        # Revised strategy:
        # 1. For each I-vertex v, the edges (v,w) for w in S_v must each be in some clique
        #    containing v and w.
        # 2. Group these edges into cliques. The natural grouping: one clique per I-vertex
        #    containing v and all of S_v. But this over-covers K-edges.
        # 3. Alternative: for each I-vertex v, we can split S_v into multiple groups,
        #    each forming a clique with v.
        
        # Let me think differently. The constraint is:
        # - Every edge is in exactly one clique
        # - Every clique has size >= 2
        
        # For I-vertex v with neighbors S_v in K:
        # - Edges (v, w) for w in S_v must be covered
        # - Each such edge must be in a clique containing v and w
        # - Multiple edges (v, w1), (v, w2) can be in the same clique if w1, w2 are also
        #   in that clique (and they are adjacent since K is a clique)
        
        # So for vertex v, we partition S_v into groups G1, G2, ..., Gm
        # Each group Gi forms a clique {v} union Gi
        # This covers edges (v, w) for w in Gi, and edges (w1, w2) for w1, w2 in Gi
        
        # The K-edges covered by v's cliques are the edges within each Gi
        # A K-edge (a, b) is covered by v's partition if a and b are in the same group Gi
        
        # K-edges NOT covered by any I-vertex's partition must be covered by K-only cliques
        
        # To minimize total cliques:
        # - We want few groups per I-vertex (fewer cliques)
        # - We want K-edges to be covered by I-vertex cliques when possible (so fewer K-only cliques needed)
        # - But a K-edge can only be in ONE clique total
        
        # Greedy approach:
        # 1. For each I-vertex v, put all of S_v in one group: clique {v} union S_v
        # 2. Mark all K-edges within S_v as "covered by v"
        # 3. For K-edges covered by multiple v's, we have a conflict!
        
        # So we need to resolve conflicts. Let's try:
        # 1. For each I-vertex v, tentatively assign all S_v to one group
        # 2. For each K-edge (a,b), count how many v's have both a,b in S_v
        # 3. Assign (a,b) to one such v (or to a K-only clique if none)
        # 4. But then v's clique {v} union S_v would cover (a,b) even if we assigned it elsewhere
        #    - This doesn't work because the clique covers ALL edges within it
        
        # The issue: if v's clique is {v} union S_v, it covers ALL edges within S_v,
        # not just the ones we want to assign to v.
        
        # So we CANNOT have a clique {v} union S_v if some edges within S_v are assigned
        # to other cliques. We must split S_v into groups such that each group's internal
        # edges are all assigned to that group's clique.
        
        # This is getting complex. Let me try a different approach:
        
        # Since the graph is small (n <= 90), let me try a greedy edge-based approach:
        
        remaining = set(edge_set)
        cliques = []
        
        # Sort edges: cross edges first, then K-edges
        # For cross edges, they can only be in cliques with at most one I-vertex
        # (since I is independent)
        
        cross_edges = sorted([(u,v) for u,v in remaining if (u < k and v >= k) or (u >= k and v < k)], 
                           key=lambda e: (min(e), max(e)))
        k_edges = sorted([(u,v) for u,v in remaining if u < k and v < k],
                        key=lambda e: (min(e), max(e)))
        
        # Process cross edges: for each, extend to maximal clique
        for e in cross_edges:
            if e not in remaining:
                continue
            u, v = e
            # Ensure u is in K, v is in I
            if u >= k:
                u, v = v, u
            # u in K, v in I
            # Maximal clique containing (u,v): {u, v} + {w in K, w != u, w adjacent to v}
            candidates = [u, v]
            for w in range(k):
                if w != u and w in adj[v]:
                    candidates.append(w)
            # Form clique with all candidates
            clique = tuple(sorted(candidates))
            # Check all edges in this clique are in remaining
            valid = True
            for a, b in combinations(clique, 2):
                edge = (min(a,b), max(a,b))
                if edge not in remaining:
                    valid = False
                    break
            if valid:
                cliques.append(list(clique))
                for a, b in combinations(clique, 2):
                    remaining.discard((min(a,b), max(a,b)))
            else:
                # Just use the edge itself as a 2-clique
                cliques.append([u, v])
                remaining.discard(e)
        
        # Now process remaining K-edges
        # These are K-edges not covered by any cross-edge clique
        for e in k_edges:
            if e not in remaining:
                continue
            u, v = e
            # Extend to maximal clique within remaining K-edges
            # Find all K-vertices w such that (u,w) and (v,w) are in remaining
            candidates = [u, v]
            for w in range(k):
                if w != u and w != v:
                    if (min(u,w), max(u,w)) in remaining and (min(v,w), max(v,w)) in remaining:
                        candidates.append(w)
            clique = tuple(sorted(candidates))
            valid = True
            for a, b in combinations(clique, 2):
                edge = (min(a,b), max(a,b))
                if edge not in remaining:
                    valid = False
                    break
            if valid:
                cliques.append(list(clique))
                for a, b in combinations(clique, 2):
                    remaining.discard((min(a,b), max(a,b)))
            else:
                cliques.append([u, v])
                remaining.discard(e)
        
        # Verify no edges remain
        assert len(remaining) == 0, f"Remaining edges: {remaining}"
        
        return cliques
    
    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        results.append(solve_instance(*spec))
    
    return results
# EVOLVE-BLOCK-END
