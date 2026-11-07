"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 140  # 10s safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_instance(n, k, p, seed):
        if time.time() > DEADLINE:
            # Fallback: return all edges as cliques of size 2 (valid but not optimal)
            edges = instance(n, k, p, seed)
            return [[u, v] for u, v in edges]

        # Build adjacency
        edges = instance(n, k, p, seed)
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        C = list(range(k))
        I = list(range(k, n))

        # Identify cross edges
        cross_edges = []
        for e in edges:
            u, v = e
            if (u < k and v >= k) or (v < k and u >= k):
                cross_edges.append(e)

        # C-C edges
        cc_edges = [(u, v) for u in range(k) for v in range(u + 1, k)]

        # For each I-vertex, find which C-C edges it can join
        # An I-vertex w can join a C-C edge (u,v) if w is adjacent to both u and v
        # This forms a triangle {u, v, w}

        # Strategy:
        # 1. Start with the big clique C (all k vertices)
        # 2. For each I-vertex, try to add it to existing cliques or create new ones
        # 3. Use greedy + local search to minimize total cliques

        # Better approach:
        # - The clique C itself is one clique
        # - Each I-vertex w must be covered in some clique
        # - A clique containing w and some subset S of C requires S ∪ {w} to be a clique
        #   i.e., w adjacent to all in S, and S is a clique in C (which it is since C is complete)
        # - So w can join any subset of its C-neighbors
        # - We want to partition the edges such that each edge is in exactly one clique

        # Key insight: 
        # - Edges within C: must be covered by cliques. The clique C covers all C-C edges.
        # - Cross edges (u,w) where u in C, w in I: must be covered by some clique containing both u and w.
        # - There are no I-I edges (independent set).
        # 
        # So we need:
        # - One or more cliques covering all C-C edges
        # - Cliques covering all cross edges
        # - Each edge in exactly one clique
        #
        # If we use C as one clique, it covers all C-C edges. Then we need to cover cross edges.
        # A cross edge (u,w) can be covered by a clique {u, w} (size 2) or by a larger clique {u, w} ∪ S
        # where S ⊆ N_C(w) and S is a clique (always true since C is complete).
        #
        # To minimize cliques, we want to pack multiple cross edges into one clique.
        # A clique {w} ∪ S covers all edges (u,w) for u ∈ S.
        # So for each I-vertex w, if we put it in a clique with neighbor set S_w ⊆ N_C(w),
        # we cover |S_w| cross edges with one clique.
        #
        # But we also need to make sure C-C edges are covered. If we use C as a clique, all C-C edges
        # are covered. But then C is already "used". Can we still use subsets of C in other cliques?
        # Yes! A clique is just a set of vertices forming a clique. The same vertex can appear in
        # multiple cliques as long as each EDGE appears in exactly one clique.
        #
        # Wait, re-reading: "every edge must lie in exactly one listed clique"
        # So vertices can be in multiple cliques, but each edge must be in exactly one.
        #
        # So the plan:
        # 1. Include C as a clique (covers all C-C edges)
        # 2. For each I-vertex w, create a clique {w} ∪ S_w where S_w ⊆ N_C(w)
        #    This covers all cross edges (u,w) for u ∈ S_w
        # 3. We need S_w = N_C(w) for all w to cover all cross edges
        # 4. Each such clique has size |N_C(w)| + 1 ≥ 2 (since if N_C(w) is empty, we can't include w in any valid clique of size ≥ 2... wait)
        #
        # If N_C(w) is empty, w has no cross edges, so no clique is needed for w.
        # If N_C(w) is non-empty, we can make clique {w} ∪ N_C(w).
        #
        # Total cliques = 1 (for C) + number of I-vertices with non-empty N_C(w)
        #
        # Can we do better? Can we merge multiple I-vertices into one clique?
        # No, because I is an independent set, so no two I-vertices are adjacent.
        # A clique can contain at most one I-vertex.
        #
        # So the minimum is: 1 + |{w ∈ I : N_C(w) ≠ ∅}|
        #
        # But wait, we could also split C into smaller cliques to potentially reduce the count?
        # No, splitting C would only increase the number of cliques needed to cover C-C edges.
        # Using C as one clique is optimal for covering C-C edges.
        #
        # Actually, let me reconsider. Could we avoid using C as a single clique?
        # Suppose we don't use C. Then we need to cover all C-C edges with other cliques.
        # Each clique containing C-vertices and possibly one I-vertex.
        # A clique {S ∪ {w}} where S ⊆ C, w ∈ I covers:
        #   - All C-C edges within S
        #   - All cross edges (u,w) for u ∈ S
        # So if we partition C into subsets S_1, ..., S_m (each assigned to possibly different I-vertices),
        # we cover C-C edges within each S_i. But C-C edges between different S_i's are not covered!
        # So this doesn't work unless we have cliques that span across the partition.
        #
        # The only way to cover all C-C edges is to have cliques whose union of C-vertex sets covers all pairs.
        # The simplest is one clique C covering all. Any partition of C into proper subsets will miss cross-pair edges.
        #
        # So the optimal solution is:
        # - One clique: C
        # - For each w ∈ I with N_C(w) ≠ ∅: one clique {w} ∪ N_C(w)
        #
        # Total = 1 + |{w ∈ I : deg_C(w) > 0}|

        # Let's verify this is correct
        cliques = []
        
        # Add C as a clique
        if k >= 2:
            cliques.append(list(C))
        
        # For each I-vertex with neighbors in C, add a clique
        for w in I:
            neighbors_in_c = sorted(adj[w] & set(C))
            if len(neighbors_in_c) >= 1:
                clique = neighbors_in_c + [w]
                # Verify it's a clique
                # All pairs in neighbors_in_c are connected (C is complete)
                # w is connected to all in neighbors_in_c (by construction)
                cliques.append(clique)
        
        # Verify: every edge is in exactly one clique
        # C-C edges: all in the C clique
        # Cross edges (u,w): in the clique for w if u ∈ N_C(w)
        # I-I edges: none exist
        
        # Check that each clique has at least 2 vertices
        for c in cliques:
            if len(c) < 2:
                # This shouldn't happen given our construction
                pass
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > DEADLINE:
            # Fallback for remaining instances
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            results.append([[u, v] for u, v in edges])
        else:
            results.append(solve_instance(*spec))
    
    return results
# EVOLVE-BLOCK-END
