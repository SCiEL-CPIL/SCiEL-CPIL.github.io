"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_one(n, k, p, seed):
        edges = instance(n, k, p, seed)
        edge_set = set(edges)

        # Identify cross-edges: (u, v) with u < k <= v
        cross_edges = [(u, v) for u in range(k) for v in range(k, n) if (u, v) in edge_set]
        k_edges = [(u, v) for u in range(k) for v in range(u + 1, k)]

        # For each independent vertex v (k <= v < n), collect its K-neighbors
        # N(v) = {u in [0,k) : (u,v) is an edge}
        N = {}
        for v in range(k, n):
            N[v] = [u for u in range(k) if (u, v) in edge_set]

        # Strategy:
        # 1. Start with one big clique on all K vertices: covers all k_edges
        # 2. For each v in I with |N(v)| >= 1, we need to cover edges (u,v) for u in N(v).
        #    Since I is independent, any clique containing v can contain at most one I-vertex.
        #    A clique {v} ∪ S where S ⊆ K is valid iff S is a clique in K (which it always is,
        #    since K is a clique) AND every u in S is adjacent to v (i.e., S ⊆ N(v)).
        #    
        #    The key constraint: every edge must appear in EXACTLY ONE clique.
        #    The big K-clique covers all k-edges. So no other clique can contain two K-vertices
        #    (that would create a k-edge in two cliques).
        #    
        #    Therefore, any clique containing v can have at most ONE K-vertex.
        #    So each cross-edge (u,v) must be its own 2-clique {u,v}.
        #    
        #    UNLESS we remove some K-vertices from the big K-clique and cover those K-edges
        #    differently. But that would require additional cliques for those K-edges, which
        #    doesn't help since K is a complete graph.
        #    
        #    Wait - what if we DON'T use one big K-clique? What if we partition K into smaller
        #    cliques? That would increase the count. So the optimal is to use one K-clique.
        #    
        #    Therefore: 1 + len(cross_edges) is the minimum.
        #    
        #    BUT WAIT - can we merge some cross-edges? No, because each cross-edge (u,v) needs
        #    to be in a clique, and that clique can only contain u, v, and possibly other K-vertices
        #    adjacent to v. But if it contains another K-vertex u', then (u,u') is a K-edge that
        #    is already in the big K-clique. Violation!
        #    
        #    So the answer is indeed 1 + len(cross_edges).

        cliques = [list(range(k))]  # The big K-clique
        for (u, v) in cross_edges:
            cliques.append([u, v])

        # Verify types
        for i, c in enumerate(cliques):
            cliques[i] = list(c)
            for j in range(len(c)):
                if not isinstance(c[j], int) or isinstance(c[j], bool):
                    cliques[i][j] = int(c[j])

        return cliques

    deadline = time.time() + 140.0
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if time.time() > deadline:
            break
        results.append(solve_one(*spec))

    # Ensure we have exactly 3 results
    while len(results) < 3:
        results.append([])

    return results
# EVOLVE-BLOCK-END
