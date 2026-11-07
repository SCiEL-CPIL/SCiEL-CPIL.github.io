"""Clique partition of split graphs using slot-based assignment."""
import time

# EVOLVE-BLOCK-START
def construct():
    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve(n, k, p, seed):
        rng = Random(seed)
        # Generate edges
        K = list(range(k))
        I = list(range(k, n))
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        cross = {}
        for v in range(k, n):
            nbrs = []
            for u in range(k):
                if rng.random() < p:
                    nbrs.append(u)
            cross[v] = nbrs
            edges += [(u, v) for u in nbrs]

        # Build adjacency sets
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Greedy clique cover: repeatedly find a maximal clique among uncovered edges
        # Use a priority: prefer cliques that cover many edges
        edge_set = set(edges)
        cliques = []

        def is_clique(vertices):
            for i in range(len(vertices)):
                for j in range(i+1, len(vertices)):
                    if vertices[j] not in adj[vertices[i]]:
                        return False
            return True

        def clique_edges(vertices):
            s = set()
            for i in range(len(vertices)):
                for j in range(i+1, len(vertices)):
                    a, b = vertices[i], vertices[j]
                    s.add((min(a,b), max(a,b)))
            return s

        # Strategy: for split graphs, we can be smarter.
        # The clique K is a clique. Each independent vertex v has neighbors in K.
        # A clique containing v must be {v} U S where S subset of neighbors(v) and S is a clique in K (always true since K is a clique).
        # So any {v} U S where S subset of neighbors(v) is a clique.
        # Edges in K-K: covered by cliques within K.
        # Edges v-u (v in I, u in K): must be in a clique containing v and u.
        # Edges v-w (v,w in I): none (I is independent).

        # So we need to cover:
        # 1. All edges within K: can be done with one clique K (if |K|>=2)
        # 2. All cross edges: each cross edge (v,u) must be in a clique {v} U S where u in S.

        # For cross edges: assign each independent vertex v to a "slot" = a subset S_v of its neighbors.
        # Then clique {v} U S_v covers all edges (v, u) for u in S_v.
        # But we need ALL neighbors of v to be covered, so S_v must = neighbors(v).
        # That gives one clique per independent vertex: {v} U neighbors(v).
        # Plus one clique K for the K-K edges.
        # Total: 1 + |I| = 1 + (n-k) cliques.

        # But we can do better: if two independent vertices v, w have the same neighbor set,
        # we can't merge them (they're not adjacent). But if v's neighbors are a subset of w's...
        # Actually, we can group: a clique can contain at most one independent vertex.
        # So each independent vertex needs its own clique (or be part of a clique with K vertices).
        # Wait: a clique can be {v, u1, u2, ...} where v is independent and u's are in K.
        # It cannot contain two independent vertices.
        # So minimum is at least the number of independent vertices with at least one neighbor.
        # Plus we need to cover K-K edges.

        # Optimal strategy:
        # - One clique for K (covers all K-K edges): 1 clique (if k >= 2)
        # - For each independent vertex v with neighbors, one clique {v} U neighbors(v): covers all cross edges from v
        # Total: 1 + count(v in I with neighbors(v) non-empty)

        # But wait: can we avoid the separate K clique by including K vertices in the v-cliques?
        # The K-K edges are between two K vertices. They can only be covered by a clique that contains both.
        # A clique {v} U S contains K-K edges only between pairs in S.
        # So if we don't have a pure K clique, we need the union of all S_v to cover all K-K edges.
        # That means for every pair (u, u') in K, there exists some v with u, u' both in neighbors(v).
        # This is not guaranteed. So safest is to include K as a clique.

        # Let's compute: 1 (for K) + number of independent vertices with at least one neighbor.

        cliques = []
        # K clique
        if k >= 2:
            cliques.append(K)

        # For each independent vertex with neighbors
        for v in I:
            nbrs = cross[v]
            if len(nbrs) >= 1:
                # Clique: {v} U nbrs
                clique = [v] + sorted(nbrs)
                # Need at least 2 vertices
                if len(clique) >= 2:
                    cliques.append(clique)
                # If nbrs is empty, no edges from v, skip
            # If v has no neighbors, it's isolated, no edges to cover

        # Verify: every edge is in exactly one clique
        # K-K edges: only in K clique
        # Cross edges (v, u): only in the clique for v (since u is in K, and v is independent,
        #   the only clique containing v is v's own clique)
        # This works! Each edge is in exactly one clique.

        # But wait: is this minimal? Could we merge some v-cliques with K?
        # No, because a clique can contain at most one independent vertex.
        # So each independent vertex with edges needs its own clique.
        # The K clique is needed for K-K edges unless we can cover them via v-cliques.
        # Let's check if we can avoid the K clique.

        # Alternative: don't add K clique, instead ensure K-K edges are covered by v-cliques.
        # For this, for every edge (u, u') in K, we need some v with u, u' in neighbors(v).
        # If that's true, we can skip the K clique and save 1.

        # Let's try both and pick the better.
        # Option 1: with K clique
        count1 = len(cliques)

        # Option 2: without K clique, check if K-K edges are covered
        # Build coverage of K-K edges by v-cliques
        kk_edges = set()
        for i in range(k):
            for j in range(i+1, k):
                kk_edges.add((i, j))

        covered_kk = set()
        for v in I:
            nbrs = cross[v]
            for i in range(len(nbrs)):
                for j in range(i+1, len(nbrs)):
                    a, b = nbrs[i], nbrs[j]
                    covered_kk.add((min(a,b), max(a,b)))

        if kk_edges.issubset(covered_kk):
            # All K-K edges are covered by v-cliques, so we don't need the K clique
            cliques2 = []
            for v in I:
                nbrs = cross[v]
                if len(nbrs) >= 1:
                    clique = [v] + sorted(nbrs)
                    if len(clique) >= 2:
                        cliques2.append(clique)
            # Check: are there any K-K edges NOT covered? We verified they are.
            # But also: are there independent vertices with no neighbors? They don't form cliques.
            # And K vertices with no cross edges and no K-K edges covered? 
            # Actually K-K edges are all covered by v-cliques.
            # But what if some K vertex has no cross edges at all? Then it only appears in K-K edges,
            # which are covered by other v-cliques. That's fine.
            count2 = len(cliques2)
            if count2 < count1:
                cliques = cliques2

        # Final verification
        # Check each clique is a valid clique
        for c in cliques:
            for i in range(len(c)):
                for j in range(i+1, len(c)):
                    if c[j] not in adj[c[i]]:
                        raise ValueError(f"Not a clique: {c}")

        # Check every edge is in exactly one clique
        edge_count_in_cliques = {}
        for c in cliques:
            for i in range(len(c)):
                for j in range(i+1, len(c)):
                    a, b = c[i], c[j]
                    e = (min(a,b), max(a,b))
                    edge_count_in_cliques[e] = edge_count_in_cliques.get(e, 0) + 1

        for e in edges:
            if edge_count_in_cliques.get(e, 0) != 1:
                raise ValueError(f"Edge {e} not in exactly one clique")

        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        cliques = solve(*spec)
        # Convert to lists of ints
        results.append([[int(v) for v in c] for c in cliques])
    return results
# EVOLVE-BLOCK-END
