"""Clique partitions of three fixed split (chordal) graphs."""
# EVOLVE-BLOCK-START
import time

def construct():
    from random import Random

    t0 = time.time()
    DEADLINE = t0 + 110.0  # safety margin before 120s kill

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                edges.add((u, v))
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def partition_cliques(n, k, p, seed):
        edges = instance(n, k, p, seed)
        cliques = []
        # 1 clique covering all clique-clique edges
        cliques.append(list(range(k)))
        # Group independent-set vertices by their neighborhood in the clique part.
        # Two independent-set vertices v1, v2 can be in the same clique iff
        # N(v1) == N(v2) (they share exactly the same clique neighbors).
        # Then {v1, v2, u1, ..., um} where {u1..um} = N(v1) is a clique.
        # We need to cover all cross edges. Each cross edge (u, v) where u < k <= v
        # must be in exactly one clique.
        # Strategy: group by neighborhood. For each neighborhood S, collect all
        # independent-set vertices with that neighborhood. For each such group,
        # we need cliques that cover all edges between S and the group.
        # Since all vertices in the group have neighborhood S, the set S ∪ {group} 
        # forms a clique. But we need to cover edges exactly once.
        # Actually, if we put S ∪ {all group members} as one clique, that covers
        # all edges from S to each group member, plus edges within S (already covered
        # by the big clique) and edges within group (none, since they're independent).
        # But wait - edges within S are already in the big clique. We need each edge
        # in EXACTLY one clique. So we can't have S in two cliques.
        # Better: for each neighborhood S with group G, create clique S ∪ G.
        # But S appears in multiple such cliques if multiple groups share S... no,
        # each group has a distinct S. So each S appears in at most one clique.
        # But the big clique already contains all edges within S. So edges (u1, u2) 
        # for u1, u2 in S are in the big clique AND in S∪G. That violates "exactly one".
        # Fix: don't use the big clique. Instead, for each neighborhood S, create
        # clique S ∪ G. Edges within S are covered by whichever clique contains S.
        # But if S is used in one clique, edges within S are covered there.
        # Actually every pair in S must be covered. If S is in clique C = S ∪ G, 
        # then all pairs in S are in C. Good. But what about pairs in S that are 
        # also needed... no, each edge is in exactly one clique.
        # Let's just use: for each distinct neighborhood S (non-empty), clique = S ∪ G.
        # For empty neighborhood, no cross edges, skip.
        # The big clique is NOT used. Each edge (u,v) with u,v < k: u is in some 
        # neighborhood S that contains u? No - neighborhoods are subsets of [0,k).
        # Edge (u, u') where u, u' < k: this edge needs to be in a clique. It's in
        # clique S ∪ G where S contains both u and u'. But does every pair u, u' < k
        # appear together in some neighborhood S? Only if there exists an independent-set
        # vertex connected to both. Not guaranteed!
        # So we still need to cover clique-clique edges. Use the big clique for those.
        # But then edges within S appear in both big clique and S∪G. Violation.
        # Solution: For the big clique, only include edges not already covered.
        # Actually simpler: use big clique for ALL clique-clique edges. For cross edges,
        # use cliques that do NOT contain two clique vertices together... but a clique
        # S∪G contains pairs from S.
        # Alternative: for each neighborhood S with group G, create cliques of size 2:
        # {u, v} for each u in S, v in G. That's the original approach.
        # Better grouping: if G has multiple vertices, {u, g1, g2} is a clique if 
        # g1, g2 have same neighborhood containing u. Yes! So for each u in S,
        # {u} ∪ G is a clique. This covers all cross edges from u to G.
        # Total cliques: sum over all neighborhoods S of |S|. Which equals 
        # number of cross edges / |G| on average... no, it's sum of |S| over groups.
        # This is better than one clique per cross edge when |G| > 1.
        from collections import defaultdict
        # Compute neighborhood of each independent-set vertex
        neigh = {}
        for v in range(k, n):
            ns = frozenset(u for u in range(k) if (u, v) in edges or (v, u) in edges)
            neigh[v] = ns
        # Group by neighborhood
        groups = defaultdict(list)
        for v in range(k, n):
            groups[neigh[v]].append(v)
        # For each non-empty neighborhood, for each u in S, create clique {u} ∪ G
        for S, G in groups.items():
            if not S:
                continue
            for u in S:
                cliques.append([u] + G)
        return cliques

    result = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > DEADLINE:
            break
        cliques = partition_cliques(*spec)
        # Ensure every clique is a list of distinct Python ints, len >= 2
        cleaned = []
        for c in cliques:
            cl = [int(x) for x in c]
            if len(cl) >= 2 and len(set(cl)) == len(cl):
                cleaned.append(cl)
        result.append(cleaned)

    return result
# EVOLVE-BLOCK-END
