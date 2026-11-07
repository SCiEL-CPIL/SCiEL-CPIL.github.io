"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    import itertools

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def find_clique_partition(n, k, p, seed):
        edges = instance(n, k, p, seed)
        edge_set = set(edges)
        
        # Build adjacency matrix
        adj = [[False] * n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
        
        # Precompute all maximal cliques or just use a greedy approach with backtracking
        # Since n is small (30, 60, 90), but k is 10, 20, 30.
        # The clique on the first k vertices is a large clique.
        # The rest are independent.
        # Cross edges are random.
        
        # A good heuristic for split graphs:
        # 1. The set of the first k vertices is a clique. Call it C0.
        # 2. Every edge in C0 is covered by C0.
        # 3. Remaining edges are cross edges (u, v) where u < k and v >= k.
        # 4. We need to cover all cross edges with cliques.
        #    A clique containing a cross edge (u, v) can include:
        #    - u and v
        #    - other u' < k such that u' is adjacent to v
        #    - other v' >= k such that v' is adjacent to u
        #    Note: v' >= k is independent, so no two v's can be in the same clique unless they are both adjacent to u?
        #    Wait, if v1, v2 >= k, they are not adjacent. So a clique can contain at most one vertex from the independent set.
        #    Therefore, any clique that covers cross edges must contain exactly one vertex from the independent set (say v) and a subset of the clique vertices (u1, u2, ...) that are all adjacent to v.
        #    Also, the clique vertices in the clique must be mutually adjacent (they are, since they are in the first k).
        #    So, for each v >= k, the neighbors in the clique part form a clique N(v).
        #    Any subset of N(v) union {v} is a clique.
        #    To cover all cross edges efficiently, we should group edges by their independent set vertex v.
        #    For a fixed v, all edges (u, v) for u in N(v) must be covered.
        #    Since v can only appear in cliques with other u's, and all u's in N(v) are mutually adjacent,
        #    we can cover all edges incident to v with a single clique: N(v) union {v}.
        #    Is this optimal?
        #    If we use a smaller clique, say {u1, v}, it covers only (u1, v). We would need another clique for (u2, v).
        #    Since u1, u2 are adjacent, {u1, u2, v} is a clique.
        #    So it is always better to include all neighbors of v in the clique with v.
        #    Thus, for each v >= k, the clique N(v) U {v} covers all edges incident to v.
        #    Does this cover all cross edges? Yes, each cross edge (u, v) is incident to v.
        #    Are these cliques disjoint in edges?
        #    Edge (u, v) is only in the clique for v.
        #    Edges within the first k are in C0.
        #    So the partition is:
        #    - C0 = [0, ..., k-1]
        #    - For each v in [k, ..., n-1], if N(v) is not empty, Cv = N(v) U {v}
        #    Wait, what if N(v) is empty? Then there are no edges incident to v, so no clique needed for v.
        #    What if N(v) has size 1? Say {u}. Then Cv = {u, v}. This is a valid clique (size 2).
        #    
        #    Let's check the number of cliques.
        #    1 (for C0) + number of v >= k with deg(v) > 0.
        #    Since p=0.7, most v will have neighbors.
        #    For n=90, k=30, there are 60 vertices in the independent set.
        #    Expected number of cliques ~ 1 + 60 = 61.
        #    
        #    Can we do better?
        #    The Erdős-Ordman-Zalcstein bound is n^2/6 + O(n).
        #    For n=90, 90^2/6 = 1350. Our heuristic gives ~61. This is much smaller.
        #    The question is whether we can merge cliques for different v's.
        #    No, because v1 and v2 are not adjacent, so they cannot be in the same clique.
        #    Thus, each clique can contain at most one vertex from the independent set.
        #    Therefore, we need at least one clique for each v that has any edges.
        #    So the minimum number of cliques is 1 + |{v >= k : deg(v) > 0}|.
        #    This is achieved by the construction above.
        
        cliques = []
        # Clique 0: the first k vertices
        if k >= 2:
            cliques.append(list(range(k)))
        
        # For each v in independent set
        for v in range(k, n):
            neighbors = [u for u in range(k) if adj[u][v]]
            if len(neighbors) > 0:
                cliques.append(neighbors + [v])
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        cliques = find_clique_partition(*spec)
        # Validate types
        valid_cliques = []
        for c in cliques:
            valid_cliques.append([int(x) for x in c])
        results.append(valid_cliques)
    
    return results
# EVOLVE-BLOCK-END
