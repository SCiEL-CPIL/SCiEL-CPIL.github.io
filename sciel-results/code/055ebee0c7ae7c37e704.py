import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    def time_left():
        return deadline - time.time()

    # Base graph: 8 disjoint K4s -> n=32, alpha=8, C5-free
    n = 32
    edges = []
    for b in range(8):
        base = b * 4
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])

    def build_adj(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def has_c5(adj, n):
        """Check if graph has a 5-cycle as a subgraph."""
        for v0 in range(n):
            n0 = adj[v0]
            for v1 in n0:
                if v1 <= v0:
                    continue
                n1 = adj[v1] - {v0}
                for v2 in n1:
                    if v2 <= v1:
                        continue
                    n2 = adj[v2] - {v0, v1}
                    for v3 in n2:
                        if v3 <= v2:
                            continue
                        n3 = adj[v3] - {v0, v1, v2}
                        # Need v4 adjacent to v3 and v0, with v4 > v3, v4 not in {v0,v1,v2,v3}
                        # v4 must be in adj[v3] & adj[v0]
                        candidates = n3 & n0
                        for v4 in candidates:
                            if v4 > v3:
                                return True
        return False

    def independence_number(adj, n):
        """Compute independence number via max clique in complement (branch and bound)."""
        comp_adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if v not in adj[u]:
                    comp_adj[u].add(v)
                    comp_adj[v].add(u)

        max_clique = [0]

        def expand(clique, candidates):
            if len(clique) + len(candidates) <= max_clique[0]:
                return
            if not candidates:
                if len(clique) > max_clique[0]:
                    max_clique[0] = len(clique)
                return
            # Order candidates by degree in candidates (descending) for better pruning
            v = max(candidates, key=lambda x: len(comp_adj[x] & candidates))
            # Include v
            new_candidates = candidates & comp_adj[v]
            expand(clique | {v}, new_candidates)
            # Exclude v
            candidates.discard(v)
            expand(clique, candidates)
            candidates.add(v)

        expand(set(), set(range(n)))
        return max_clique[0]

    def alpha_of_subset(adj, n, subset):
        """Independence number of the induced subgraph on subset."""
        if not subset:
            return 0
        s = set(subset)
        # Build local complement
        local_comp = {v: set() for v in s}
        for u in s:
            for v in s:
                if u != v and v not in adj[u]:
                    local_comp[u].add(v)
        max_c = [0]
        def expand(clique, candidates):
            if len(clique) + len(candidates) <= max_c[0]:
                return
            if not candidates:
                if len(clique) > max_c[0]:
                    max_c[0] = len(clique)
                return
            v = max(candidates, key=lambda x: len(local_comp[x] & candidates))
            new_cand = candidates & local_comp[v]
            expand(clique | {v}, new_cand)
            candidates.discard(v)
            expand(clique, candidates)
            candidates.add(v)
        expand(set(), set(s))
        return max_c[0]

    # Verify base graph
    adj = build_adj(n, edges)
    assert not has_c5(adj, n), "Base graph has C5"
    alpha = independence_number(adj, n)
    assert alpha < 9, f"Base graph alpha={alpha}"

    best_n = n
    best_edges = edges[:]
    best_adj = [set(s) for s in adj]

    # Try to add vertex 32
    # For C5-freedom: vertex 32 connected to set S.
    # A C5 through 32: 32-a-b-c-d-32 where a,d in S, edges a-b, b-c, c-d exist.
    # This is a path of length 3 from a to d in G[S] or through other vertices.
    # More precisely: need a,b,c,d distinct, a,d in S, a-b, b-c, c-d in E.
    # b,c can be anywhere (not necessarily in S).
    # So for each pair (a,d) in S with a!=d, check if there's a path of length 3 from a to d
    # where the intermediate vertices b,c are distinct from a,d,32.
    # Actually the path is a-b-c-d, so b is adj to a, c is adj to b and d.
    # b != a,d,32; c != a,b,d,32.

    def creates_c5(adj, n, new_v, S):
        """Check if adding new_v connected to S creates a C5."""
        S = set(S)
        for a in S:
            for d in S:
                if a == d:
                    continue
                # Need path a-b-c-d with b,c distinct from a,d,new_v
                for b in adj[a]:
                    if b == a or b == d or b == new_v:
                        continue
                    for c in adj[b]:
                        if c == a or c == b or c == d or c == new_v:
                            continue
                        if c in adj[d]:
                            return True
        return False

    # Try all subsets of size 0..32 for S (neighborhood of vertex 32)
    # To be efficient, we need alpha constraint:
    # alpha(G + v32) <= 8
    # Independent set containing 32: {32} union I where I is independent in G[V\S]
    # So 1 + alpha(G[V\S]) <= 8, i.e., alpha(G[V\S]) <= 7
    # Also alpha(G) <= 8 must hold (it does)
    # And independent sets not containing 32: alpha(G) <= 8 (ok)
    # So constraint: alpha(G[V\S]) <= 7

    # For C5: S must not contain pairs (a,d) with a path of length 3 between them.
    # Let's precompute which pairs (a,d) have a length-3 path between them.
    # pair_has_path3[a][d] = True if there exist b,c distinct from a,d with a-b, b-c, c-d edges.

    # Precompute for base graph
    pair_path3 = [[False]*n for _ in range(n)]
    for a in range(n):
        for d in range(n):
            if a == d:
                continue
            for b in adj[a]:
                if b == d:
                    continue
                for c in adj[b]:
                    if c == a or c == b or c == d:
                        continue
                    if c in adj[d]:
                        pair_path3[a][d] = True
                        break
                if pair_path3[a][d]:
                    break

    # Now S must be a set with no pair (a,d) in S where pair_path3[a][d] is True.
    # I.e., S must be an independent set in the "path3 graph" H where H has edge (a,d) if pair_path3[a][d].
    # Also need alpha(G[V\S]) <= 7.

    # Let's find valid S values.
    # The path3 graph H: edges between vertices that have a length-3 path between them in G.
    # In 8 disjoint K4s: within a K4, any two vertices are adjacent, so path of length 1 exists,
    # but we need length 3. In K4, a-b-c-d: a~b, b~c, c~d. Since K4 is complete, this works
    # for any distinct a,b,c,d. But K4 only has 4 vertices, so a,b,c,d must be all 4.
    # So within a K4, any pair (a,d) has a length-3 path (using the other 2 vertices).
    # Between different K4s: no edges, so no path of any length.
    # So H has edges only within each K4, and it's a K4 (complete) on each group of 4.
    # So S can contain at most 1 vertex from each K4 to avoid C5!
    # Wait: if S contains 2 vertices from the same K4, say a and d, then pair_path3[a][d] is True,
    # which would create a C5. So S can have at most 1 vertex from each K4.

    # So S is a set with at most 1 vertex from each of the 8 K4s.
    # |S| <= 8.
    # Constraint: alpha(G[V\S]) <= 7.
    # V\S has 32 - |S| vertices.
    # In G[V\S]: each K4 has either 4 or 3 vertices remaining (if S took 1 from it).
    # If S took 1 from K4_i, then K4_i \ S is K3 (alpha=1).
    # If S took 0 from K4_i, then K4_i is K4 (alpha=1).
    # So alpha(G[V\S]) = number of K4s = 8 (since each contributes alpha=1).
    # That's 8, not <= 7. So no valid S exists!

    # This confirms n=32 is the max for this construction.
    # Let's just return the base graph.

    return {'n': best_n, 'edges': best_edges}
