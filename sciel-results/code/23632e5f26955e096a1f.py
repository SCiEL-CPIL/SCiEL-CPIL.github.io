import time
import itertools
from collections import deque

def construct():
    start = time.time()
    DEADLINE = 135.0  # safety margin before 150s kill

    # Build the n=32 graph: 7 disjoint K4s + a K4 of new vertices,
    # each new vertex connected to one vertex in each K4.
    #
    # Vertices 0..27: 7 K4s (4 each)
    # Vertices 28..31: 4 new vertices forming a K4
    # Each new vertex w connects to the first vertex (index 0) of each K4
    #
    # C5-free proof: The only possible C5 would need to use vertices from
    # different K4s. Since each K4 is a clique, any path through a K4
    # using 2+ vertices of that K4 creates a triangle or uses only one vertex.
    # The new vertices form a K4, so they connect to each other.
    # A C5 would need 5 distinct vertices. The structure is:
    #   - Within a K4: clique, no C5
    #   - Among new vertices: K4, no C5
    #   - Mixed: A C5 would need to go through at least 2 different K4s and
    #     the new vertices. But each new vertex connects to only ONE vertex
    #     per K4, so any path from new vertices to a K4 uses exactly one
    #     vertex of that K4. Thus any cycle involving new vertices and
    #     K4 vertices can only use at most one vertex per K4. With 7 K4s
    #     and 4 new vertices, a cycle of length 5 would need to visit
    #     vertices from at most 5 distinct "groups" (K4s or the new K4).
    #     Since the new vertices form a K4 and each connects to one vertex
    #     per K4, any cycle must alternate between the new K4 and the K4s.
    #     But the new K4 has diameter 1, so going from one new vertex to
    #     another is one edge. A C5 would be: w1 - u_i - w2 - u_j - w3 - w1
    #     but w3-w1 is an edge, and we need 5 edges. Let's check:
    #     w1-u_i-w2: 2 edges, then w2-u_j: 1 edge, then need 2 more edges
    #     to close to w1. w3 connects to w1 and w2 (K4), so w3-w1 is 1 edge,
    #     but that's only 4 edges total (w1,u_i,w2,u_j,w3,w1) = 5 vertices,
    #     5 edges? w1-u_i, u_i-w2, w2-u_j, u_j-w3, w3-w1. Yes! That's a C5!
    #
    # Wait, I need to re-examine. Is u_i connected to w2? u_i is the first
    # vertex of K4_i. w2 is a new vertex. w2 connects to the first vertex of
    # each K4, so w2 connects to u_i (if i is the K4 index). So yes, w2-u_i
    # is an edge. Similarly u_j-w3 is an edge.
    #
    # So w1-u_i-w2-u_j-w3-w1 IS a 5-cycle if u_i != u_j (different K4s)
    # and w1, w2, w3 are distinct new vertices. This IS a C5!
    #
    # So my construction has C5s. I need a different approach.
    #
    # Let me reconsider. The known best is 32 vertices. Let me think of
    # a valid construction.
    #
    # Alternative: Use the complement approach. We need a C5-free graph with
    # alpha <= 8. Equivalently, the complement has no independent set of size
    # 5 (i.e., no K5 in the complement... wait, no).
    #
    # Actually, let me think differently. A well-known construction for
    # R(C5, K9) > 32 uses a specific structure.
    #
    # Let me try: Take 8 copies of K4, giving 32 vertices. The complement
    # of a K4 is 4 independent vertices. So the complement of 8 disjoint K4s
    # is a complete 8-partite graph with parts of size 4. The independence
    # number of the original graph = clique number of the complement = 8
    # (one vertex from each part). Wait, no: alpha of G = omega of G-bar.
    # G = 8 disjoint K4s. G-bar = complete 8-partite K_{4,4,4,4,4,4,4,4}.
    # omega(G-bar) = 8 (pick one from each part). So alpha(G) = 8. Good.
    #
    # Now is G C5-free? G = 8 disjoint K4s. A C5 must use vertices from
    # at least 2 different K4s (since a K4 has no C5). But within a K4,
    # all pairs are connected. So if a C5 uses vertices from K4_a and K4_b,
    # say vertices a1, a2 from K4_a and b1, b2 from K4_b, plus one more.
    # The edges a1-a2 exist (same K4), b1-b2 exist (same K4). Edges between
    # different K4s don't exist. So a1 can only connect to a2 and other
    # vertices in K4_a. Thus any cycle is entirely within one K4, but K4
    # has no C5. So G is C5-free!
    #
    # Wait, but can we have a cycle that uses vertices from multiple K4s?
    # No, because there are no edges between different K4s. So every
    # connected component is a K4, and each K4 has no C5. Therefore G
    # is C5-free.
    #
    # So 8 disjoint K4s on 32 vertices: C5-free, alpha = 8. 
    #
    # Let me verify alpha more carefully. An independent set in G means a set
    # of vertices with no two adjacent. Since each K4 is a clique, we can
    # pick at most 1 vertex from each K4. With 8 K4s, max independent set
    # size is 8. And we can achieve 8 by picking one from each. So alpha = 8 < 9. 
    #
    # This is a valid construction with n=32!

    # Construct 8 disjoint K4s
    n = 32
    edges = []
    for block in range(8):
        base = block * 4
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])

    # Validate
    def validate(n, edges):
        if time.time() > DEADLINE:
            return False
        adj = [[False] * n for _ in range(n)]
        for u, v in edges:
            if u == v or u < 0 or u >= n or v < 0 or v >= n:
                return False
            adj[u][v] = True
            adj[v][u] = True

        # Check C5-free: for n=32, use the component structure
        # Each component is a K4, which is C5-free
        # But let's do a general check for safety
        # For small n, check all 5-subsets
        vertices = list(range(n))
        for combo in itertools.combinations(vertices, 5):
            a, b, c, d, e = combo
            found = False
            for perm in itertools.permutations([b, c, d, e]):
                x, y, z, w = perm
                if adj[a][x] and adj[x][y] and adj[y][z] and adj[z][w] and adj[w][a]:
                    found = True
                    break
                if time.time() > DEADLINE:
                    return False
            if found:
                return False
            if time.time() > DEADLINE:
                return False

        # Check no independent set of size 9
        # For this specific graph, we know alpha=8, but let's verify
        # Since each K4 is a clique, independent set has at most 1 per K4
        # With 8 K4s, alpha <= 8
        # Quick verification: check if any 9-subset is independent
        # C(32,9) is large, but we can reason: any 9 vertices must include
        # 2 from some K4 (pigeonhole), and those 2 are adjacent. So no
        # independent set of size 9. This is airtight.
        return True

    if validate(n, edges):
        return {'n': n, 'edges': edges}

    # Fallback: 7 disjoint K4s (n=28)
    n_base = 28
    edges_base = [[4 * b + i, 4 * b + j] for b in range(7) for i in range(4) for j in range(i + 1, 4)]
    return {'n': n_base, 'edges': edges_base}
