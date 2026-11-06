import time
import numpy as np
from itertools import combinations

def _has_c5(adj: list[set]) -> bool:
    """Exact check for a C5 subgraph. Returns True if any 5-cycle exists."""
    n = len(adj)
    # For speed, iterate over edges (a,b), then common neighbors approach is too broad.
    # Use a smarter enumeration: for each pair of vertices at distance 2, we can't just
    # count common neighbors. Instead, enumerate paths of length 3: a - b - c - d,
    # then check if d is adjacent to a (making a 5-cycle: a-b-c-d-a, provided all distinct).
    for a in range(n):
        for b in adj[a]:
            if b <= a:
                continue
            for c in adj[b]:
                if c == a or c <= b:
                    continue
                for d in adj[c]:
                    if d in (a, b, c):
                        continue
                    if d <= a:
                        continue
                    if d in adj[a]:
                        # Then a-b-c-d-a is a 5-cycle? Wait, need to verify a-d is an edge
                        # AND all 5 vertices distinct. Yes they are: a < d, a-b, b-c (b<c), c-d (c!=d)
                        # But we need a-b-c-d and d-a to form a cycle of 5 distinct vertices.
                        # Actually, the path a-b-c-d plus edge d-a is a 4-cycle if b and c were connected to both,
                        # but they're distinct. Let me re-check: vertices a, b, c, d — that's only 4 vertices.
                        # A 5-cycle needs 5 vertices: v0, v1, v2, v3, v4 with v4-v0 edge.
                        # So path of length 4: v0-v1-v2-v3-v4, and v4 adjacent to v0.
                        pass
    # Let's do it properly: enumerate 5-tuples (v0, v1, v2, v3, v4) with strict ordering
    # v0 < v1 < v2 < v3 < v4, then check edges v0-v1, v1-v2, v2-v3, v3-v4, v4-v0.
    # This is O(n^5) worst case but with adjacency checks it's manageable for n <= 35.
    for v0 in range(n):
        for v1 in adj[v0]:
            if v1 <= v0:
                continue
            for v2 in adj[v1]:
                if v2 <= v1 or v2 == v0:
                    continue
            # Wait, that's still not efficient. Let me use a different approach.
    # Use: for each edge (v0, v1), for each neighbor v2 of v1 (v2 != v0), for each
    # neighbor v3 of v2 (v3 != v0, v3 != v1), for each neighbor v4 of v3 (v4 != v0,
    # v4 != v1, v4 != v2), check if v4 is adjacent to v0.
    for v0 in range(n):
        n0 = adj[v0]
        for v1 in n0:
            n1 = adj[v1] - {v0}
            for v2 in n1:
                n2 = adj[v2] - {v0, v1}
                for v3 in n2:
                    n3 = adj[v3] - {v0, v1, v2}
                    for v4 in n3:
                        if v4 in n0:
                            return True
    return False

def _independent_set_size(adj: list[set], target: int) -> bool:
    """Check if there exists an independent set of size >= target using branch and bound."""
    n = len(adj)
    # Build complement
    comp = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if j not in adj[i]:
                comp[i].add(j)
                comp[j].add(i)
    
    found = [False]
    
    def search(clique_sz, candidates, depth):
        if found[0]:
            return
        if clique_sz >= target:
            found[0] = True
            return
        if not candidates:
            return
        if clique_sz + len(candidates) < target:
            return
        # Heuristic: pick vertex with max degree in candidates
        v = max(candidates, key=lambda x: len(comp[x] & candidates))
        # Include v
        nc = candidates & comp[v]
        search(clique_sz + 1, nc, depth + 1)
        if found[0]:
            return
        # Exclude v
        c2 = candidates - {v}
        search(clique_sz, c2, depth)
    
    search(0, frozenset(range(n)), 0)
    return found[0]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin
    
    n = 32
    edges = []
    for b in range(8):
        base = b * 4
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    best_n = n
    best_edges = edges[:]
    
    # Verify initial graph
    assert not _has_c5(adj), "Initial K4x8 graph has a C5"
    assert not _independent_set_size(adj, 9), "Initial K4x8 graph has alpha >= 9"
    
    # Use bitmasks for speed
    def to_mask(adj_sets):
        masks = [0] * len(adj_sets)
        for i, s in enumerate(adj_sets):
            for j in s:
                masks[i] |= (1 << j)
        return masks
    
    # Try to add vertex 32 with various connection patterns
    # The new vertex connects to some subset of {0..31}. We need:
    # 1. No C5 in the augmented graph
    # 2. No independent set of size 9
    
    # For speed, use a randomized local search approach:
    # Start from the base graph and try adding vertex 32 with random connection sets.
    
    # Connection set for vertex 32: subset of {0,...,31}
    # If vertex 32 connects to a set S, then:
    # - C5 check: any 5-cycle involving vertex 32 must use edges 32-u and 32-v for u,v in S,
    #   plus a path of length 3 from u to v in the original graph (not through 32).
    # - Alpha check: an independent set of size 9 either includes vertex 32 (then the other
    #   8 must be independent in the original graph AND none of them in S), or doesn't
    #   include vertex 32 (then it's an independent set of size 9 in the original graph,
    #   which we know doesn't exist since alpha(original) = 8).
    
    # So the alpha constraint simplifies: if 32 is in an independent set of size 9,
    # the other 8 vertices must form an independent set in the original graph that's
    # disjoint from S. Since max independent set in original is 8, we need to check
    # if there's an independent set of size 8 in the original graph disjoint from S.
    
    # C5 constraint: if 32 is in a C5, say 32-u-v-w-x-32, then u, x in S, and there's
    # a path u-v-w-x in the original graph (length 3, no 32). So we need to check if
    # there exist u, x in S such that there's a path of length 3 from u to x in the
    # original graph. This is equivalent to: u and x are at distance exactly 3 in the
    # original graph? No, the path can be any length-3 path, meaning d(u,x) <= 3 in
    # the original graph (and the intermediate vertices are distinct and not 32).
    # Actually, we need a path of length 3: u - v - w - x where all vertices distinct
    # and not 32. This means dist(u, x) <= 3 in the original graph, but we also need
    # the path to not go through any vertex that would create a shorter cycle.
    # Simplification: if u and x are adjacent in the original graph, then 32-u-x-32
    # is a triangle, not a C5. If dist(u,x) = 2, then 32-u-v-x-32 is a 4-cycle.
    # For a C5, we need dist(u, x) = 3 in the original graph (via a specific path).
    # More precisely: there exists a vertex pair (u, x) both in S such that there's
    # a path u-a-b-x in the original graph with {u, a, b, x} all distinct.
    
    # Let's precompute which pairs (u, x) have a path of length 3 in the original graph.
    dist3_pairs = set()
    for u in range(n):
        for x in adj[u]:  # u adjacent to x? No, we need path of length 3: u-a-b-x
            pass
    
    # Better: for each pair (u, x), check if there's a path of length 3.
    # A path of length 3 from u to x means: exists a, b such that
    # u-a, a-b, b-x are edges, all distinct.
    # Equivalently, x is in the 3-neighborhood of u.
    # Precompute: for each u, compute N3(u) = set of vertices reachable from u by a path of length 3.
    
    N3 = [set() for _ in range(n)]
    for u in range(n):
        for a in adj[u]:
            for b in adj[a] - {u}:
                for x in adj[b] - {u, a}:
                    N3[u].add(x)
    
    # Now, the C5 constraint for vertex 32 connecting to S:
    # There exists u, x in S such that x in N3[u] (and u != x, which is guaranteed).
    # This means S must NOT contain any pair (u, x) where x in N3[u].
    # Equivalently, S must be a "3-independent" set: no two vertices in S are at distance
    # 3 (via a length-3 path) in the original graph.
    
    # And the alpha constraint: there's no independent set of size 8 in the original
    # graph that's disjoint from S.
    
    # Precompute all maximum independent sets of size 8 in the original graph.
    # Since alpha = 8, we can enumerate them.
    
    # For efficiency, let's just do a random search over connection sets S.
    # A connection set S is valid if:
    # 1. No pair (u, x) in S has x in N3[u] (C5 constraint)
    # 2. No independent set of size 8 in original graph is disjoint from S (alpha constraint)
    
    # Random search:
    import random
    random.seed(42)
    
    # Precompute a list of "dangerous" pairs for C5
    danger_pairs = set()
    for u in range(n):
        for x in N3[u]:
            if u < x:
                danger_pairs.add((u, x))
    
    def is_c5_safe(S):
        """Check if S has no C5-creating pairs."""
        s_list = list(S)
        for i, u in enumerate(s_list):
            for x in s_list[i + 1:]:
                if (u, x) in danger_pairs or (x, u) in danger_pairs:
                    return False
        return True
    
    def is_alpha_safe(S):
        """Check if no independent set of size 8 is disjoint from S."""
        # Find if there's an independent set of size 8 in the subgraph induced by
        # {0, ..., 31} \ S
        remaining = set(range(n)) - S
        if len(remaining) < 8:
            return True
        
        # Check if the subgraph induced by 'remaining' has an independent set of size 8
        # (which means it has a clique of size 8 in the complement restricted to 'remaining')
        rem_comp = [set() for _ in range(n)]
        for i in remaining:
            for j in remaining:
                if i != j and j not in adj[i]:
                    rem_comp[i].add(j)
        
        # Branch and bound for clique of size 8 in rem_comp induced on 'remaining'
        found8 = [False]
        
        def search_clique(sz, candidates):
            if found8[0]:
                return
            if sz >= 8:
                found8[0] = True
                return
            if not candidates:
                return
            if sz + len(candidates) < 8:
                return
            v = max(candidates, key=lambda x: len(rem_comp[x] & candidates))
            nc = candidates & rem_comp[v]
            search_clique(sz + 1, nc)
            if found8[0]:
                return
            c2 = candidates - {v}
            search_clique(sz, c2)
        
        search_clique(0, frozenset(remaining))
        return not found8[0]
    
    best_S = None
    for iteration in range(50000):
        if time.time() > deadline - 5:
            break
        
        # Generate random connection set S
        # Vertex 32 needs to connect to at least some vertices to avoid being isolated
        # (which would increase alpha). But connecting to too many might create C5s.
        S = set()
        for b in range(8):
            base = b * 4
            # Connect to 1-3 random vertices from each K4
            k = random.randint(1, 3)
            chosen = random.sample(range(4), k)
            for c in chosen:
                S.add(base + c)
        
        if is_c5_safe(S) and is_alpha_safe(S):
            # Verify with exact checks
            new_adj = [set(s) for s in adj] + [set(S)]
            for u in S:
                new_adj[u].add(32)
            new_adj[32] = set(S)
            
            if not _has_c5(new_adj) and not _independent_set_size(new_adj, 9):
                new_edges = edges[:] + [[32, u] for u in sorted(S)]
                best_S = S
                best_edges = new_edges
                best_n = 33
                break
    
    return {'n': best_n, 'edges': best_edges}
