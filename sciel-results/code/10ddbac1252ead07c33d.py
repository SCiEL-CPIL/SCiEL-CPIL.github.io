"""Construct a C5-free graph with no independent 9-set, maximizing n."""

def _has_c5(n, adj):
    """Check if graph has a C5 (5-cycle as subgraph)."""
    # For small n, check all 5-tuples
    for a in range(n):
        adj_a = adj[a]
        if len(adj_a) < 2:
            continue
        for b in adj_a:
            if b <= a:
                continue
            adj_b = adj[b]
            for c in adj_b:
                if c <= b:
                    continue
                adj_c = adj[c]
                for d in adj_c:
                    if d <= c:
                        continue
                    adj_d = adj[d]
                    # Need edge d-a and a!=c, b!=d (already distinct since we enforce ordering a<b<c<d<a? No, C5 is a cycle)
                    # Actually for a C5 subgraph we need 5 distinct vertices forming a cycle.
                    # Let's just check if a is in adj_d and a < b < c < d < a doesn't work for cycle.
                    # Simpler: check if there's a 5-cycle by checking specific patterns.
                    # For n up to 35, this brute force might be slow. Let's use a smarter approach.
                    if a in adj_d and c not in adj_d and d not in adj_a:
                        # Not quite right. Let's do it properly.
                        pass
    # Better approach for n<=35: enumerate all 5-subsets and check if they contain a C5
    # But that's C(35,5) = 324632 subsets, each check is cheap.
    for a in range(n):
        for b in range(a+1, n):
            if b not in adj[a]:
                continue
            for c in range(b+1, n):
                if c not in adj[b]:
                    continue
                for d in range(c+1, n):
                    if d not in adj[c]:
                        continue
                    # Need a 5th vertex e such that e-d, e-a edges exist, and no other constraints for subgraph C5
                    # Actually a C5 subgraph just needs 5 vertices with at least a 5-cycle.
                    # Let's check if there exists e > d such that e in adj[d] and e in adj[a]
                    # Wait, that only checks for C5s with vertices in order a,b,c,d,e in the cycle?
                    # No. A C5 can be in any order. 
                    # For small n, let's just check all 5-tuples for being a C5.
                    pass
    # Simpler: for each 5-subset, check if it contains a C5
    # A 5-set contains a C5 iff it has at least 5 edges forming a cycle of length 5.
    # Actually, just check if the induced subgraph on 5 vertices contains a 5-cycle.
    # This is equivalent to: the 5 vertices can be ordered v1-v2-v3-v4-v5-v1 with all edges present.
    # For n up to 35, iterating all 5-subsets and checking for C5 is feasible.
    
    import itertools
    for subset in itertools.combinations(range(n), 5):
        # Check if this 5-set contains a C5
        # Try all 12 distinct cycles on 5 labeled vertices
        # Actually, just check if there's a permutation that forms a cycle
        # Simpler: check if the induced subgraph has a cycle of length 5
        # A 5-vertex graph has a C5 iff it has at least 5 edges and contains a 5-cycle.
        # Let's just check all 5! / 10 = 12 possible cycle orderings? No, there are 12 distinct 5-cycles on 5 labeled vertices.
        # Actually, we can just check if there exist 5 edges forming a cycle.
        # For a 5-set {v0,v1,v2,v3,v4}, check if there's a Hamiltonian cycle.
        vs = subset
        # Check all 5! permutations? No, 5-cycles: fix v0, permute the rest: 4! = 24, but each cycle counted twice (forward/backward), so 12.
        # Let's just try: is there a sequence v0-v1-v2-v3-v4-v0 where all edges exist?
        # We can iterate over all permutations of vs[1:]
        found = False
        for perm in itertools.permutations(vs[1:]):
            seq = (vs[0],) + perm
            ok = True
            for i in range(5):
                if seq[(i+1) % 5] not in adj[seq[i]]:
                    ok = False
                    break
            if ok:
                return True
    return False


def _max_independent_set_size(n, adj):
    """Find size of maximum independent set. For n<=35, use Bron-Kerbosch on complement."""
    # Complement graph: edge in complement means NOT edge in original
    comp_adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if j not in adj[i]:
                comp_adj[i].add(j)
                comp_adj[j].add(i)
    
    # Find max clique in complement = max independent set in original
    best = [0]
    
    def bron_kerbosch(R, P, X):
        if not P and not X:
            if len(R) > best[0]:
                best[0] = len(R)
            return
        # Pivotal selection
        if P:
            # Choose pivot from P ∪ X to maximize |N(u) ∩ P|
            pivot = max(P | X, key=lambda u: len(comp_adj[u] & P))
            candidates = P - comp_adj[pivot]
        else:
            candidates = P
        
        for v in candidates:
            R.add(v)
            bron_kerbosch(R, P & comp_adj[v], X & comp_adj[v])
            R.discard(v)
            P.discard(v)
            X.add(v)
    
    bron_kerbosch(set(), set(range(n)), set())
    return best[0]


def construct():
    import time
    start = time.time()
    deadline = start + 140  # Safety margin
    
    # Start with 8 disjoint K4s: n=32, alpha=8, C5-free
    n = 32
    edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    
    best_n = n
    best_edges = edges
    
    # Try to add vertices by modifying the structure
    # Key insight: we can't just add isolated vertices (would increase alpha).
    # We need to add vertices that are well-connected to keep alpha <= 8.
    
    # Try: replace some K4s with larger C5-free components with better n/alpha ratio.
    # Known C5-free graphs:
    # K4: n=4, alpha=1, ratio 4
    # K4 minus edge: n=4, alpha=2, ratio 2 (worse)
    # What about K5 minus a matching? n=5, alpha=2, ratio 2.5 (worse than K4)
    # What about the 5-vertex graph with edges forming a "house"? No, that has C4.
    # Actually, for maximizing n with alpha <= 8, K4 components are optimal per-vertex.
    # 8 K4s gives n=32, alpha=8.
    # Can we do n=33? We'd need alpha=8 still.
    # That means we need a C5-free graph on 33 vertices with alpha=8.
    # This is equivalent to: complement is triangle-free on 33 vertices with clique number <= 8? No.
    # Complement: C5-free in G means complement has no "C5-complement" which is complex.
    # Alpha <= 8 in G means omega(complement) <= 8.
    
    # Let's try a randomized search: start from 8 K4s, try adding a vertex with random edges,
    # check if C5-free and alpha <= 8.
    
    import random
    random.seed(42)
    
    def build_adj(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    adj = build_adj(n, edges)
    
    # Try adding vertices one by one
    # For each candidate new vertex, try different connection patterns
    # The new vertex must not create a C5 and must not increase alpha beyond 8.
    
    for attempt in range(5000):
        if time.time() > deadline:
            break
        
        # Try adding a vertex connected to a subset of existing vertices
        # The new vertex v must be connected to enough vertices to keep alpha <= 8.
        # If we add v, alpha can increase by at most 1 (if v joins the max independent set).
        # So we need alpha(G + v) <= 8, meaning v must NOT be able to join any independent set of size 8.
        # Equivalently, N(v) must intersect every independent set of size 8.
        
        # Also, adding v must not create a C5.
        # v creates a C5 if there exist u1, u2 in N(v) such that u1-u2 edge exists and there's a path of length 3 from u1 to u2 avoiding v, 
        # plus u1 and u2 not adjacent to each other? No.
        # v is in a C5 iff there exist two neighbors u, w of v with dist(u, w) = 3 in G-v, and u not adjacent to w.
        # Or dist(u,w) = 2 and... no, that would make a 4-cycle with v.
        # For C5 containing v: v-a-b-c-d-v. So a,d in N(v), and there's a path a-b-c-d of length 3, with a not adjacent to d.
        
        # Let's try: connect new vertex to a random subset of size k
        # To keep alpha <= 8, v must be "hitting" all max independent sets.
        # In 8 K4s, max independent sets choose 1 vertex from each K4.
        # If v connects to at least one vertex in each K4, then any independent set containing v can have at most 7 other vertices (one from each K4 except the ones v connects to... no).
        # If v connects to vertex x_i in K4_i, then an independent set containing v cannot contain x_i, but can contain the other 3 vertices of K4_i? No, independent set picks at most 1 per K4.
        # So if v connects to at least one vertex in each K4, an independent set containing v picks v and then 0 from some K4s and 1 from others. Max size = 1 + (number of K4s where v doesn't connect to the chosen vertex).
        # To ensure alpha <= 8, we need: for every independent set S of G of size 8, v is not independent from S.
        # I.e., N(v) intersects S.
        
        # Simplest: connect v to one vertex in each K4. Then any independent set of size 8 (which picks 1 from each K4) must intersect N(v) if v connects to at least one vertex in the chosen set.
        # But if v connects to vertex x_i in K4_i, and the independent set picks a different vertex y_i != x_i from K4_i, then v is not adjacent to y_i.
        # So v can be added to an independent set that avoids all x_i's.
        # The independent set picks y_i from each K4_i where y_i != x_i. There are 3 choices per K4, so 3^8 such sets.
        # So alpha would be 9.
        
        # To prevent this, v must connect to ALL vertices in at least one K4? No, that's not enough.
        # Actually, if v connects to all 4 vertices of K4_0, then any independent set can contain at most 1 vertex from K4_0, and v is adjacent to all of them, so v cannot be in an independent set with any vertex from K4_0.
        # But v can still be in an independent set with 1 vertex from each of K4_1..K4_7, giving size 8.
        # So alpha stays 8 if v connects to all vertices of at least one K4? No, alpha could still be 8 (v + 1 from each of 7 other K4s).
        # But we need alpha <= 8, so this is fine!
        
        # Wait: if v connects to all 4 vertices of K4_0, then max independent set containing v has size 1 + 7 = 8 (v + 1 from each K4_i, i=1..7).
        # Max independent set not containing v has size 8 (1 from each K4).
        # So alpha = 8. 
        
        # But does adding v create a C5?
        # v is connected to all 4 vertices of K4_0. Within K4_0, there are no C4s (K4 is complete, so any 4 vertices form K4, no C4 as a cycle since all edges present, but C4 is a subgraph? A C4 subgraph requires 4 vertices with exactly 4 edges in a cycle. K4 has 6 edges, so it contains C4s as subgraphs! Wait, does K4 contain a C5? No. Does adding v connected to all K4_0 vertices create a C5?
        # v-a-b-c-d-v: a,b,c,d from K4_0? K4_0 has only 4 vertices, so a,b,c,d would be all 4 vertices of K4_0. v-a, v-d edges exist. a-b, b-c, c-d edges exist. Is a-d an edge? In K4, yes, but we don't need it. The cycle is v-a-b-c-d-v. Edges: v-a, a-b, b-c, c-d, d-v. All present! So this is a C5!
        
        # So connecting v to all vertices of a K4 creates a C5.
        
        # What if v connects to exactly 3 vertices of K4_0? Then v + 3 vertices from K4_0: can we form a C5?
        # v-a-b-c-d-v: need 4 vertices from K4_0. But v only connects to 3, so d must be one of the 3, but then we need 4 distinct vertices from K4_0 in the cycle, impossible since only 3 are connected to v.
        # So no C5 through v using K4_0.
        # But v might form C5 with vertices from other K4s? No, because other K4s are disconnected from K4_0 (in the original graph), so no paths between them.
        
        # So if v connects to exactly 3 vertices of one K4 and 0 to others, no C5.
        # But alpha: v can be in an independent set with 1 vertex from each other K4 (7 vertices) + 1 vertex from K4_0 that is NOT connected to v. So size 1+7+1 = 9. Bad.
        
        # We need a different approach. The 8 K4s might be optimal.
        # Let's just return the 32-vertex solution.
        
        pass
    
    return {'n': best_n, 'edges': best_edges}
