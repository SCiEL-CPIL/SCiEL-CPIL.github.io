"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random

# EVOLVE-BLOCK-START

def _build_8k4():
    """Build 8 disjoint K4s, return (n, adj_set, edge_list)."""
    blocks = 8
    size = 4
    n = blocks * size
    adj = set()
    edges = []
    for b in range(blocks):
        base = b * size
        for i in range(size):
            for j in range(i + 1, size):
                u, v = base + i, base + j
                adj.add((min(u,v), max(u,v)))
                edges.append([u, v])
    return n, adj, edges

def _has_c5(adj, n):
    """Check if graph with adjacency set has a 5-cycle using BFS from each vertex."""
    # For each vertex, do BFS up to depth 4, check if we can form a 5-cycle
    # A 5-cycle through vertex v: v-a-b-c-d-v where all distinct
    # More efficient: check if any vertex has a path of length 4 back to itself
    # with all 5 vertices distinct.
    
    # Use the fact that in a C5-free graph, for every edge (u,v), 
    # N(u)\{v} and N(v)\{u} must be disjoint (no common neighbor creates triangle, 
    # but we need to check 5-cycles specifically)
    
    # Actually, a 5-cycle exists iff there exist vertices v0,v1,v2,v3,v4 all distinct
    # with edges (v0,v1),(v1,v2),(v2,v3),(v3,v4),(v4,v0).
    
    # Efficient check: for each pair of vertices at distance 2, check if they share
    # a common neighbor at distance 2 that completes a 5-cycle.
    
    # Simpler approach: enumerate all 5-cycles by checking paths of length 4.
    # For n <= 35, we can be clever.
    
    # Key insight: A 5-cycle v0-v1-v2-v3-v4-v0 exists iff there is a path
    # v0-v1-v2-v3-v4 where v4 is adjacent to v0 and all 5 are distinct.
    
    # For efficiency with n<=35, check: for each edge (a,b), for each c in N(a)\{b},
    # for each d in N(c)\{a}, for each e in N(d)\{c}, check if e==b or (e in N(a) and e!=a)
    # Wait, that's getting complicated. Let me use a direct but bounded approach.
    
    # For n <= 35, we can check all 5-tuples but that's O(n^5) = ~50M which is too slow.
    
    # Better: For each vertex v, look at its neighbors. A 5-cycle through v means
    # v-a-b-c-d-v. So we need a path of length 3 from a to d (a-b-c-d) where
    # a,d are both neighbors of v, a != d, and all distinct.
    
    # For each vertex v:
    #   For each pair (a,d) in N(v) with a < d:
    #     Check if there exists a path a-b-c-d of length 3 with b,c distinct from v,a,d
    #     i.e., b in N(a)\{v}, c in N(b)\{a}, d in N(c)\{b} and d is the target.
    
    # Even better: for each v, for each a in N(v), for each d in N(v) with d > a:
    #   Check if there's a path of length 3 from a to d avoiding v.
    #   Path a-b-c-d: b in N(a)\{v}, c in N(b)\{a,v}, d in N(c)\{b,v}
    #   But we also need d to be the specific d we're checking.
    
    # Let me just do: for each v, build the subgraph induced by N(v) plus check
    # paths of length 3 between pairs of neighbors.
    
    # Most practical for n<=35:
    # For each vertex v (0 to n-1):
    #   neighbors = N(v)
    #   For each pair (a, d) in neighbors, a < d:
    #     For each b in N(a) \ {v}:
    #       For each c in N(b) \ {a, v}:
    #         If d in N(c) and c != d and c != a and c != v and b != d and b != c:
    #           Then we have v-a-b-c-d-v, a 5-cycle.
    
    # Worst case: n=35, each vertex has ~17 neighbors, so for each v: C(17,2)*16*15 ≈ 136*240 ≈ 32640
    # Times 35 vertices ≈ 1.1M operations. That's fine.
    
    for v in range(n):
        nv = [x for x in range(n) if (min(v,x), max(v,x)) in adj]
        nv_set = set(nv)
        for i in range(len(nv)):
            a = nv[i]
            for j in range(i+1, len(nv)):
                d = nv[j]
                # Find path a-b-c-d avoiding v, with b,c distinct from v,a,d
                for b in range(n):
                    if b == v or b == a or b == d:
                        continue
                    if (min(a,b), max(a,b)) not in adj:
                        continue
                    for c in range(n):
                        if c == v or c == a or c == d or c == b:
                            continue
                        if (min(b,c), max(b,c)) not in adj:
                            continue
                        if (min(c,d), max(c,d)) in adj:
                            return True
    return False

def _max_independent_set_size(adj, n):
    """Find maximum independent set size using Bron-Kerbosch on complement."""
    # Complement graph: edge (u,v) in complement iff (u,v) NOT in adj
    # Independent set in G = clique in complement
    # Use Bron-Kerbosch with pivot
    
    comp_adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u+1, n):
            if (u, v) not in adj:
                comp_adj[u].add(v)
                comp_adj[v].add(u)
    
    max_clique = [0]
    
    def bronk(R, P, X):
        if not P and not X:
            if len(R) > max_clique[0]:
                max_clique[0] = len(R)
            return
        # Pivot selection
        if P:
            # Choose pivot u from P ∪ X with max |N(u) ∩ P|
            union = P | X
            pivot = max(union, key=lambda u: len(comp_adj[u] & P))
            candidates = P - comp_adj[pivot]
        else:
            candidates = P
        
        for v in list(candidates):
            R2 = R | {v}
            P2 = P & comp_adj[v]
            X2 = X & comp_adj[v]
            bronk(R2, P2, X2)
            P.remove(v)
            X.add(v)
    
    bronk(frozenset(), frozenset(range(n)), frozenset())
    return max_clique[0]

def _max_independent_set_size_fast(adj, n):
    """Faster max independent set using bitset-based Bron-Kerbosch."""
    # Represent sets as Python ints (bitsets) for n <= 35
    # comp_adj[u] as bitmask
    comp_mask = [0] * n
    for u in range(n):
        for v in range(n):
            if u != v and (min(u,v), max(u,v)) not in adj:
                comp_mask[u] |= (1 << v)
    
    max_clique = [0]
    
    def bronk(R_count, P, X):
        if P == 0 and X == 0:
            if R_count > max_clique[0]:
                max_clique[0] = R_count
            return
        if P == 0:
            return  # can't grow further, and X non-empty means no larger clique in this branch
        # Pivot
        union = P | X
        # Pick pivot with most neighbors in P
        best_pivot = -1
        best_count = -1
        tmp = union
        while tmp:
            lb = tmp & (-tmp)
            u = lb.bit_length() - 1
            cnt = (comp_mask[u] & P).bit_count()
            if cnt > best_count:
                best_count = cnt
                best_pivot = u
            tmp ^= lb
        candidates = P & ~(comp_mask[best_pivot])
        
        tmp = candidates
        while tmp:
            lb = tmp & (-tmp)
            v = lb.bit_length() - 1
            bronk(R_count + 1, P & comp_mask[v], X & comp_mask[v])
            P &= ~lb
            X |= lb
            tmp ^= lb
    
    bronk(0, (1 << n) - 1, 0)
    return max_clique[0]

def _try_add_vertex(adj, n, new_vertex, neighbors_of_new):
    """Check if adding new_vertex with given neighbors maintains C5-free and alpha < 9."""
    new_n = n + 1
    new_adj = set(adj)
    for nb in neighbors_of_new:
        new_adj.add((min(nb, new_vertex), max(nb, new_vertex)))
    
    if _has_c5(new_adj, new_n):
        return False
    if _max_independent_set_size_fast(new_adj, new_n) >= 9:
        return False
    return True

def construct():
    deadline = time.time() + 110  # 110s safety margin
    
    # Start with 8 disjoint K4s
    n, adj, edges = _build_8k4()
    best_result = {'n': n, 'edges': edges}
    
    # Try to add vertex 32 (n=33)
    # Vertex 32 needs to connect to some subset of {0..31}
    # For C5-freeness: vertex 32's neighbors must not create a 5-cycle
    # A 5-cycle through 32: 32-a-b-c-d-32 where a,d in N(32), and a-b-c-d is a path
    # Since original graph is 8 disjoint K4s, paths of length 3 within a K4 don't exist
    # (K4 has diameter 1), so any path a-b-c-d must use vertices from different K4s.
    # But in 8 disjoint K4s, there are NO edges between different K4s!
    # So in the original graph, the only edges are within K4s.
    # A path a-b-c-d requires edges (a,b),(b,c),(c,d).
    # Since edges only exist within K4s, a,b,c,d must all be in the same K4.
    # But K4 has 4 vertices, so a,b,c,d are all 4 vertices of one K4.
    # The path would be a permutation of the 4 vertices.
    # In K4, any 3 consecutive vertices form a path (complete graph).
    # So: if 32 connects to two vertices a,d in the SAME K4, and there exist b,c in that K4
    # (the other two vertices) such that a-b-c-d is a path, then 32-a-b-c-d-32 is a C5.
    # In K4 with vertices {a,b,c,d}, the path a-b-c-d exists (it's a Hamiltonian path).
    # So: 32 CANNOT connect to two vertices in the same K4!
    
    # Therefore, N(32) can contain at most 1 vertex from each K4.
    # Max |N(32)| = 8 (one from each K4).
    
    # For alpha < 9: We need alpha(G + v32) < 9.
    # An independent set of size 9 in the new graph either:
    # (a) includes v32: then the other 8 vertices must be independent from each other AND
    #     non-adjacent to v32. So we need 8 mutually non-adjacent vertices all NOT in N(32).
    #     In 8 disjoint K4s, an independent set picks at most 1 from each K4.
    #     So we need one vertex from each K4, none of which are in N(32).
    #     If N(32) has exactly 1 vertex from each K4, then for each K4 there are 3 choices
    #     not in N(32). So we can pick 8 vertices (one from each K4, avoiding N(32)).
    #     This gives an independent set of size 9 (including v32)! 
    #     So we need alpha < 9, meaning we must PREVENT this.
    #     To prevent: for at least one K4, ALL 4 vertices must be in N(32). 
    #     But we showed N(32) can have at most 1 per K4. Contradiction!
    #     Wait - that means if we add v32 with any valid neighbor set (at most 1 per K4),
    #     there's always an independent set of size 9?
    #     
    #     Let me re-examine. If N(32) has k_i vertices from K4_i (k_i <= 1),
    #     then from K4_i we can pick (4 - k_i) vertices for the independent set.
    #     Total independent set size including v32 = 1 + sum(4 - k_i) = 1 + 32 - sum(k_i) = 33 - sum(k_i).
    #     Wait no - independent set picks at most 1 per K4, not all non-neighbors.
    #     
    #     Correction: An independent set in 8 disjoint K4s picks at most 1 vertex from each K4.
    #     If v32 is in the independent set, then no neighbor of v32 can be in it.
    #     From K4_i, if k_i vertices are neighbors of v32, we can pick at most 1 from the remaining (4-k_i).
    #     Since 4-k_i >= 3 > 0, we can always pick 1 from each K4.
    #     So independent set = {v32} + {one from each K4, not in N(32)} has size 1+8=9.
    #     This is ALWAYS possible when N(32) has at most 1 per K4!
    #     
    #     Therefore: it's IMPOSSIBLE to add a 33rd vertex to 8 disjoint K4s while
    #     maintaining both C5-free and alpha < 9.
    #     
    #     What if we restructure the graph? Maybe merge some K4s or use different blocks.
    #     E.g., 7 K4s + 1 K5? K5 has no C5? No! K5 contains C5 as a subgraph.
    #     K4 is the largest complete graph with no C5.
    #     
    #     What about using C4s? C4 has no C5. Independence number of C4 is 2.
    #     8 C4s: n=32, alpha=16. Too high.
    #     
    #     What about mixing? The key constraint is alpha < 9 with n up to 35.
    #     With n=33, we need alpha <= 8.
    #     
    #     Let me try a different approach: start from scratch with a different structure.
    #     What about 9 copies of K4 minus an edge? K4-e has 4 vertices, 5 edges, no C5, alpha=2.
    #     9*(K4-e): n=36 > 35. Too many.
    #     8*(K4-e) + 1 vertex: n=33. Alpha of 8*(K4-e) = 16. Adding 1 vertex can't fix that.
    #     
    #     The fundamental issue: to get alpha < 9 with n=33, we need a graph where
    #     vertices are densely connected. The 8* K4 construction is optimal for alpha
    #     given the C5-free constraint (each K4 contributes 4 vertices and alpha 1).
    #     For n=33, we'd need "8.25 K4s" worth of density, which isn't possible with
    #     disjoint cliques.
    #     
    #     Perhaps we need a non-disjoint structure. E.g., a graph where some vertices
    #     are shared between cliques, or a more complex C5-free structure.
    #     
    #     Given the time constraint and the difficulty, let's return the known n=32 result.
    
    return best_result

# EVOLVE-BLOCK-END
