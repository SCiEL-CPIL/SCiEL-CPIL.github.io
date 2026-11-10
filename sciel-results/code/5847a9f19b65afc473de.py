import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 120s limit
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = len(all_edges)  # 120

    def creates_q3(u, v, adj):
        """Check if adding edge (u,v) to graph with adjacency 'adj' creates a Q3."""
        # Fast pre-checks: need at least 3 common neighbors with u or v
        mask_u = adj[u]
        mask_v = adj[v]
        if not mask_u or not mask_v:
            return False
        
        # For a Q3, u and v must be opposite vertices or adjacent vertices.
        # If u-v is an edge in Q3, then u and v share 2 neighbors in Q3 (the other two vertices adjacent to both)
        # Wait, in Q3, if u-v is an edge, they share 2 common neighbors (the two vertices connected to both u and v).
        # If u and v are not connected in Q3, they share 0 common neighbors (if distance 1 or 3) or 2 (if distance 2).
        # Actually, the 8 vertices of Q3 form 4 pairs of opposite vertices (distance 3, 0 common neighbors),
        # 12 pairs of distance 2 (2 common neighbors), and 12 pairs of distance 1 (2 common neighbors).
        # Wait, let's re-examine.
        # In Q3, the adjacency matrix has 0 diagonal, 1 for edges, 0 otherwise.
        # Number of common neighbors:
        # - Distance 1 (adjacent): 2 common neighbors.
        # - Distance 2 (not adjacent, not opposite): 2 common neighbors.
        # - Distance 3 (opposite): 0 common neighbors.
        
        # So if we are adding edge (u,v), in the resulting Q3 subgraph, u and v must be at distance 1 or 2 in Q3?
        # No, the edge (u,v) IS an edge in the graph. So in the Q3 subgraph, (u,v) must be an edge.
        # Thus u and v are at distance 1 in Q3, so they must share exactly 2 common neighbors in the Q3.
        # Therefore, in the graph, u and v must share at least 2 common neighbors.
        common_uv = mask_u & mask_v
        if common_uv.bit_count() < 2:
            return False
        
        # We need to find a Q3 containing edge (u,v).
        # The Q3 has vertices: u, v, and 6 others.
        # The 6 others form a cycle of length 6 in the "equator" or something?
        # Let's use the structure: Q3 can be seen as two squares connected by matching edges.
        # If (u,v) is an edge, then u is connected to 2 other vertices (call them a1, a2) and v is connected to 2 other vertices (call them b1, b2).
        # The common neighbors of u and v are the vertices c1, c2 such that u-c1-v and u-c2-v are paths of length 2? No.
        # In Q3, if u-v is an edge, the neighbors of u are v, a1, a2. The neighbors of v are u, b1, b2.
        # The common neighbors of u and v are the vertices adjacent to both. In Q3, these are c1, c2.
        # Wait, if u is connected to v, a1, a2, and v is connected to u, b1, b2, and common neighbors are c1, c2.
        # Then the edges of Q3 are:
        # (u,v), (u,a1), (u,a2), (u,c1), (u,c2)? No, u has degree 3.
        # Let's label Q3 vertices as binary strings.
        # u = 000, v = 001 (edge).
        # Neighbors of 000: 001, 010, 100.
        # Neighbors of 001: 000, 011, 101.
        # Common neighbors of 000 and 001: none!
        # Wait, my earlier analysis was wrong.
        # In Q3, adjacent vertices have 0 common neighbors?
        # 000 and 001:
        # N(000) = {001, 010, 100}
        # N(001) = {000, 011, 101}
        # Intersection: empty.
        # So adjacent vertices in Q3 have 0 common neighbors.
        # What about distance 2? 000 and 011.
        # N(000) = {001, 010, 100}
        # N(011) = {001, 010, 111}
        # Intersection: {001, 010}. So 2 common neighbors.
        # What about opposite? 000 and 111.
        # N(000) = {001, 010, 100}
        # N(111) = {011, 101, 110}
        # Intersection: empty.
        
        # So, if (u,v) is an edge in Q3, they have 0 common neighbors in Q3.
        # But in the graph, they might have common neighbors that are NOT part of the Q3.
        # So the check `common_uv.bit_count() < 2` is NOT a valid fast rejection for edge (u,v) in Q3.
        # In fact, if they have 0 common neighbors in the graph, it's impossible for them to be in a Q3 as distance 2, but they could be distance 1.
        # If they are distance 1 in Q3, they have 0 common neighbors in Q3.
        # So we can't use common neighbors to reject immediately.
        
        # Let's go back to the standard approach.
        # If (u,v) is an edge in Q3, then u has 2 other neighbors in Q3, say a, b.
        # v has 2 other neighbors in Q3, say c, d.
        # The structure is:
        # u is connected to v, a, b.
        # v is connected to u, c, d.
        # a is connected to u, and two others. One must be common with c or d?
        # Let's use the property that Q3 is bipartite.
        # Partition into X and Y. If u in X, v in Y.
        # Then a, b in Y. c, d in X.
        # Edges: (u,v), (u,a), (u,b), (v,c), (v,d).
        # Also (a, c'), (a, d') for some c', d' in X? No.
        # The neighbors of a (in Y) are u, and two in X. One of them must be c or d? No.
        # Let's look at the 6-cycle: u-c-v-d-u? No, that's a 4-cycle.
        # The 6 other vertices form a 6-cycle? No.
        
        # Let's use the standard enumeration:
        # Choose 2 neighbors of u (other than v): a, b.
        # Choose 2 neighbors of v (other than u): c, d.
        # Check if {u,v,a,b,c,d, e} forms a Q3 for some e? No, 7 vertices. Q3 has 8.
        # The 8th vertex e must be adjacent to a, b, c, d in specific ways.
        
        # Actually, a simpler check:
        # A graph contains Q3 iff it contains a set of 8 vertices inducing a subgraph isomorphic to Q3.
        # We can check all 8-vertex subsets? C(16,8) = 12870. For each, check if isomorphic to Q3.
        # Isomorphism check is fast.
        # But we need to check if ADDING (u,v) creates a Q3.
        # So we check if the new graph has a Q3 that includes (u,v).
        # Such a Q3 must include u and v.
        # So we need to find 6 other vertices such that the 8 form a Q3 containing (u,v).
        
        # Let's implement a direct check for Q3 containing (u,v).
        # In Q3, if (u,v) is an edge, then:
        # - u has 2 other neighbors in Q3, say a, b.
        # - v has 2 other neighbors in Q3, say c, d.
        # - The remaining vertex e is connected to a, b, c, d? No.
        # Let's list edges of Q3 (000-111) containing (000, 001):
        # Edges: (000,001), (000,010), (000,100), (001,011), (001,101), (010,011), (010,110), (100,101), (100,110), (011,111), (101,111), (110,111).
        # Vertices: 000(u), 001(v), 010(a), 100(b), 011(c), 101(d), 110(e), 111(f).
        # Neighbors of u(000): v(001), a(010), b(100).
        # Neighbors of v(001): u(000), c(011), d(101).
        # Neighbors of a(010): u(000), c(011), e(110).
        # Neighbors of b(100): u(000), d(101), e(110).
        # Neighbors of c(011): v(001), a(010), f(111).
        # Neighbors of d(101): v(001), b(100), f(111).
        # Neighbors of e(110): a(010), b(100), f(111).
        # Neighbors of f(111): c(011), d(101), e(110).
        
        # So the conditions for {u,v,a,b,c,d,e,f} to be a Q3 with (u,v) as edge:
        # 1. a, b are neighbors of u (in graph), a != b, a != v, b != v.
        # 2. c, d are neighbors of v (in graph), c != d, c != u, d != u.
        # 3. a, c are neighbors in graph.
        # 4. b, d are neighbors in graph.
        # 5. e is a neighbor of a, b, f.
        # 6. f is a neighbor of c, d, e.
        # Also, we need to ensure no extra edges? No, subgraph isomorphic means the 8 vertices induce a supergraph of Q3.
        # Wait, "contains no subgraph isomorphic to Q3". This usually means no INDUCED subgraph? Or no subgraph?
        # The task says "no subgraph isomorphic to the 3-cube Q3". This typically means no subgraph (not necessarily induced).
        # So if the 8 vertices contain all 12 edges of Q3, it's a hit, even if they have extra edges.
        # So we just need to verify the 12 edges of the specific Q3 pattern exist.
        
        # Algorithm:
        # For each pair (a,b) of neighbors of u (a != v):
        #   For each pair (c,d) of neighbors of v (c != u):
        #     Check if (a,c) and (b,d) are edges.
        #     If yes, find e such that (a,e), (b,e), (e,f), (c,f), (d,f) are edges.
        #     Actually, e must be neighbor of a and b. f must be neighbor of c and d.
        #     And (e,f) must be an edge.
        #     Also e != f, and e,f not in {u,v,a,b,c,d}.
        
        adj_u = adj[u] & ~(1 << v)
        adj_v = adj[v] & ~(1 << u)
        
        # Get lists of neighbors
        neigh_u = [i for i in range(n) if (adj_u >> i) & 1]
        neigh_v = [i for i in range(n) if (adj_v >> i) & 1]
        
        # Precompute adj matrix for faster lookup? adj is bitmasks, so adj[i] & (1<<j) is fast.
        
        for a in neigh_u:
            for b in neigh_u:
                if a >= b: continue
                for c in neigh_v:
                    for d in neigh_v:
                        if c >= d: continue
                        if a == c or a == d or b == c or b == d:
                            continue
                        # Check edges (a,c) and (b,d)
                        if not ((adj[a] >> c) & 1) or not ((adj[b] >> d) & 1):
                            continue
                        # Find e in N(a) ∩ N(b)
                        # Find f in N(c) ∩ N(d)
                        # e != f, e not in {u,v,a,b,c,d}, f not in {u,v,a,b,c,d}
                        forbidden = (1 << u) | (1 << v) | (1 << a) | (1 << b) | (1 << c) | (1 << d)
                        mask_e = (adj[a] & adj[b]) & ~forbidden
                        mask_f = (adj[c] & adj[d]) & ~forbidden
                        if not mask_e or not mask_f:
                            continue
                        # Check if there exists e in mask_e, f in mask_f such that (e,f) is edge
                        # and e != f (implied by forbidden if we remove them, but e and f could be same? No, if e==f, then e is in forbidden? No, e is in mask_e which excludes a,b,c,d,u,v. f is in mask_f. If e==f, then e is in both. But we need e != f? In Q3, e and f are distinct.
                        # So we need e != f.
                        # Iterate over smaller set
                        if mask_e.bit_count() <= mask_f.bit_count():
                            me = mask_e
                            while me:
                                lsb_e = me & -me
                                e = lsb_e.bit_length() - 1
                                me ^= lsb_e
                                # f in mask_f, f != e
                                mf = mask_f & ~lsb_e
                                if mf & adj[e]:
                                    return True
                        else:
                            mf = mask_f
                            while mf:
                                lsb_f = mf & -mf
                                f = lsb_f.bit_length() - 1
                                mf ^= lsb_f
                                me = mask_e & ~lsb_f
                                if me & adj[f]:
                                    return True
        return False

    def edge_count(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def greedy_construct():
        adj = [0] * n
        shuffled_edges = all_edges[:]
        random.shuffle(shuffled_edges)
        for u, v in shuffled_edges:
            if not creates_q3(u, v, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def local_search_add(adj):
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            missing = [(i, j) for i in range(n) for j in range(i + 1, n)
                       if not (adj[i] >> j) & 1]
            random.shuffle(missing)
            for u, v in missing:
                if not creates_q3(u, v, adj):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    improved = True
        return adj

    def local_search_add_exhaustive(adj):
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] >> j) & 1:
                    if not creates_q3(i, j, adj):
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                        improved = True
        return improved

    def kick_and_refill(adj, kick_size):
        current_adj = adj[:]
        edges = [(i, j) for i in range(n) for j in range(i + 1, n) if (current_adj[i] >> j) & 1]
        for _ in range(kick_size):
            if not edges:
                break
            idx = random.randint(0, len(edges) - 1)
            u, v = edges[idx]
            current_adj[u] &= ~(1 << v)
            current_adj[v] &= ~(1 << u)
            edges.pop(idx)
        current_adj = local_search_add(current_adj)
        while local_search_add_exhaustive(current_adj) and time.time() < deadline - 5:
            pass
        return current_adj

    best_adj = [0] * n
    best_count = 0

    # Try multiple initializations
    while time.time() < deadline - 10:
        adj = greedy_construct()
        adj = local_search_add(adj)
        while local_search_add_exhaustive(adj) and time.time() < deadline - 5:
            pass
        count = edge_count(adj)
        if count > best_count:
            best_count = count
            best_adj = adj[:]

        # Kicks
        if time.time() < deadline - 5:
            for kick_size in [2, 3, 4, 5]:
                if time.time() > deadline - 5:
                    break
                new_adj = kick_and_refill(adj, kick_size)
                new_count = edge_count(new_adj)
                if new_count > best_count:
                    best_count = new_count
                    best_adj = new_adj[:]
                    adj = new_adj

    # Final polish
    best_adj = local_search_add(best_adj)
    while local_search_add_exhaustive(best_adj) and time.time() < deadline - 1:
        pass
    best_count = edge_count(best_adj)

    result = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_adj[i] >> j) & 1:
                result.append([i, j])
    return result

