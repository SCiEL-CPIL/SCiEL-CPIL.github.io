"""K3,3-free graph construction on 20 vertices."""
import random
import time
import sys

def construct():
    start = time.time()
    deadline = start + 110  # safety margin before 150s kill

    n = 20
    # All possible edges as (u, v) with u < v
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Precompute for each edge, which triples (a,b,c) would be violated if this edge is added
    # Actually, let's think about this differently.
    # A K3,3 subgraph exists if there are 3 vertices on one side and 3 on the other,
    # with all 9 edges between them present.
    # Equivalently: there exist 3 vertices S and 3 vertices T (disjoint) such that
    # every vertex in S is connected to every vertex in T.
    # Another equivalent: there exist 3 vertices a,b,c such that they have at least 3 common neighbors.
    # Wait, that's not quite right either. Let me reconsider.
    # 
    # K3,3 has two parts of size 3. A K3,3 subgraph in G means there exist disjoint sets
    # S = {s1,s2,s3} and T = {t1,t2,t3} such that for all s in S, t in T, edge (s,t) exists.
    # 
    # Efficient check: For each pair of vertices (i,j), compute common neighbors.
    # If any 3 vertices i,j,k have >= 3 common neighbors, we need to check more carefully.
    # Actually, the standard way: for each triple (a,b,c), let CN = common neighbors of a,b,c.
    # If |CN| >= 3, then pick any 3 from CN, say d,e,f. But we also need that d,e,f are NOT
    # connected to each other? No, K3,3 doesn't require that. K3,3 just requires the bipartite
    # structure. So if a,b,c all connect to d,e,f, then {a,b,c} and {d,e,f} form a K3,3 
    # IF a,b,c are not in {d,e,f} and d,e,f are not in {a,b,c}.
    # But wait, we also need that there are no edges within {a,b,c} or within {d,e,f}? 
    # No! K3,3 as a subgraph (not induced) only requires the 9 cross edges. Edges within
    # parts are allowed. So the condition is: there exist disjoint 3-sets S and T such that
    # all 9 edges between S and T exist.
    #
    # So the check is: do there exist disjoint triples S, T with all cross edges present?
    #
    # For efficiency with bitmask:
    # For each vertex v, adj[v] is a bitmask of neighbors.
    # For a triple (a,b,c): common = adj[a] & adj[b] & adj[c]
    # If popcount(common) >= 3, we have potential K3,3. But we need the 3 common neighbors
    # to be distinct from a,b,c. Since it's a simple graph, a is not adjacent to itself, so
    # a is not in adj[a], hence not in common. Same for b and c. So common neighbors are
    # automatically distinct from a,b,c.
    # So if popcount(adj[a] & adj[b] & adj[c]) >= 3, we have a K3,3.
    
    def is_k33_free(adj):
        """Check if graph with adjacency bitmasks is K3,3-free."""
        for a in range(n):
            for b in range(a + 1, n):
                common_ab = adj[a] & adj[b]
                if common_ab == 0:
                    continue
                for c in range(b + 1, n):
                    common_abc = common_ab & adj[c]
                    if common_abc & (common_abc - 1):  # at least 2 bits set, need >=3
                        # Check if at least 3 bits
                        if common_abc & (common_abc - 1) & (common_abc - (common_abc & (common_abc - 1))):
                            # That's getting complicated. Just count bits.
                            pass
                    # Simpler: count bits
                    cnt = 0
                    tmp = common_abc
                    while tmp:
                        tmp &= tmp - 1
                        cnt += 1
                        if cnt >= 3:
                            return False
        return True
    
    # Better: use a fast popcount
    def popcount(x):
        cnt = 0
        while x:
            x &= x - 1
            cnt += 1
        return cnt

    def is_k33_free_fast(adj):
        """Check if graph with adjacency bitmasks is K3,3-free."""
        for a in range(n):
            for b in range(a + 1, n):
                common_ab = adj[a] & adj[b]
                if common_ab == 0:
                    continue
                for c in range(b + 1, n):
                    common_abc = common_ab & adj[c]
                    if popcount(common_abc) >= 3:
                        return False
        return True
    
    def try_add_edge(adj, u, v):
        """Try to add edge (u,v) to adj. Returns True if valid."""
        # After adding edge, check if any K3,3 is created.
        # The new K3,3 must use the new edge (u,v).
        # So we need 3 vertices on one side containing u, and 3 on the other containing v,
        # or vice versa.
        # Case 1: u is in S (size 3), v is in T (size 3).
        #   S = {u, x, y}, T = {v, z, w}
        #   Need: u-z, u-w, x-v, x-z, x-w, y-v, y-z, y-w all exist (u-v is the new edge).
        #   x,y are common neighbors of u and v (excluding u,v themselves).
        #   z,w are common neighbors of u and v.
        #   Actually: x must be adjacent to v, z, w. y must be adjacent to v, z, w.
        #   z must be adjacent to u, x, y. w must be adjacent to u, x, y.
        #
        # Let's think more carefully. The K3,3 uses edge (u,v). WLOG u in left part, v in right part.
        # Left part: {u, a, b}, Right part: {v, c, d}
        # Required edges: u-v (new), u-c, u-d, a-v, a-c, a-d, b-v, b-c, b-d
        # So: a and b must be adjacent to v, c, d.
        # c and d must be adjacent to u, a, b.
        # 
        # Let N_u = neighbors of u (before adding edge)
        # Let N_v = neighbors of v (before adding edge)
        # a, b must be in N_v (adjacent to v) and also adjacent to c, d.
        # c, d must be in N_u (adjacent to u) and also adjacent to a, b.
        # 
        # So: pick c, d from N_u. Then a, b must be in N_v ∩ N_c ∩ N_d.
        # We need at least 2 such vertices a, b, and they must be adjacent to both c and d
        # (which they are by construction) and to v (which they are by being in N_v).
        # Also a ≠ b, a ≠ c, a ≠ d, b ≠ c, b ≠ d, a ≠ u, b ≠ u, c ≠ v, d ≠ v.
        # Since it's a simple graph and u,v are not self-adjacent, and c,d ∈ N_u means c,d ≠ u,
        # and a,b ∈ N_v means a,b ≠ v. Also c,d are neighbors of u so c,d ≠ v? Not necessarily.
        # c could equal v? No, c ∈ N_u and if c=v then v is adjacent to u, but we're adding that edge,
        # so before adding, v is NOT in N_u. So c ≠ v. Similarly d ≠ v.
        # a,b ∈ N_v, so a,b ≠ v. a,b could equal u? u is not in N_v before adding edge, so no.
        # 
        # So the check is:
        # For each pair (c,d) from N_u (c < d):
        #   candidates = N_v & adj[c] & adj[d]
        #   Remove u and v from candidates (shouldn't be there anyway)
        #   If popcount(candidates) >= 2, then we can form a K3,3.
        
        N_u = adj[u]
        N_v = adj[v]
        
        # c, d are in N_u
        # a, b are in N_v ∩ adj[c] ∩ adj[d]
        
        # Iterate over pairs in N_u
        cu = N_u
        while cu:
            lsb_c = cu & (-cu)
            cu ^= lsb_c
            c = lsb_c.bit_length() - 1
            cd = N_u & (~lsb_c)
            while cd:
                lsb_d = cd & (-cd)
                cd ^= lsb_d
                d = lsb_d.bit_length() - 1
                # candidates for a, b
                candidates = N_v & adj[c] & adj[d]
                # Remove u and v just in case
                candidates &= ~(1 << u)
                candidates &= ~(1 << v)
                if popcount(candidates) >= 2:
                    return False
        return True
    
    # Greedy construction
    def greedy_construct():
        adj = [0] * n
        random.shuffle(all_edges)
        for u, v in all_edges:
            if try_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        # Count edges
        cnt = 0
        for i in range(n):
            cnt += popcount(adj[i])
        return cnt // 2, adj
    
    # Local search: try edge swaps
    def local_search(adj, max_iters=50000):
        adj = adj[:]  # copy
        current_edges = 0
        for i in range(n):
            current_edges += popcount(adj[i])
        current_edges //= 2
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            # Try to add a random non-edge
            # Or try a swap: remove edge (a,b), add edge (c,d)
            # Let's try adding a random non-edge first
            u, v = random.choice(all_edges)
            if not (adj[u] & (1 << v)):
                if try_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    current_edges += 1
            # Also try removing an edge and adding two others? That's more complex.
            # For now, just try adding.
        
        return current_edges, adj
    
    # Run multiple greedy constructions and pick the best
    best_edges = 0
    best_adj = None
    num_trials = 30
    
    for trial in range(num_trials):
        if time.time() > deadline - 10:
            break
        edges, adj = greedy_construct()
        if edges > best_edges:
            best_edges = edges
            best_adj = adj[:]
    
    # Local search on the best
    if best_adj is not None and time.time() < deadline - 10:
        best_edges, best_adj = local_search(best_adj)
    
    # Convert adj to edge list
    edges_list = []
    for u in range(n):
        v_mask = best_adj[u]
        while v_mask:
            lsb = v_mask & (-v_mask)
            v_mask ^= lsb
            v = lsb.bit_length() - 1
            if u < v:
                edges_list.append([u, v])
    
    return edges_list
