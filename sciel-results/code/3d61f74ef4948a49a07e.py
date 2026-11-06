"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n = 20
    # Adjacency matrix for O(1) lookup
    adj = [[False] * n for _ in range(n)]
    edges = []
    
    # Helper to check if adding edge (u, v) creates a K2,3
    # A K2,3 is formed if there exist two vertices x, y such that they have 3 common neighbors.
    # When we add edge (u, v), the only new common neighbors that can be formed are:
    # 1. u and v themselves? No, u and v are not neighbors of each other in the "common neighbor" sense for the pair (u,v).
    #    Actually, the condition is: for any pair (x, y), |N(x) ∩ N(y)| <= 2.
    #    Adding edge (u, v) changes N(u) and N(v).
    #    Specifically:
    #    - For pair (u, v): |N(u) ∩ N(v)| must be <= 2. Note that u is not in N(u), v is not in N(v).
    #      The common neighbors of u and v are vertices w such that w is connected to both u and v.
    #      Adding (u, v) does NOT change the set of common neighbors of u and v (since u is not in N(u) and v is not in N(v), and the edge (u,v) doesn't add u to N(v) or v to N(u) in a way that affects the intersection N(u)∩N(v) for the *other* pairs? Wait.
    #      Let's re-evaluate.
    #      N(u) is the set of neighbors of u.
    #      Before adding (u, v), v is not in N(u). After, v is in N(u).
    #      The common neighbors of u and v are {w : w in N(u) and w in N(v)}.
    #      Adding (u, v) adds v to N(u) and u to N(v).
    #      Does v become a common neighbor of u and v? No, v is not in N(v).
    #      Does u become a common neighbor of u and v? No, u is not in N(u).
    #      So the set N(u) ∩ N(v) remains exactly the same as it was before adding the edge?
    #      No! The set N(u) changes.
    #      Let C(u, v) = N(u) ∩ N(v).
    #      If we add edge (u, v), does C(u, v) change?
    #      w is in C(u, v) if w~u and w~v.
    #      The edge (u, v) does not affect whether w~u or w~v for w != u, v.
    #      For w = u: u~u is false.
    #      For w = v: v~v is false.
    #      So C(u, v) is invariant under adding edge (u, v).
    #      
    #      However, adding (u, v) DOES affect other pairs.
    #      Consider pair (u, w) where w != v.
    #      N(u) gains v. N(w) is unchanged.
    #      New C(u, w) = Old C(u, w) ∪ ({v} if v~w else empty).
    #      So if v~w, then |C(u, w)| increases by 1.
    #      We must ensure that for all w != v, if v~w, then |Old C(u, w)| <= 1.
    #      Similarly for pair (v, w) where w != u.
    #      N(v) gains u.
    #      If u~w, then |Old C(v, w)| must be <= 1.
    #      
    #      Also consider pair (x, y) where x, y != u, v.
    #      N(x) and N(y) are unchanged. So C(x, y) is unchanged.
    #      
    #      So the check is:
    #      1. |N(u) ∩ N(v)| <= 2 (This was already true if the graph was K2,3-free, but we must ensure it doesn't exceed 2. Since it's invariant, we just need to check the current count).
    #      2. For all w in N(v) (excluding u, which isn't in N(v) yet, but v is in N(u) now? No, we are checking validity of adding (u,v)), 
    #         actually, let's just iterate all w.
    #         If w is a neighbor of v (w != u), then u is added to N(v)?? No, v is added to N(u).
    #         Let's stick to the impact on C(x, y).
    #         Adding (u, v) increases |C(u, w)| by 1 if w is a neighbor of v.
    #         Adding (u, v) increases |C(v, w)| by 1 if w is a neighbor of u.
    #         
    #      So, to check if adding (u, v) is safe:
    #      a) Current |N(u) ∩ N(v)| must be <= 2. (It will remain <= 2 after adding).
    #      b) For every neighbor w of v (w != u), the current |N(u) ∩ N(w)| must be <= 1. (Because it will become 2).
    #      c) For every neighbor w of u (w != v), the current |N(v) ∩ N(w)| must be <= 1. (Because it will become 2).
    
    def can_add(u, v):
        if adj[u][v]:
            return False
        # Check condition (a): common neighbors of u and v
        common_uv = 0
        for w in range(n):
            if w != u and w != v:
                if adj[u][w] and adj[v][w]:
                    common_uv += 1
        if common_uv > 2:
            return False
            
        # Check condition (b): for all w in N(v), |N(u) ∩ N(w)| must be <= 1
        for w in range(n):
            if w == u or w == v:
                continue
            if adj[v][w]:
                # w is neighbor of v. Adding (u,v) makes u a neighbor of v? No, v is neighbor of u.
                # Wait. N(u) gets v. N(w) is unchanged.
                # C(u, w) = N(u) ∩ N(w).
                # New C(u, w) = Old C(u, w) ∪ ({v} if v in N(w) else empty).
                # We know v in N(w) is true (since adj[v][w] is true).
                # So |New C(u, w)| = |Old C(u, w)| + 1.
                # We need |New C(u, w)| <= 2, so |Old C(u, w)| <= 1.
                common_uw = 0
                for z in range(n):
                    if z != u and z != w:
                        if adj[u][z] and adj[w][z]:
                            common_uw += 1
                if common_uw > 1:
                    return False
                    
        # Check condition (c): for all w in N(u), |N(v) ∩ N(w)| must be <= 1
        for w in range(n):
            if w == u or w == v:
                continue
            if adj[u][w]:
                # w is neighbor of u. Adding (u,v) makes v a neighbor of u.
                # C(v, w) = N(v) ∩ N(w).
                # New C(v, w) = Old C(v, w) ∪ ({u} if u in N(w) else empty).
                # We know u in N(w) is true (since adj[u][w] is true).
                # So |New C(v, w)| = |Old C(v, w)| + 1.
                # We need |New C(v, w)| <= 2, so |Old C(v, w)| <= 1.
                common_vw = 0
                for z in range(n):
                    if z != v and z != w:
                        if adj[v][z] and adj[w][z]:
                            common_vw += 1
                if common_vw > 1:
                    return False
                    
        return True

    # Greedy strategy: Try to add edges in a specific order or random order with restarts?
    # Deterministic order might get stuck. Random order with multiple restarts is better.
    # But we have a time limit. Let's do a few random restarts.
    
    best_edge_list = []
    best_count = 0
    
    # Seed for reproducibility? No, random is fine.
    # We will run several iterations of greedy insertion with different random edge orderings.
    # n=20, total possible edges = 190.
    # Greedy insertion is O(n^3) per edge in worst case if we check all, but we can optimize.
    # Actually, the check above is O(n^2) per edge candidate. 190 edges * 190 candidates * O(n^2) is too slow?
    # 190 * 190 * 400 = ~14 million operations. That's fast in Python (under 1s).
    # So we can do many restarts.
    
    max_restarts = 50
    for r in range(max_restarts):
        if time.time() > deadline - 5:
            break
            
        # Reset graph
        adj = [[False] * n for _ in range(n)]
        current_edges = []
        
        # Create list of all possible edges
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_edges)
        
        # Greedy insertion
        for u, v in all_edges:
            if time.time() > deadline - 5:
                break
            if can_add(u, v):
                adj[u][v] = True
                adj[v][u] = True
                current_edges.append([u, v])
                
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edge_list = current_edges
            
    return best_edge_list
# EVOLVE-BLOCK-END
