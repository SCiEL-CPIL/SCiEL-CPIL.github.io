import time
import random
import numpy as np

N = 20
DEADLINE = 150.0
MARGIN = 10.0

def _edge_list_from_adj(adj):
    out = []
    for u in range(N):
        for v in range(u + 1, N):
            if adj[u][v]:
                out.append([u, v])
    return out

def construct():
    start = time.time()
    best_edges = 0
    best_adj = np.zeros((N, N), dtype=int)
    
    # Pairs of vertices
    pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]
    
    def _is_valid(adj, u, v):
        # Check if adding (u,v) creates a K2,3.
        # A K2,3 is formed if there are 3 common neighbors of u and v.
        # Note: Adding (u,v) does not change the common neighbors of u and v directly 
        # in terms of existing edges, but it *does* affect the common neighbor counts 
        # of pairs (u, w) and (v, w) for w != u, v.
        # Wait, the condition is: for ALL pairs (x,y), |N(x) \cap N(y)| <= 2.
        # If we add (u,v), we increase the degree of u and v.
        # This increases |N(u) \cap N(w)| for all w != u.
        # Specifically, for a pair (u, w), the new common neighbors are the old common neighbors 
        # PLUS v (if v was a neighbor of w).
        # So we need to check:
        # 1. For the pair (u, v): The new common neighbors are the old common neighbors. 
        #    Wait, u and v are now connected. Does that change their common neighbors?
        #    N(u) gains v. N(v) gains u.
        #    Common neighbors of u and v: Vertices connected to BOTH u and v.
        #    Before adding (u,v), common neighbors were C.
        #    After adding (u,v), is v a common neighbor of u and v? No, v is not a neighbor of v (simple graph).
        #    Is u a common neighbor? No.
        #    So the set of common neighbors of u and v remains the same?
        #    Yes. The edges added are (u,v). The neighbors of u are N(u) \cup {v}.
        #    The neighbors of v are N(v) \cup {u}.
        #    Intersection: (N(u) \cup {v}) \cap (N(v) \cup {u})
        #    = (N(u) \cap N(v)) \cup (N(u) \cap {u}) \cup ({v} \cap N(v)) \cup ({v} \cap {u})
        #    N(u) \cap N(v) is the old set.
        #    N(u) \cap {u} is empty (no self loops).
        #    {v} \cap N(v) is empty (no self loops).
        #    {v} \cap {u} is empty.
        #    So |N(u) \cap N(v)| is UNCHANGED by adding (u,v).
        #    
        # 2. However, adding (u,v) DOES change the common neighbor counts for pairs (u, w) and (v, w).
        #    For pair (u, w):
        #    New N(u) = Old N(u) \cup {v}.
        #    New N(w) = Old N(w).
        #    New Common = (Old N(u) \cup {v}) \cap Old N(w) = (Old N(u) \cap Old N(w)) \cup ({v} \cap Old N(w)).
        #    If v is a neighbor of w, then v is added to the common neighbors.
        #    So the count increases by 1 if w is connected to v.
        #    We must ensure that for all w, the new count <= 2.
        #    i.e., Old Common(u,w) + (1 if w~v else 0) <= 2.
        #
        #    Similarly for pair (v, w):
        #    New Common(v,w) = Old Common(v,w) + (1 if w~u else 0).
        #    Must be <= 2.
        
        # So to check if adding (u,v) is valid:
        # 1. Check if u and v are already connected. If yes, invalid.
        # 2. For all w in 0..N-1, w != u, w != v:
        #    - If adj[u][w] and adj[v][w]:
        #      - Check pair (u, w): current common count of (u,w) must be < 2 if v is neighbor of w? 
        #        No, if v is neighbor of w, then adding (u,v) makes v a common neighbor of u and w.
        #        So we need current_common(u,w) + 1 <= 2 => current_common(u,w) <= 1.
        #      - Check pair (v, w): current_common(v,w) + 1 <= 2 => current_common(v,w) <= 1.
        #    - If not (adj[u][w] and adj[v][w]):
        #      - v is not a neighbor of w (or u is not).
        #      - If v is neighbor of w:
        #         - Pair (u,w) gains v as common neighbor. Need current_common(u,w) <= 1.
        #         - Pair (v,w) does NOT gain u as common neighbor (because u is not neighbor of w? 
        #           Wait, if w~v and w!~u, then u is not in N(w). So (v,w) common neighbors 
        #           are N(v) \cap N(w). Adding (u,v) adds u to N(v). Is u in N(w)? No. 
        #           So (v,w) count is UNCHANGED.
        #      - If u is neighbor of w (and v is not):
        #         - Symmetric. (u,w) unchanged, (v,w) gains u? No, u is already in N(w). 
        #           Adding (u,v) adds v to N(u). Is v in N(w)? No. So (v,w) unchanged.
        #           Wait, let's re-evaluate.
        #           Pair (v,w). N(v) gains u. N(w) is fixed.
        #           New Common = (Old N(v) \cup {u}) \cap N(w) = Old Common \cup ({u} \cap N(w)).
        #           If u is in N(w), then u is added to common neighbors.
        #           So if w~u, then (v,w) count increases by 1.
        #           We need current_common(v,w) <= 1.
        
        # Summary of checks for adding (u,v):
        # For each w != u, v:
        #   inc_uw = 1 if (adj[v][w] == 1) else 0  # v becomes common neighbor of u,w if v~w
        #   inc_vw = 1 if (adj[u][w] == 1) else 0  # u becomes common neighbor of v,w if u~w
        #   
        #   if inc_uw:
        #       if current_common(u, w) + inc_uw > 2: return False
        #   if inc_vw:
        #       if current_common(v, w) + inc_vw > 2: return False
        #
        # Also, we must ensure that the pair (u,v) itself doesn't violate.
        # As established, common(u,v) is unchanged. But we must ensure it was <= 2 before.
        # Since the graph is currently valid, common(u,v) <= 2.
        # So we just need to check the effects on (u,w) and (v,w).
        
        if adj[u][v]:
            return False
            
        # Vectorized check for all w
        # common_uw = number of common neighbors of u and w
        # common_vw = number of common neighbors of v and w
        
        # We can compute common neighbors for all pairs involving u or v quickly.
        # But doing this for every candidate edge in a loop is slow.
        # For the greedy step, we check one edge at a time.
        
        # Let's compute common counts for pairs (u, w) and (v, w) for all w.
        # adj[u] is a vector. adj[w] is a row.
        # common(u, w) = sum(adj[u] * adj[w])
        
        # We need to check for all w:
        # if adj[v][w]: check common(u,w) <= 1
        # if adj[u][w]: check common(v,w) <= 1
        
        # Let's compute common_u = sum(adj[u] * adj, axis=1) -> vector of common neighbors of u with all others
        # common_v = sum(adj[v] * adj, axis=1)
        
        common_u = np.sum(adj[u] * adj, axis=1)
        common_v = np.sum(adj[v] * adj, axis=1)
        
        # For each w:
        # Condition 1: If adj[v][w] is 1, we need common_u[w] <= 1.
        # Condition 2: If adj[u][w] is 1, we need common_v[w] <= 1.
        
        # Note: common_u[w] includes v if adj[u][v] was 1? No, we checked adj[u][v] is 0.
        # So common_u[w] is the count of neighbors of u that are also neighbors of w.
        
        # Mask for w where v is a neighbor
        mask_v = (adj[v] == 1)
        # Mask for w where u is a neighbor
        mask_u = (adj[u] == 1)
        
        # Exclude u and v themselves from w?
        # w iterates over all vertices.
        # If w == u:
        #   mask_v[u] is adj[v][u] which is 0 (since adj[u][v] is 0).
        #   mask_u[u] is adj[u][u] which is 0.
        #   So conditions are not triggered.
        # If w == v:
        #   mask_v[v] is adj[v][v] which is 0.
        #   mask_u[v] is adj[u][v] which is 0.
        #   So conditions are not triggered.
        
        # So we can just check all w.
        
        # Check 1: For all w where adj[v][w]==1, common_u[w] must be <= 1.
        # If any such w has common_u[w] >= 2, invalid.
        if np.any(common_u[mask_v] >= 2):
            return False
            
        # Check 2: For all w where adj[u][w]==1, common_v[w] must be <= 1.
        if np.any(common_v[mask_u] >= 2):
            return False
            
        return True

    def _add_edge(adj, u, v):
        adj[u][v] = 1
        adj[v][u] = 1

    def _remove_edge(adj, u, v):
        adj[u][v] = 0
        adj[v][u] = 0

    # Greedy construction with random restarts
    for _ in range(200): # Max restarts, but limited by time
        if time.time() > DEADLINE - MARGIN:
            break
            
        adj = np.zeros((N, N), dtype=int)
        
        # Shuffle pairs
        shuffled_pairs = pairs[:]
        random.shuffle(shuffled_pairs)
        
        for u, v in shuffled_pairs:
            if time.time() > DEADLINE - MARGIN:
                break
            if adj[u][v]:
                continue
            if _is_valid(adj, u, v):
                _add_edge(adj, u, v)
                
        n_edges = int(np.sum(adj) // 2)
        if n_edges > best_edges:
            best_edges = n_edges
            best_adj = adj.copy()
            
    # Local Search: Try to swap edges to improve
    # We will try to remove an edge and add a different one, or add an edge if possible.
    # Since we used a greedy that adds all possible edges in a random order, 
    # the graph is "maximal" with respect to that order. 
    # A local search can try to "fix" order dependencies.
    
    # A simpler improvement: Try to add any missing edge that is valid.
    # If we can't, try to remove one edge and see if we can add two?
    # Given the constraint is local (common neighbors), a 1-swap (remove 1, add 1) 
    # might not help if the graph is saturated.
    # However, a 2-swap (remove 2, add 2) might.
    
    # Let's do a quick local search:
    # Iterate over missing edges. If valid, add it.
    # If not valid, try to remove one incident edge to make it valid?
    
    # To save time and ensure correctness, I will stick to the greedy restarts 
    # which are robust. The greedy with random order is a standard heuristic.
    # For n=20, 200 restarts should be fast enough in NumPy.
    
    return _edge_list_from_adj(best_adj)
