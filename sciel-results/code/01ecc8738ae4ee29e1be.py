"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill
    
    n = 20
    
    # We want to maximize edges in a K2,3-free graph on 20 vertices.
    # K2,3-free means no two vertices have 3 or more common neighbors.
    # Equivalently, for every pair of vertices (u, v), |N(u) ∩ N(v)| <= 2.
    
    # The seed solution is 5 disjoint K4s, which has 5 * 6 = 30 edges.
    # We can do better by mixing vertices between cliques or using other structures.
    
    # A known strong construction for K2,3-free graphs is related to polarity graphs or
    # specific regular graphs. However, for n=20, we can use local search to improve.
    
    # Representation: adjacency matrix (boolean) for fast common neighbor counting.
    # adj[i][j] = 1 if edge (i,j) exists.
    
    def count_common_neighbors(adj, u, v):
        # Count common neighbors of u and v
        # Using numpy for speed
        return np.sum(adj[u] & adj[v])
    
    def is_valid(adj):
        # Check if graph is K2,3-free
        for u in range(n):
            for v in range(u + 1, n):
                if count_common_neighbors(adj, u, v) > 2:
                    return False
        return True
    
    def edge_count(adj):
        return np.sum(adj) // 2
    
    def get_edges_from_adj(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u][v]:
                    edges.append([u, v])
        return edges
    
    # Initialize with seed solution: 5 disjoint K4s
    def init_seed():
        adj = np.zeros((n, n), dtype=bool)
        for k in range(5):
            for a in range(4):
                for b in range(a + 1, 4):
                    u = 4 * k + a
                    v = 4 * k + b
                    adj[u][v] = True
                    adj[v][u] = True
        return adj
    
    best_adj = init_seed()
    best_score = edge_count(best_adj)
    
    # Local search: try adding edges, removing edges if conflicts arise
    # We use a greedy approach with local search:
    # 1. Try to add all possible edges that don't create K2,3
    # 2. If stuck, try removing an edge to allow more additions
    
    def try_add_edges(adj):
        # Greedily add edges in random order
        possible_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if not adj[u][v]:
                    possible_edges.append((u, v))
        
        random.shuffle(possible_edges)
        
        for u, v in possible_edges:
            # Check if adding (u,v) creates K2,3
            # Adding (u,v) means:
            # - For any w, if w is common neighbor of u and v, that's fine as long as count <= 2
            # - But we need to check all pairs (x, y) where x is in N(u) or y is in N(v)
            # Actually, adding (u,v) affects:
            # - Pair (u, v): common neighbors = |N(u) ∩ N(v)| must be <= 2
            # - For any w in N(u) \ N(v): pair (w, v) now has common neighbors including u
            #   New common neighbors of (w, v) = (N(w) ∩ N(v)) ∪ {u} if u in N(w) and u not in N(v)
            #   Wait, u is in N(w) (since w in N(u)) and u is now in N(v) (since we add (u,v))
            #   So u becomes a common neighbor of (w, v)
            #   Similarly for any w in N(v) \ N(u): pair (u, w) gains v as common neighbor
            
            # Let's check carefully:
            # After adding (u,v):
            # 1. Pair (u,v): |N(u) ∩ N(v)| must be <= 2
            # 2. For each w in N(u) \ N(v): pair (w,v) gains u as common neighbor
            #    So |N(w) ∩ N(v) ∪ {u}| = |N(w) ∩ N(v)| + 1 (since u not in N(v) before)
            #    This must be <= 2, so |N(w) ∩ N(v)| <= 1
            # 3. For each w in N(v) \ N(u): pair (u,w) gains v as common neighbor
            #    So |N(u) ∩ N(w) ∪ {v}| = |N(u) ∩ N(w)| + 1
            #    This must be <= 2, so |N(u) ∩ N(w)| <= 1
            # 4. For w in N(u) ∩ N(v): pairs (w,u) and (w,v) already had u and v as neighbors
            #    But adding (u,v) doesn't change common neighbors of (w,u) or (w,v)
            #    Wait, common neighbors of (w,u) are N(w) ∩ N(u). Adding (u,v) doesn't change N(w) or N(u)
            #    So no change for those pairs.
            
            # So conditions to add (u,v):
            # a) |N(u) ∩ N(v)| <= 2
            # b) For all w in N(u) \ N(v): |N(w) ∩ N(v)| <= 1
            # c) For all w in N(v) \ N(u): |N(u) ∩ N(w)| <= 1
            
            nu = np.where(adj[u])[0]
            nv = np.where(adj[v])[0]
            
            # Condition a
            common_uv = np.sum(adj[u] & adj[v])
            if common_uv > 2:
                continue
            
            # Condition b: for w in N(u) not in N(v)
            nu_not_nv = np.setdiff1d(nu, nv)
            ok = True
            for w in nu_not_nv:
                common_wv = np.sum(adj[w] & adj[v])
                if common_wv > 1:
                    ok = False
                    break
            if not ok:
                continue
            
            # Condition c: for w in N(v) not in N(u)
            nv_not_nu = np.setdiff1d(nv, nu)
            for w in nv_not_nu:
                common_uw = np.sum(adj[u] & adj[w])
                if common_uw > 1:
                    ok = False
                    break
            if not ok:
                continue
            
            # All conditions met, add edge
            adj[u][v] = True
            adj[v][u] = True
        
        return adj
    
    # Start with seed and try to add edges
    adj = best_adj.copy()
    adj = try_add_edges(adj)
    current_score = edge_count(adj)
    if current_score > best_score:
        best_score = current_score
        best_adj = adj.copy()
    
    # Now try local search with restarts
    # If we get stuck, try removing an edge to unlock more additions
    
    for iteration in range(200):
        if time.time() > deadline:
            break
        
        adj = best_adj.copy()
        
        # Randomly remove an edge to create space
        if np.sum(adj) > 0:
            # Pick a random existing edge
            edges_list = []
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u][v]:
                        edges_list.append((u, v))
            if edges_list:
                u, v = random.choice(edges_list)
                adj[u][v] = False
                adj[v][u] = False
        
        # Try to add edges
        adj = try_add_edges(adj)
        current_score = edge_count(adj)
        
        if current_score > best_score:
            best_score = current_score
            best_adj = adj.copy()
    
    return get_edges_from_adj(best_adj)
# EVOLVE-BLOCK-END

