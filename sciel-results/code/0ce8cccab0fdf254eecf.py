import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    n = 20
    # Adjacency matrix for fast lookup
    # adj[u][v] = 1 if edge exists
    
    def is_k33_free(adj):
        # This is not needed for incremental check, but good for verification if we had one
        pass

    def can_add_edge(adj, u, v):
        # Check if adding (u,v) creates a K3,3
        # u in A, v in B
        # Need a1, a2 in N(v)\{u} and b1, b2 in N(u)\{v}
        # such that a1,a2 connected to b1,b2 in G
        
        N_v = [x for x in range(n) if adj[v][x] and x != u]
        N_u = [x for x in range(n) if adj[u][x] and x != v]
        
        # If either set is too small, no K3,3 can be formed using (u,v) in this orientation
        if len(N_v) < 2 or len(N_u) < 2:
            return True
            
        # Check for K2,2 between N_v and N_u
        # For each pair b1, b2 in N_u, check if there are a1, a2 in N_v connected to both
        # Optimization: For each b in N_u, let C_b = {a in N_v | adj[a][b]}
        # We need two b's, b1, b2, such that |C_{b1} intersect C_{b2}| >= 2
        
        # Precompute common neighbors for pairs in N_u?
        # n is small, just loop.
        
        # Sort N_u by degree to N_v?
        # For b in N_u, count connections to N_v
        # If sum of top 2 counts < 2 * 2, maybe impossible? No, just check.
        
        # Simple check:
        # For each pair (b1, b2) in N_u:
        #   common = [a for a in N_v if adj[a][b1] and adj[a][b2]]
        #   if len(common) >= 2: return False
        
        # Number of pairs in N_u is at most C(18, 2) = 153.
        # For each pair, scan N_v (size <= 18).
        # 153 * 18 = ~2754 ops. Very fast.
        
        n_u_list = N_u
        n_v_set = set(N_v) # for faster lookup if needed, but list is fine
        
        # To speed up, sort n_u_list by number of neighbors in N_v descending
        # and break early if possible?
        
        for i in range(len(n_u_list)):
            b1 = n_u_list[i]
            for j in range(i+1, len(n_u_list)):
                b2 = n_u_list[j]
                count = 0
                for a in N_v:
                    if adj[a][b1] and adj[a][b2]:
                        count += 1
                        if count >= 2:
                            return False
        return True

    best_edges = 36
    best_graph = None
    
    # Initialize best graph with seed
    seed_adj = [[0]*n for _ in range(n)]
    for a in range(2):
        for b in range(2, 20):
            seed_adj[a][b] = 1
            seed_adj[b][a] = 1
    best_graph = seed_adj
    
    # Local Search
    # Try multiple random orders of adding edges
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        adj = [[0]*n for _ in range(n)]
        
        # Start with a random sparse graph or seed?
        # Let's start with seed and shuffle the missing edges
        # Or start with random p=0.5 and try to add?
        # Starting with seed is safer.
        
        # Copy seed
        for i in range(n):
            for j in range(n):
                adj[i][j] = seed_adj[i][j]
                
        missing_edges = []
        for u in range(n):
            for v in range(u+1, n):
                if not adj[u][v]:
                    missing_edges.append((u, v))
        
        random.shuffle(missing_edges)
        
        current_edges = 36
        for u, v in missing_edges:
            if time.time() > deadline:
                break
            if can_add_edge(adj, u, v):
                adj[u][v] = 1
                adj[v][u] = 1
                current_edges += 1
                
        if current_edges > best_edges:
            best_edges = current_edges
            best_graph = [row[:] for row in adj]
            
        # If we found a very good one, maybe we can stop early?
        # ex(20, K3,3) is likely around 45-50.
        # If we hit 50, we might be close to optimal.
        if best_edges >= 50:
            break
            
    # Convert best_graph to list of edges
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_graph[u][v]:
                edges.append([u, v])
                
    return edges
