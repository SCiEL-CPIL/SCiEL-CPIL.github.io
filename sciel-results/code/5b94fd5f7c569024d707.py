import random
import time

def construct():
    n = 40
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Helper functions
    def has_path_3(a, b, adj, N):
        for x in N[a]:
            for y in N[b]:
                if adj[x][y]:
                    return True
        return False

    def creates_c6(u, v, adj, N):
        for w1 in N[u]:
            for w4 in N[v]:
                if has_path_3(w1, w4, adj, N):
                    return True
        return False

    def get_edge_count(adj):
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    count += 1
        return count

    def extract_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    edges.append([i, j])
        return edges

    # Initialize with seed graph (8 copies of K5)
    adj = [[0]*n for _ in range(n)]
    N = [[] for _ in range(n)]
    
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a+1, 5):
                u = base + a
                v = base + b
                adj[u][v] = 1
                adj[v][u] = 1
                N[u].append(v)
                N[v].append(u)
                
    best_edges = extract_edges(adj)
    best_count = get_edge_count(adj)
    
    # Search Loop
    while time.time() < deadline:
        # 1. Greedy Saturation
        # Try to add all possible edges
        # We can randomize the order of checks to avoid bias
        # But a full pass is fast.
        
        # To make it more effective, we can try adding edges in a random order
        # and stop if no more can be added? 
        # No, adding one edge might block another, but it won't unblock a previously rejected one.
        # So a single pass through all non-edges is sufficient for saturation.
        
        for i in range(n):
            for j in range(i+1, n):
                if not adj[i][j]:
                    if not creates_c6(i, j, adj, N):
                        adj[i][j] = 1
                        adj[j][i] = 1
                        N[i].append(j)
                        N[j].append(i)
                        
        current_count = get_edge_count(adj)
        if current_count > best_count:
            best_count = current_count
            best_edges = extract_edges(adj)
            
        # 2. Perturbation
        # Remove some edges to escape local optimum
        # Number of edges to remove: 1 to 10
        num_removals = random.randint(1, 10)
        
        all_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    all_edges.append((i, j))
                    
        if len(all_edges) < num_removals:
            num_removals = len(all_edges)
            
        if num_removals > 0:
            to_remove = random.sample(all_edges, num_removals)
            for u, v in to_remove:
                adj[u][v] = 0
                adj[v][u] = 0
                N[u].remove(v)
                N[v].remove(u)

    return best_edges
