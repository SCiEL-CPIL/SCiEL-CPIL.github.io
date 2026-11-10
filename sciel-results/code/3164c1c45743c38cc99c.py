import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85
    n = 15
    
    def count_edges(adj):
        return sum(sum(row) for row in adj) // 2
    
    def is_valid_fast(adj, degs):
        for u in range(n):
            for v in range(u+1, n):
                if not adj[u][v]:
                    continue
                for s in range(n):
                    if s == u or s == v or not adj[u][s]:
                        continue
                    ds = degs[s]
                    for t in range(n):
                        if t == u or t == v or t == s or not adj[v][t]:
                            continue
                        if ds == degs[t]:
                            return False
        return True
    
    def full_check(adj):
        degs = [sum(adj[i]) for i in range(n)]
        return is_valid_fast(adj, degs)
    
    # Start with complete graph and remove edges?
    # Or start with a good heuristic.
    # Let's try a random restart with local search.
    
    best_edges = 0
    best_adj = [[0]*n for _ in range(n)]
    
    random.seed(42)
    
    for restart in range(100):
        if time.time() > deadline - 5:
            break
            
        # Initialize with random graph
        adj = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i][j] = 1
                    adj[j][i] = 1
        
        degs = [sum(adj[i]) for i in range(n)]
        
        # Local search
        for step in range(5000):
            if time.time() > deadline - 2:
                break
                
            # Pick a random edge to toggle
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
                
            if adj[u][v]:
                # Try removing
                adj[u][v] = 0
                adj[v][u] = 0
                new_degs = degs[:]
                new_degs[u] -= 1
                new_degs[v] -= 1
                if is_valid_fast(adj, new_degs):
                    degs = new_degs
                    ec = count_edges(adj)
                    if ec > best_edges:
                        best_edges = ec
                        best_adj = [row[:] for row in adj]
                else:
                    # Revert
                    adj[u][v] = 1
                    adj[v][u] = 1
            else:
                # Try adding
                adj[u][v] = 1
                adj[v][u] = 1
                new_degs = degs[:]
                new_degs[u] += 1
                new_degs[v] += 1
                if is_valid_fast(adj, new_degs):
                    degs = new_degs
                    ec = count_edges(adj)
                    if ec > best_edges:
                        best_edges = ec
                        best_adj = [row[:] for row in adj]
                else:
                    # Revert
                    adj[u][v] = 0
                    adj[v][u] = 0
    
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i][j]:
                edges.append([i, j])
                
    return edges
