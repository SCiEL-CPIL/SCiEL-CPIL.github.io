import time
import random
import numpy as np

def construct():
    n = 16
    m_target = 65
    deadline = time.time() + 110.0  # Safety margin for 120s limit
    
    best_graph = None
    best_score = float('inf')
    
    # Helper to compute score
    def compute_score(adj, deg):
        max_sum = 0
        # Iterate over edges
        # adj is n x n numpy array
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    # Find common neighbors
                    # adj[u] is a row, adj[v] is a row
                    # Common neighbors w such that adj[u,w] and adj[v,w] are 1
                    # w can be u or v? No, simple graph, no self loops, so adj[u,u]=0.
                    # Triangles are (u, v, w) with w != u, v.
                    common = np.where(adj[u] & adj[v])[0]
                    for w in common:
                        s = deg[u] + deg[v] + deg[w]
                        if s > max_sum:
                            max_sum = s
        return max_sum

    # Helper to initialize K_{8,8} + 1 edge
    def init_graph():
        adj = np.zeros((n, n), dtype=int)
        # K_{8,8}
        for u in range(8):
            for v in range(8, 16):
                adj[u, v] = 1
                adj[v, u] = 1
        # Add one random intra-part edge to get 65 edges
        # Randomly choose a part (0-7 or 8-15) and two distinct vertices
        part = random.choice([0, 1])
        if part == 0:
            u, v = random.sample(range(8), 2)
        else:
            u, v = random.sample(range(8, 16), 2)
        adj[u, v] = 1
        adj[v, u] = 1
        deg = adj.sum(axis=1)
        return adj, deg

    # Local search
    while time.time() < deadline:
        adj, deg = init_graph()
        current_score = compute_score(adj, deg)
        
        if current_score < best_score:
            best_score = current_score
            best_graph = adj.copy()
            if best_score <= 23: # Target reached?
                break

        no_improve = 0
        max_iters = 10000 # Cap per restart to avoid getting stuck
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick a random edge to remove
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u, v]:
                        edges.append((u, v))
            
            if not edges:
                break
            
            u, v = random.choice(edges)
            
            # Pick a random non-edge to add
            non_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if not adj[i, j]:
                        non_edges.append((i, j))
            
            if not non_edges:
                break
            
            u2, v2 = random.choice(non_edges)
            
            # Check if swap is valid (it always is for simple graph swap if u!=v and u2!=v2 and no self loops)
            # But we must ensure we don't create a multigraph (adj is 0/1, so adding 1 to 0 is fine)
            
            # Perform swap
            adj[u, v] = 0
            adj[v, u] = 0
            adj[u2, v2] = 1
            adj[v2, u2] = 1
            
            # Update degrees
            deg[u] -= 1
            deg[v] -= 1
            deg[u2] += 1
            deg[v2] += 1
            
            new_score = compute_score(adj, deg)
            
            if new_score < current_score:
                current_score = new_score
                no_improve = 0
                if current_score < best_score:
                    best_score = current_score
                    best_graph = adj.copy()
                    if best_score <= 23:
                        break
            else:
                # Revert
                adj[u, v] = 1
                adj[v, u] = 1
                adj[u2, v2] = 0
                adj[v2, u2] = 0
                
                deg[u] += 1
                deg[v] += 1
                deg[u2] -= 1
                deg[v2] -= 1
                
                no_improve += 1
                if no_improve > 500: # Restart if stuck
                    break
                    
        if best_score <= 23:
            break

    # Convert best_graph to edge list
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_graph[u, v]:
                edges.append([u, v])
                
    return {'n': n, 'edges': edges}
