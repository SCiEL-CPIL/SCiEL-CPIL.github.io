import time
import random
import numpy as np

def construct():
    n = 16
    start_time = time.time()
    deadline = start_time + 110.0
    
    def init_graph():
        adj = np.zeros((n, n), dtype=bool)
        # K_{8,8}
        for u in range(8):
            for v in range(8, 16):
                adj[u, v] = True
                adj[v, u] = True
        # One extra edge
        adj[0, 1] = True
        adj[1, 0] = True
        return adj

    def get_score(adj):
        degs = np.sum(adj, axis=1)
        max_sum = 0
        for i in range(n):
            for j in range(i+1, n):
                if not adj[i, j]: continue
                for k in range(j+1, n):
                    if adj[i, k] and adj[j, k]:
                        s = degs[i] + degs[j] + degs[k]
                        if s > max_sum:
                            max_sum = s
        return max_sum

    adj = init_graph()
    edge_set = set()
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j]:
                edge_set.add((i, j))
                
    current_score = get_score(adj)
    best_score = current_score
    best_adj = adj.copy()
    
    temperature = 5.0
    
    while time.time() < deadline:
        # Find worst triangle to target
        degs = np.sum(adj, axis=1)
        worst_tri = None
        max_s = 0
        for i in range(n):
            for j in range(i+1, n):
                if not adj[i, j]: continue
                for k in range(j+1, n):
                    if adj[i, k] and adj[j, k]:
                        s = degs[i] + degs[j] + degs[k]
                        if s > max_s:
                            max_s = s
                            worst_tri = (i, j, k)
                            
        if worst_tri is None:
            break
            
        i, j, k = worst_tri
        tri_edges = [(i, j), (i, k), (j, k)]
        random.shuffle(tri_edges)
        
        improved = False
        for u, v in tri_edges:
            if (u, v) not in edge_set:
                continue
                
            # Try to swap (u,v) with a random edge
            for _ in range(30):
                x = random.randint(0, n-1)
                y = random.randint(0, n-1)
                if x == y: continue
                if x > y: x, y = y, x
                if (x, y) in edge_set: continue
                
                # Swap
                adj[u, v] = False
                adj[v, u] = False
                adj[x, y] = True
                adj[y, x] = True
                
                new_score = get_score(adj)
                
                if new_score < max_s or (temperature > 0.05 and random.random() < np.exp(-(new_score - max_s) / temperature)):
                    edge_set.remove((u, v))
                    edge_set.add((x, y))
                    max_s = new_score
                    improved = True
                    break
                else:
                    adj[u, v] = True
                    adj[v, u] = True
                    adj[x, y] = False
                    adj[y, x] = False
                    
            if improved:
                break
                
        if not improved:
            # If no improvement, maybe do a random swap to escape
            if random.random() < 0.1:
                if edge_set:
                    u, v = random.choice(list(edge_set))
                    # Find a non-edge
                    for _ in range(50):
                        x = random.randint(0, n-1)
                        y = random.randint(0, n-1)
                        if x == y: continue
                        if x > y: x, y = y, x
                        if (x, y) in edge_set: continue
                        
                        adj[u, v] = False
                        adj[v, u] = False
                        adj[x, y] = True
                        adj[y, x] = True
                        
                        new_score = get_score(adj)
                        if new_score <= max_s: # Accept if not worse
                            edge_set.remove((u, v))
                            edge_set.add((x, y))
                            max_s = new_score
                        else:
                            adj[u, v] = True
                            adj[v, u] = True
                            adj[x, y] = False
                            adj[y, x] = False
                        break

        if max_s < best_score:
            best_score = max_s
            best_adj = adj.copy()
            
        temperature *= 0.999
        if temperature < 0.01:
            temperature = 0.01

    # Reconstruct edges
    final_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i, j]:
                final_edges.append([i, j])
                
    return {'n': n, 'edges': final_edges}
