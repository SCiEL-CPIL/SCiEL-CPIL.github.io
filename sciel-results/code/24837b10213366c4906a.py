import random
import time

def construct():
    n = 40
    # Initialize with 8 disjoint K5s
    edges = []
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                edges.append([5 * k + a, 5 * k + b])
    
    adj = [[] for _ in range(n)]
    adj_mat = [[False]*n for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        adj_mat[u][v] = True
        adj_mat[v][u] = True
        
    best_edges = len(edges)
    
    def can_add_edge(u, v):
        stack = [(u, 0, 1 << u)]
        target = v
        
        while stack:
            node, depth, visited_mask = stack.pop()
            
            if depth == 5:
                if node == target:
                    return True
                continue
            
            for next_node in adj[node]:
                nm = 1 << next_node
                if visited_mask & nm:
                    continue
                
                if next_node == target:
                    if depth == 5:
                        return True
                    else:
                        continue
                
                stack.append((next_node, depth + 1, visited_mask | nm))
        
        return False

    start_time = time.time()
    timeout = 115
    
    while time.time() - start_time < timeout:
        added = False
        attempts = 0
        max_attempts = 2000 # Increased attempts per phase
        
        while attempts < max_attempts and not added:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v or adj_mat[u][v]:
                attempts += 1
                continue
            
            if can_add_edge(u, v):
                adj[u].append(v)
                adj[v].append(u)
                adj_mat[u][v] = True
                adj_mat[v][u] = True
                edges.append([u, v])
                if len(edges) > best_edges:
                    best_edges = len(edges)
                added = True
            attempts += 1
        
        if not added:
            if len(edges) > 0:
                idx = random.randint(0, len(edges) - 1)
                u, v = edges[idx]
                edges.pop(idx)
                adj[u].remove(v)
                adj[v].remove(u)
                adj_mat[u][v] = False
                adj_mat[v][u] = False

    return edges
