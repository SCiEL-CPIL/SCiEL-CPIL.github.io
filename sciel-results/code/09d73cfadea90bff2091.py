import time
import random

def construct():
    start_time = time.time()
    deadline = 115.0
    n = 40
    best_edges = []
    best_count = 0
    
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    while time.time() < deadline:
        random.shuffle(all_edges)
        adj = [[] for _ in range(n)]
        current_edges = []
        
        for u, v in all_edges:
            # Check if u and v are already connected
            # We can maintain an adjacency matrix for O(1) check
            # But adj list is fine for DFS
            
            # Check if adding (u,v) creates C6
            # We need to check if there is a path of length 5 between u and v in the current graph
            
            # Optimization: If u and v are not in the same connected component, 
            # or if the distance is > 5, no C6.
            # But we don't track components. 
            # Just run DFS.
            
            # To speed up, we can check if degree is 0
            if not adj[u] or not adj[v]:
                # If degree is 0, no path exists.
                # But if degree is 1, path might exist.
                pass
            
            # Run DFS
            visited = [False] * n
            visited[u] = True
            
            def dfs(node, depth):
                if depth == 5:
                    return node == v
                for nei in adj[node]:
                    if not visited[nei]:
                        visited[nei] = True
                        if dfs(nei, depth + 1):
                            return True
                        visited[nei] = False
                return False
            
            if not dfs(u, 0):
                # Safe to add
                adj[u].append(v)
                adj[v].append(u)
                current_edges.append([u, v])
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges
            
    return best_edges
