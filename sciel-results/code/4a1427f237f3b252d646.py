"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 16}

def construct():
    n = P['n']
    start_time = time.time()
    deadline = start_time + 80
    
    # Start with a star (known valid, 15 edges)
    best_edges = [[0, v] for v in range(1, n)]
    best_count = len(best_edges)
    
    def is_valid(edges):
        m = len(edges)
        if m < n - 1:
            return False
        
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        def bfs_dist(start, adj_list):
            dist = [-1] * n
            dist[start] = 0
            queue = [start]
            for node in queue:
                for nb in adj_list[node]:
                    if dist[nb] == -1:
                        dist[nb] = dist[node] + 1
                        queue.append(nb)
            return dist
        
        # Check diameter is exactly 2
        for i in range(n):
            d = bfs_dist(i, adj)
            if -1 in d:
                return False
            if max(d) > 2:
                return False
        # Check diameter >= 2 (not a complete graph)
        if all(max(bfs_dist(i, adj)) <= 1 for i in range(n)):
            return False
        
        # Check each edge is critical
        for u, v in edges:
            adj[u].remove(v)
            adj[v].remove(u)
            
            critical = False
            for i in range(n):
                d = bfs_dist(i, adj)
                if -1 in d or max(d) > 2:
                    critical = True
                    break
            
            adj[u].add(v)
            adj[v].add(u)
            
            if not critical:
                return False
        
        return True
    
    current_edges = set((min(u,v), max(u,v)) for u, v in best_edges)
    
    all_possible = [(u, v) for u in range(n) for v in range(u+1, n)]
    random.seed(42)
    random.shuffle(all_possible)
    
    for u, v in all_possible:
        if time.time() > deadline:
            break
        if (u, v) in current_edges:
            continue
        
        new_edges = [[a, b] for a, b in current_edges] + [[u, v]]
        if is_valid(new_edges):
            current_edges.add((u, v))
            best_edges = new_edges
            best_count = len(new_edges)
    
    return [[int(u), int(v)] for u, v in best_edges]
# EVOLVE-BLOCK-END
