import time
import random

def construct():
    start_time = time.time()
    time_limit = 140.0
    n = 16

    def is_bipartite(adj_sets):
        color = [-1] * n
        for start in range(n):
            if color[start] != -1:
                continue
            color[start] = 0
            queue = [start]
            while queue:
                u = queue.pop()
                for v in adj_sets[u]:
                    if color[v] == -1:
                        color[v] = color[u] ^ 1
                        queue.append(v)
                    elif color[v] == color[u]:
                        return False
        return True

    def is_3_colorable(adj_sets):
        if is_bipartite(adj_sets):
            return True
            
        vertices = sorted(range(n), key=lambda x: len(adj_sets[x]), reverse=True)
        colors = [-1] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            v = vertices[idx]
            available = {0, 1, 2}
            for u in adj_sets[v]:
                if colors[u] != -1:
                    available.discard(colors[u])
            
            if not available:
                return False
            
            for c in available:
                colors[v] = c
                if backtrack(idx + 1):
                    return True
                colors[v] = -1
            return False
        
        return backtrack(0)

    def can_add_edge(adj_sets, u, v):
        if u in adj_sets[v]:
            return False
        return len(adj_sets[u].intersection(adj_sets[v])) == 0

    initial_edges = [
        (0,1), (1,2), (2,3), (3,4), (4,0),
        (0,9), (0,6), (1,5), (1,7), (2,6), (2,8), (3,7), (3,9), (4,8), (4,5),
        (5,10), (6,10), (7,10), (8,10), (9,10)
    ]
    
    best_edges = initial_edges.copy()
    best_count = len(best_edges)
    
    def build_adjacency(edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    while time.time() - start_time < time_limit:
        current_edges = best_edges.copy()
        adj = build_adjacency(current_edges)
        
        all_possible = [(i, j) for i in range(n) for j in range(i+1, n) if (i, j) not in current_edges and (j, i) not in current_edges]
        random.shuffle(all_possible)
        
        added_any = False
        for u, v in all_possible:
            if time.time() - start_time > time_limit:
                break
            
            if can_add_edge(adj, u, v):
                adj[u].add(v)
                adj[v].add(u)
                
                if not is_3_colorable(adj):
                    current_edges.append((u, v))
                    added_any = True
                else:
                    adj[u].remove(v)
                    adj[v].remove(u)
        
        if len(current_edges) > best_count:
            best_edges = current_edges
            best_count = len(best_edges)
            
        if not added_any:
            if len(current_edges) > 0:
                idx = random.randint(0, len(current_edges)-1)
                current_edges.pop(idx)
                adj = build_adjacency(current_edges)
                
                all_possible = [(i, j) for i in range(n) for j in range(i+1, n) if (i, j) not in current_edges and (j, i) not in current_edges]
                random.shuffle(all_possible)
                
                for u2, v2 in all_possible:
                    if time.time() - start_time > time_limit:
                        break
                    if can_add_edge(adj, u2, v2):
                        adj[u2].add(v2)
                        adj[v2].add(u2)
                        if not is_3_colorable(adj):
                            current_edges.append((u2, v2))
                
                if len(current_edges) > best_count:
                    best_edges = current_edges
                    best_count = len(best_edges)

    return [[u, v] for u, v in best_edges]
