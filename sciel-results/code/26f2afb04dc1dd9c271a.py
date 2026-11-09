import random
import time

def construct():
    n = 40
    # Start with 8 disjoint K5s
    edges = set()
    for k in range(8):
        base = k * 5
        for a in range(5):
            for b in range(a + 1, 5):
                edges.add((base + a, base + b))
    
    # Generate all potential edges
    potential_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in edges:
                potential_edges.append((u, v))
    
    best_edges = set(edges)
    best_count = len(edges)
    
    deadline = time.time() + 110
    
    while time.time() < deadline:
        current_edges = set(best_edges)
        
        # Build adjacency list
        cur_adj = {i: set() for i in range(n)}
        for u, v in current_edges:
            cur_adj[u].add(v)
            cur_adj[v].add(u)
            
        random.shuffle(potential_edges)
        
        for u, v in potential_edges:
            if (u, v) in current_edges or (v, u) in current_edges:
                continue
            
            # Check if adding (u, v) creates a C6
            if not has_path_length_4(u, v, cur_adj):
                current_edges.add((u, v))
                cur_adj[u].add(v)
                cur_adj[v].add(u)
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges
            if best_count >= 105:
                break

    return [list(e) for e in best_edges]

def has_path_length_4(u, v, adj):
    visited = set()
    
    def dfs(node, depth):
        if depth == 4:
            return node == v
        visited.add(node)
        for neighbor in adj[node]:
            if neighbor not in visited:
                if dfs(neighbor, depth + 1):
                    return True
        visited.remove(node)
        return False
    
    return dfs(u, 0)
