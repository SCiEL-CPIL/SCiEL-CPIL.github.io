import time
import random
import sys

def construct():
    n = 16
    # Clebsch graph edges
    clebsch_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            xor_val = u ^ v
            dist = bin(xor_val).count('1')
            if dist == 1 or dist == 4:
                clebsch_edges.append([u, v])
    
    best_edges = clebsch_edges
    best_count = len(clebsch_edges)
    
    # Helper functions
    def is_triangle_free(edges):
        # Use adjacency matrix for speed
        adj = [[False] * n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u][v]:
                    for w in range(v + 1, n):
                        if adj[u][w] and adj[v][w]:
                            return False
        return True

    def is_3_colorable(edges):
        # Backtracking
        adj = [[False] * n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
        
        colors = [-1] * n
        
        def backtrack(v):
            if v == n:
                return True
            # Order neighbors by degree (optional optimization)
            for c in range(3):
                valid = True
                for u in range(v):
                    if adj[v][u] and colors[u] == c:
                        valid = False
                        break
                if valid:
                    colors[v] = c
                    if backtrack(v + 1):
                        return True
                    colors[v] = -1
            return False
        
        return backtrack(0)

    def try_improve(edges):
        # Randomly remove an edge and try to add others
        if not edges:
            return edges
        e = random.choice(edges)
        new_edges = [ed for ed in edges if ed != e]
        
        # Try to add random edges
        all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(all_pairs)
        
        current_edges = new_edges
        current_adj = [[False] * n for _ in range(n)]
        for u, v in current_edges:
            current_adj[u][v] = True
            current_adj[v][u] = True
            
        for u, v in all_pairs:
            if current_adj[u][v]:
                continue
            # Check if adding (u,v) creates a triangle
            creates_triangle = False
            for w in range(n):
                if current_adj[u][w] and current_adj[v][w]:
                    creates_triangle = True
                    break
            if not creates_triangle:
                current_edges.append([u, v])
                current_adj[u][v] = True
                current_adj[v][u] = True
                
        return current_edges

    deadline = time.time() + 110
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Start from best or random
        if attempts % 10 == 0:
            current = try_improve(best_edges)
        else:
            # Random restart
            current = []
            all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
            random.shuffle(all_pairs)
            for u, v in all_pairs:
                if is_triangle_free(current + [[u, v]]):
                    current.append([u, v])
                    if len(current) >= best_count:
                        break
        
        if len(current) > best_count:
            if not is_3_colorable(current):
                best_edges = current
                best_count = len(current)
                
    return best_edges
