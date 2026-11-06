import random
import time

def construct():
    n = 20
    colors = 20
    deadline = time.time() + 110
    
    best_edges = 0
    best_solution = None
    
    def generate_c4_free():
        adj = [[False] * n for _ in range(n)]
        potential_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(potential_edges)
        
        for u, v in potential_edges:
            if time.time() > deadline:
                break
            creates_c4 = False
            for a in range(n):
                if u == a or v == a: continue
                if adj[u][a]:
                    for b in range(n):
                        if a == b or v == b: continue
                        if adj[a][b] and adj[b][v]:
                            creates_c4 = True
                            break
                if creates_c4:
                    break
            if not creates_c4:
                adj[u][v] = True
                adj[v][u] = True
        return adj

    for restart in range(100):
        if time.time() > deadline:
            break
        adj = generate_c4_free()
        edge_count = sum(sum(row) for row in adj) // 2
        if edge_count < best_edges:
            continue
            
        col = [[-1] * n for _ in range(n)]
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v]:
                    col[u][v] = 0
                    col[v][u] = 0
                    
        missing_edges = [(u, v) for u in range(n) for v in range(u+1, n) if not adj[u][v]]
        random.shuffle(missing_edges)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Reshuffle missing edges to avoid bias
            random.shuffle(missing_edges)
            for u, v in missing_edges:
                if adj[u][v]:
                    continue
                valid = True
                valid_colors = set(range(colors))
                
                # Check all paths u-a-b-v
                for a in range(n):
                    if u == a or v == a: continue
                    if adj[u][a]:
                        for b in range(n):
                            if a == b or v == b: continue
                            if adj[a][b] and adj[b][v]:
                                c1 = col[u][a]
                                c2 = col[a][b]
                                c3 = col[b][v]
                                
                                # Check if path colors are distinct
                                if c1 == c2 or c2 == c3 or c1 == c3:
                                    valid = False
                                    break
                                    
                                valid_colors.discard(c1)
                                valid_colors.discard(c2)
                                valid_colors.discard(c3)
                if not valid:
                    continue
                if not valid_colors:
                    continue
                    
                c = random.choice(list(valid_colors))
                adj[u][v] = True
                adj[v][u] = True
                col[u][v] = c
                col[v][u] = c
                improved = True
        
        final_edge_count = sum(sum(row) for row in adj) // 2
        if final_edge_count > best_edges:
            best_edges = final_edge_count
            result = []
            for u in range(n):
                for v in range(u+1, n):
                    if adj[u][v]:
                        result.append([u, v, col[u][v]])
            best_solution = result

    return best_solution

