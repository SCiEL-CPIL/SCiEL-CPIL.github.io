"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 115.0
    
    n = 40
    target_edges = 401
    
    # Initial Graph: K20,20 + 1 internal edge
    edges = set()
    for u in range(20):
        for v in range(20, 40):
            edges.add((u, v))
    edges.add((0, 1))
    
    def get_adjacency(edges_set):
        adj = [[0]*n for _ in range(n)]
        for u, v in edges_set:
            adj[u][v] = 1
            adj[v][u] = 1
        return adj

    def calc_max_book(edges_set):
        adj = get_adjacency(edges_set)
        max_b = 0
        for u, v in edges_set:
            count = 0
            row_u = adj[u]
            row_v = adj[v]
            for k in range(n):
                if row_u[k] and row_v[k]:
                    count += 1
            if count > max_b:
                max_b = count
        return max_b

    current_edges = set(edges)
    current_max = calc_max_book(current_edges)
    best_edges = set(current_edges)
    best_max = current_max
    
    temp = 5.0
    cooling = 0.9995
    
    while time.time() < deadline:
        if time.time() > deadline - 5:
            break
            
        # Find worst edge
        adj = get_adjacency(current_edges)
        worst_edge = None
        worst_count = -1
        for e in current_edges:
            a, b = e
            cnt = 0
            row_a = adj[a]
            row_b = adj[b]
            for k in range(n):
                if row_a[k] and row_b[k]:
                    cnt += 1
            if cnt > worst_count:
                worst_count = cnt
                worst_edge = e
        
        if worst_count <= 0:
            break
            
        a, b = worst_edge
        common_neighbors = [k for k in range(n) if adj[a][k] and adj[b][k]]
        
        if not common_neighbors:
            continue
            
        w = random.choice(common_neighbors)
        
        if random.random() < 0.5:
            remove_edge = (a, w) if a < w else (w, a)
        else:
            remove_edge = (b, w) if b < w else (w, b)
            
        if remove_edge not in current_edges:
            continue
            
        for _ in range(20):
            x = random.randint(0, n-1)
            y = random.randint(0, n-1)
            if x == y:
                continue
            new_edge = (x, y) if x < y else (y, x)
            if new_edge in current_edges:
                continue
                
            current_edges.remove(remove_edge)
            current_edges.add(new_edge)
            
            new_max = calc_max_book(current_edges)
            
            delta = new_max - current_max
            
            if delta <= 0 or random.random() < pow(2.718, -delta/temp):
                current_max = new_max
                if new_max < best_max:
                    best_max = new_max
                    best_edges = set(current_edges)
                break
            else:
                current_edges.remove(new_edge)
                current_edges.add(remove_edge)
                
        temp *= cooling
        
    return [list(e) for e in best_edges]
# EVOLVE-BLOCK-END
