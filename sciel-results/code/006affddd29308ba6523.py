import time
import random
import itertools

def construct():
    n = 16
    target_edges = 65
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    def get_degrees(adj):
        return [sum(row) for row in adj]
    
    def compute_score(adj, degs):
        max_sum = 0
        for u, v, w in itertools.combinations(range(n), 3):
            if adj[u][v] and adj[v][w] and adj[w][u]:
                s = degs[u] + degs[v] + degs[w]
                if s > max_sum:
                    max_sum = s
        return max_sum

    def init_graph():
        adj = [[0]*n for _ in range(n)]
        # K_{8,8}
        for u in range(8):
            for v in range(8, 16):
                adj[u][v] = 1
                adj[v][u] = 1
        # Add random internal edge
        u, v = random.sample(range(8), 2)
        adj[u][v] = 1
        adj[v][u] = 1
        return adj

    best_score = float('inf')
    best_adj = None
    current_adj = init_graph()
    current_degs = get_degrees(current_adj)
    current_score = compute_score(current_adj, current_degs)
    
    if current_score < best_score:
        best_score = current_score
        best_adj = [row[:] for row in current_adj]
        
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        # Find worst triangles
        worst_triangles = []
        for u, v, w in itertools.combinations(range(n), 3):
            if current_adj[u][v] and current_adj[v][w] and current_adj[w][u]:
                s = current_degs[u] + current_degs[v] + current_degs[w]
                if s >= best_score:
                    worst_triangles.append((s, u, v, w))
        
        if worst_triangles:
            worst_triangles.sort(reverse=True)
            s, u, v, w = worst_triangles[0]
            edge_to_remove = random.choice([(u, v), (v, w), (w, u)])
        else:
            # Random edge removal
            edges = [(u, v) for u in range(n) for v in range(u+1, n) if current_adj[u][v]]
            if not edges:
                break
            edge_to_remove = random.choice(edges)
            
        # Find a non-edge to add
        # Try a few random non-edges
        added = False
        for _ in range(10):
            u_a, v_a = random.sample(range(n), 2)
            if not current_adj[u_a][v_a]:
                # Check if it's the edge we removed
                if (u_a == edge_to_remove[0] and v_a == edge_to_remove[1]) or (u_a == edge_to_remove[1] and v_a == edge_to_remove[0]):
                    continue
                break
        else:
            continue
            
        u_r, v_r = edge_to_remove
        u_a, v_a = edge_to_add if 'edge_to_add' in locals() else (u_a, v_a)
        # Fix variable scope issue
        edge_to_add = (u_a, v_a)
        
        candidate_adj = [row[:] for row in current_adj]
        candidate_adj[u_r][v_r] = 0
        candidate_adj[v_r][u_r] = 0
        candidate_adj[u_a][v_a] = 1
        candidate_adj[v_a][u_a] = 1
        
        candidate_degs = get_degrees(candidate_adj)
        candidate_score = compute_score(candidate_adj, candidate_degs)
        
        if candidate_score < best_score:
            best_score = candidate_score
            best_adj = [row[:] for row in candidate_adj]
            current_adj = [row[:] for row in candidate_adj]
            current_degs = candidate_degs
            current_score = candidate_score
            no_improve_count = 0
        else:
            no_improve_count += 1
            # Accept if equal to allow movement
            if candidate_score == current_score:
                current_adj = [row[:] for row in candidate_adj]
                current_degs = candidate_degs
                current_score = candidate_score
            elif no_improve_count > max_no_improve:
                # Restart
                current_adj = init_graph()
                current_degs = get_degrees(current_adj)
                current_score = compute_score(current_adj, current_degs)
                if current_score < best_score:
                    best_score = current_score
                    best_adj = [row[:] for row in current_adj]
                no_improve_count = 0

    final_edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u][v]:
                final_edges.append([u, v])
                
    return {'n': n, 'edges': final_edges}
