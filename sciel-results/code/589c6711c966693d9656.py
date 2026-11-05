import time
import random

def construct():
    n = 16
    target_edges = 65
    
    # Start with K_8,8 + (0,1)
    edges = set()
    for u in range(8):
        for v in range(8, 16):
            edges.add((u, v))
    edges.add((0, 1))
    
    def evaluate(edge_set):
        deg = [0] * n
        for u, v in edge_set:
            deg[u] += 1
            deg[v] += 1
            
        adj = [[0]*n for _ in range(n)]
        for u, v in edge_set:
            adj[u][v] = 1
            adj[v][u] = 1
            
        max_sum = 0
        worst_tri = None
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v]:
                    for w in range(v+1, n):
                        if adj[u][w] and adj[v][w]:
                            s = deg[u] + deg[v] + deg[w]
                            if s > max_sum:
                                max_sum = s
                                worst_tri = (u, v, w)
        return max_sum, worst_tri, deg, adj

    best_score, best_tri, best_deg, best_adj = evaluate(edges)
    best_edges = set(edges)
    current_edges = set(edges)
    current_score = best_score
    
    start_time = time.time()
    timeout = 110
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    while time.time() - start_time < timeout:
        # Strategy: Try to improve by removing an edge from the worst triangle
        # and adding a "safe" edge.
        
        if not best_tri:
            break
            
        u, v, w = best_tri
        # Candidate edges to remove: (u,v), (v,w), (u,w)
        candidate_removals = [(u,v), (v,w), (u,w)]
        random.shuffle(candidate_removals)
        
        improved = False
        for e_rem in candidate_removals:
            if e_rem not in current_edges:
                continue
                
            new_edges = set(current_edges)
            new_edges.remove(e_rem)
            
            # Try to add a random edge
            # To be efficient, we can try a few random additions
            for _ in range(10):
                e_add = random.choice(all_edges)
                if e_add in new_edges or e_add == e_rem:
                    continue
                
                new_edges.add(e_add)
                
                # Quick check: does this create a triangle with high sum?
                # We can just evaluate fully if promising
                # But full evaluation is O(n^3).
                # Let's just evaluate.
                new_score, new_tri, _, _ = evaluate(new_edges)
                
                if new_score < current_score:
                    current_score = new_score
                    current_edges = new_edges
                    if current_score < best_score:
                        best_score = current_score
                        best_edges = set(current_edges)
                    improved = True
                    break # Restart search with new current state
            if improved:
                break
                
        if not improved:
            # If no improvement from worst triangle, try random moves
            # Or accept a slightly worse move to escape local minimum (Simulated Annealing)
            # For simplicity, just try random swaps
            for _ in range(5):
                e_rem = random.choice(list(current_edges))
                e_add = random.choice(all_edges)
                if e_add in current_edges or e_add == e_rem:
                    continue
                    
                new_edges = set(current_edges)
                new_edges.remove(e_rem)
                new_edges.add(e_add)
                
                new_score, new_tri, _, _ = evaluate(new_edges)
                
                # Accept if better, or with probability exp(-delta/T)
                # Here we just accept if better to keep it simple hill climbing
                if new_score < current_score:
                    current_score = new_score
                    current_edges = new_edges
                    if current_score < best_score:
                        best_score = current_score
                        best_edges = set(current_edges)
                    break

    res_edges = [[u, v] for u, v in best_edges]
    return {'n': n, 'edges': res_edges}
