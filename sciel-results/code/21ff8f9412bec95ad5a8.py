import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145  # seconds
    
    n = 16
    m_target = 65
    
    # Initialize with K_8,8 + 1 edge (0, 1)
    # Vertices 0-7 in A, 8-15 in B
    adj = np.zeros((n, n), dtype=int)
    
    for u in range(8):
        for v in range(8, 16):
            adj[u, v] = 1
            adj[v, u] = 1
            
    adj[0, 1] = 1
    adj[1, 0] = 1
    
    deg = np.array([sum(adj[i]) for i in range(n)])
    
    # Precompute all possible pairs
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def get_triangles(adj, deg):
        tris = []
        for u in range(n):
            for v in range(u+1, n):
                if adj[u, v]:
                    for w in range(v+1, n):
                        if adj[u, w] and adj[v, w]:
                            tris.append((u, v, w))
        return tris

    def calc_score(tris, deg):
        if not tris:
            return 0
        max_s = 0
        for u, v, w in tris:
            s = deg[u] + deg[v] + deg[w]
            if s > max_s:
                max_s = s
        return max_s

    current_tris = get_triangles(adj, deg)
    current_score = calc_score(current_tris, deg)
    
    best_score = current_score
    best_adj = adj.copy()
    
    # ILS Parameters
    T = 1.0
    cooling = 0.9995
    
    # We will perform many iterations. 
    # The bottleneck is usually finding valid moves. 
    # We maintain lists of edges and non-edges for O(1) random selection.
    
    current_edges = set()
    current_non_edges = set()
    for u, v in all_pairs:
        if adj[u, v]:
            current_edges.add((u, v))
        else:
            current_non_edges.add((u, v))
            
    edges_list = list(current_edges)
    non_edges_list = list(current_non_edges)
    
    while time.time() < deadline:
        # 1. Local Search Phase
        # We do a few steps of local search (or just one big step)
        # Actually, let's do a "hill climb" loop until no improvement is found in a batch
        
        improved = True
        while improved:
            improved = False
            
            # Sample a batch of moves to find an improvement
            # Since we have 65 edges and 55 non-edges, there are ~3575 possible swaps
            # We can just iterate over a random sample of them
            
            # To speed up, we can just pick random swaps. 
            # If we find an improvement, we accept it and restart the inner loop (or continue).
            
            # Let's try to find the best move in a small random sample? 
            # Or just first improvement? First improvement is faster.
            
            # To avoid getting stuck in 1-opt local minima, we rely on the restart phase.
            
            # Generate a random swap
            if not edges_list or not non_edges_list:
                break
                
            rem = random.choice(edges_list)
            add = random.choice(non_edges_list)
            
            # Apply
            u, v = rem
            x, y = add
            
            # Update state
            adj[u, v] = 0; adj[v, u] = 0
            adj[x, y] = 1; adj[y, x] = 1
            
            deg[u] -= 1; deg[v] -= 1
            deg[x] += 1; deg[y] += 1
            
            # Update sets
            current_edges.remove(rem)
            current_non_edges.add(rem)
            current_non_edges.remove(add)
            current_edges.add(add)
            
            edges_list.remove(rem)
            edges_list.append(add)
            non_edges_list.remove(add)
            non_edges_list.append(rem)
            
            # Update triangles
            # A triangle is affected if it contains u, v, x, or y
            # Actually, it's easier to just recompute triangles involving these vertices
            # But n=16 is small. Let's see if we can do a fast delta.
            
            # Fast Delta:
            # Old triangles: T_old
            # New triangles: T_new
            # Triangles that break: those containing rem edge (u,v) and a common neighbor.
            # Triangles that form: those containing add edge (x,y) and a common neighbor.
            
            # This is complex to track incrementally without a full list.
            # Given n=16, full re-calculation of score is O(E_tri). 
            # Number of triangles in K_8,8+1 is small (only 1 triangle: 0-1-8? No, 0-1 is in A. 
            # 0 is connected to 8..15. 1 is connected to 8..15. 
            # So 0-1-8 is a triangle. 0-1-9 ... 0-1-15. 
            # So there are 8 triangles initially.
            
            # If we just re-calculate score from scratch:
            # get_triangles is O(n^3) worst case, but usually sparse.
            # calc_score is O(T).
            # Total ~ 100-200 ops.
            # We can do ~10,000 iterations per second in Python if optimized?
            # Let's try to optimize the score calculation.
            
            # Instead of get_triangles (which creates a list), let's just compute max score directly.
            
            new_score = 0
            # Find max triangle degree sum
            # Triangles must include edges.
            # We can iterate over edges.
            # But we just changed the graph.
            
            # Let's just re-run the triangle finder. It's small.
            
            # Optimization: 
            # We can pre-compute common neighbors? No, adj changes.
            
            # Let's just call the function.
            
            # To make it faster, let's inline the triangle check.
            # For u in range(n):
            #   for v in range(u+1, n):
            #     if adj[u,v]:
            #       for w in range(v+1, n):
            #         if adj[u,w] and adj[v,w]:
            #            s = deg[u]+deg[v]+deg[w]
            #            if s > new_score: new_score = s
            
            # This is ~ 16*15/2 * 16/2 * 2 checks = 120 * 8 * 2 = 1920 checks.
            # In Python, 1920 iterations is ~ 0.1 ms.
            # So ~10,000 iterations per second.
            # 145 seconds -> 1.4 million iterations.
            # This is plenty.
            
            # Let's do it.
            
            for u in range(n):
                du = deg[u]
                for v in range(u+1, n):
                    if adj[u, v]:
                        dv = deg[v]
                        for w in range(v+1, n):
                            if adj[u, w] and adj[v, w]:
                                s = du + dv + deg[w]
                                if s > new_score:
                                    new_score = s
                                    
            # Acceptance
            if new_score < current_score:
                current_score = new_score
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    best_adj = adj.copy()
            else:
                # Revert
                adj[u, v] = 1; adj[v, u] = 1
                adj[x, y] = 0; adj[y, x] = 0
                
                deg[u] += 1; deg[v] += 1
                deg[x] -= 1; deg[y] -= 1
                
                current_edges.add(rem)
                current_non_edges.remove(rem)
                current_non_edges.add(add)
                current_edges.remove(add)
                
                edges_list.remove(add)
                edges_list.append(rem)
                non_edges_list.remove(rem)
                non_edges_list.append(add)
                
                # current_score remains unchanged
            
        # 2. Restart Phase (Perturbation)
        # If we are stuck (improved is False), we do a random perturbation.
        # To break the max triangle, we can specifically target it.
        
        # Find a max triangle
        max_tri = None
        max_s = -1
        for u in range(n):
            for v in range(u+1, n):
                if adj[u, v]:
                    for w in range(v+1, n):
                        if adj[u, w] and adj[v, w]:
                            s = deg[u] + deg[v] + deg[w]
                            if s > max_s:
                                max_s = s
                                max_tri = (u, v, w)
        
        if max_tri:
            u, v, w = max_tri
            # Try to break this triangle by removing one of its edges and adding a random edge
            # This is a targeted perturbation.
            
            # Pick a random edge from the triangle to remove
            tri_edges = [(u, v), (u, w), (v, w)]
            rem = random.choice(tri_edges)
            
            # Pick a random non-edge to add
            if non_edges_list:
                add = random.choice(non_edges_list)
                
                # Apply
                a, b = rem
                x, y = add
                
                adj[a, b] = 0; adj[b, a] = 0
                adj[x, y] = 1; adj[y, x] = 1
                
                deg[a] -= 1; deg[b] -= 1
                deg[x] += 1; deg[y] += 1
                
                current_edges.remove(rem)
                current_non_edges.add(rem)
                current_non_edges.remove(add)
                current_edges.add(add)
                
                edges_list.remove(rem)
                edges_list.append(add)
                non_edges_list.remove(add)
                non_edges_list.append(rem)
                
                # Calculate new score
                new_score = 0
                for u2 in range(n):
                    du = deg[u2]
                    for v2 in range(u2+1, n):
                        if adj[u2, v2]:
                            dv = deg[v2]
                            for w2 in range(v2+1, n):
                                if adj[u2, w2] and adj[v2, w2]:
                                    s = du + dv + deg[w2]
                                    if s > new_score:
                                        new_score = s
                                        
                # Accept perturbation with Metropolis criterion (or just always accept to escape)
                # Since we want to escape local minima, let's accept if it's not too bad, or always accept.
                # Given the small size, let's just accept it to ensure exploration.
                current_score = new_score
                
                # If this perturbation led to a better score, update best
                if current_score < best_score:
                    best_score = current_score
                    best_adj = adj.copy()
                
                # If it got much worse, we might revert? 
                # No, ILS usually accepts the perturbation and then local searches from there.
                # So we keep it.
                
        T *= cooling
        if T < 0.01:
            T = 0.01

    # Reconstruct edge list from best_adj
    final_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i, j]:
                final_edges.append([i, j])
                
    return {'n': n, 'edges': final_edges}
