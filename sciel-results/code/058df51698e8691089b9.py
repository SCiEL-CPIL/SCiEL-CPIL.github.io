import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_result = {'n': 39, 'edges': []}
    
    for n in [46, 45, 44, 43, 42, 41, 40, 39]:
        if time.time() > deadline:
            break
            
        min_deg_needed = n - 39
        if min_deg_needed <= 0:
            best_result = {'n': n, 'edges': []}
            continue
            
        success = False
        # Multiple random restarts
        for attempt in range(200):
            if time.time() > deadline:
                break
                
            random.seed(attempt * 1000 + n)
            
            adj = [set() for _ in range(n)]
            deg = [0] * n
            # common[u][v] is the number of common neighbors of u and v
            # Using a list of lists for speed
            common = [[0] * n for _ in range(n)]
            edges = []
            
            # Greedy addition
            # We want to add edges to increase degrees.
            # Strategy: Pick edge (u,v) that minimizes deg[u] + deg[v]
            # among valid edges.
            
            max_passes = n * 5
            for pass_num in range(max_passes):
                if time.time() > deadline:
                    break
                    
                best_edge = None
                best_score = float('inf')
                
                # To speed up, we can just iterate and pick the first valid one 
                # with low degree sum, or randomize.
                # Given n=46, n^2 is small.
                
                # Iterate over all pairs
                for u in range(n):
                    for v in range(u+1, n):
                        if v in adj[u]:
                            continue
                        
                        # Check validity
                        # 1. common[u][v] <= 1
                        if common[u][v] > 1:
                            continue
                        
                        # 2. For w in N(v), common[u][w] must be 0
                        valid = True
                        for w in adj[v]:
                            if common[u][w] > 0:
                                valid = False
                                break
                        if not valid:
                            continue
                        
                        # 3. For w in N(u), common[v][w] must be 0
                        for w in adj[u]:
                            if common[v][w] > 0:
                                valid = False
                                break
                        if not valid:
                            continue
                        
                        score = deg[u] + deg[v]
                        if score < best_score:
                            best_score = score
                            best_edge = (u, v)
                
                if best_edge is None:
                    break
                    
                u, v = best_edge
                adj[u].add(v)
                adj[v].add(u)
                deg[u] += 1
                deg[v] += 1
                edges.append([u, v])
                
                # Update common matrix
                # For w in N(v): common[u][w] += 1
                for w in adj[v]:
                    if w != u:
                        common[u][w] += 1
                        common[w][u] += 1
                        
                # For w in N(u): common[v][w] += 1
                for w in adj[u]:
                    if w != v:
                        common[v][w] += 1
                        common[w][v] += 1
                        
                # Early check: if min degree reached, we can stop adding?
                # Yes, we want to ensure min degree holds.
                # If we add more, we might break C4-free? No, we check safety.
                # But we don't need more edges.
                min_deg = min(deg)
                if min_deg >= min_deg_needed:
                    break
                    
            # Final check
            min_deg = min(deg)
            if min_deg >= min_deg_needed:
                # Verify C4-free
                has_c4 = False
                for u in range(n):
                    for v in range(u+1, n):
                        if common[u][v] > 1:
                            has_c4 = True
                            break
                    if has_c4:
                        break
                if not has_c4:
                    success = True
                    best_result = {'n': n, 'edges': edges}
                    break
                    
        if success:
            break
            
    return best_result
