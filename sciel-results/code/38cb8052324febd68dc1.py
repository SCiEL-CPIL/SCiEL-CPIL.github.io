import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_n = 0
    best_edges = []
    
    # Try n=47 first, then 46
    for target_n in [47, 46]:
        if time.time() > deadline:
            break
            
        min_deg_req = target_n - 39
        
        # Multiple random restarts
        found = False
        for attempt in range(500):
            if time.time() > deadline:
                break
            
            # Initialize random graph
            # Expected degree ~ min_deg_req
            p = min_deg_req / target_n
            # Slightly higher to ensure we can fix degrees
            p = min(p * 1.5, 0.5)
            
            adj = [set() for _ in range(target_n)]
            
            # Random edges
            for u in range(target_n):
                for v in range(u + 1, target_n):
                    if random.random() < p:
                        adj[u].add(v)
                        adj[v].add(u)
            
            # Local search
            max_iter = 2000
            for _ in range(max_iter):
                if time.time() > deadline:
                    break
                    
                # Check min degree
                min_deg = min(len(adj[i]) for i in range(target_n))
                if min_deg < min_deg_req:
                    # Degrees are too low. Restart is better than trying to add edges safely.
                    break
                
                # Check for C4
                # If any pair (u,v) has >= 2 common neighbors, there is a C4.
                c4_found = False
                for u in range(target_n):
                    if c4_found: break
                    for v in range(u + 1, target_n):
                        common = adj[u] & adj[v]
                        if len(common) >= 2:
                            # Found C4 involving u, v and two common neighbors
                            # Pick two common neighbors w, x
                            w, x = list(common)[:2]
                            # The C4 is u-w-v-x-u
                            # Edges: (u,w), (w,v), (v,x), (x,u)
                            # Remove one edge to break it.
                            # Remove (u,w)
                            adj[u].discard(w)
                            adj[w].discard(u)
                            c4_found = True
                            break
                            
                if not c4_found:
                    # No C4 found, and min degree is satisfied
                    found = True
                    break

            if found:
                edges = []
                for i in range(target_n):
                    for j in adj[i]:
                        if i < j:
                            edges.append([i, j])
                
                if target_n > best_n:
                    best_n = target_n
                    best_edges = edges
                if target_n == 47:
                    return {'n': best_n, 'edges': best_edges}

    if best_n > 0:
        return {'n': best_n, 'edges': best_edges}
    
    # Fallback
    return {'n': 39, 'edges': []}
