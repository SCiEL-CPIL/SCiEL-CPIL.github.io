import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    best_n = 39
    best_edges = []
    
    # Try to build for n=47, then 46, etc.
    for n in range(47, 38, -1):
        min_deg = n - 39
        if min_deg < 0: min_deg = 0
        
        # If min_deg is 0, empty graph is valid
        if min_deg == 0:
            best_n = n
            best_edges = []
            break
        
        # Try multiple random restarts for this n
        success = False
        # Limit restarts to avoid infinite loop if time is running out
        max_restarts = 10000 
        restarts = 0
        
        while time.time() < deadline and not success and restarts < max_restarts:
            restarts += 1
            adj = [0] * n
            deg = [0] * n
            
            # We'll use a greedy approach with randomization
            
            # Loop until all degrees are satisfied or we get stuck
            stuck = False
            steps = 0
            max_steps = n * min_deg * 5 # Generous step limit
            
            while steps < max_steps:
                steps += 1
                
                # Find vertices with minimum degree
                min_deg_val = n
                min_deg_verts = []
                for u in range(n):
                    if deg[u] < min_deg_val:
                        min_deg_val = deg[u]
                        min_deg_verts = [u]
                    elif deg[u] == min_deg_val:
                        min_deg_verts.append(u)
                
                if min_deg_val >= min_deg:
                    # All degrees satisfied
                    success = True
                    break
                
                # Pick a random vertex from the minimum degree set
                u = random.choice(min_deg_verts)
                
                # Find candidates for u
                # Candidates are v != u such that (u, v) is not an edge
                candidates = []
                for v in range(n):
                    if v != u and not ((adj[u] >> v) & 1):
                        candidates.append(v)
                
                random.shuffle(candidates)
                
                added = False
                for v in candidates:
                    # Check if adding (u, v) creates a C4
                    # C4: u-v-x-w-u
                    # Requires x in N(v) and w in N(u) and (x, w) in E
                    safe = True
                    av = adj[v]
                    while av:
                        lsb = av & -av
                        x = lsb.bit_length() - 1
                        # Check if x is connected to any neighbor of u
                        if adj[x] & adj[u]:
                            safe = False
                            break
                        av ^= lsb
                    
                    if safe:
                        # Add edge
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        deg[u] += 1
                        deg[v] += 1
                        added = True
                        break
                
                if not added:
                    stuck = True
                    break
            
            if success:
                # Extract edges
                edges = []
                for u in range(n):
                    au = adj[u]
                    while au:
                        lsb = au & -au
                        v = lsb.bit_length() - 1
                        if u < v:
                            edges.append([u, v])
                        au ^= lsb
                best_n = n
                best_edges = edges
                break # Found solution for this n
        
        if success:
            break
            
    return {'n': best_n, 'edges': best_edges}
