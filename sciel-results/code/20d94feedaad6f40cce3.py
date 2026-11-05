import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # We know n=47 is impossible by edge count bound.
    # Try 46, 45, ...
    for n in range(46, 38, -1):
        min_deg = n - 39
        if min_deg < 0:
            min_deg = 0
            
        # If min_deg is 0, empty graph works
        if min_deg == 0:
            return {'n': n, 'edges': []}
            
        max_restarts = 1000
        found = False
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            neighbors = [set() for _ in range(n)]
            edges = []
            degrees = [0] * n
            
            # Randomize
            # We can also try to start with a random sparse graph?
            # No, start empty is fine.
            
            success = True
            # To speed up, we can use a list of vertices sorted by degree?
            # Or just scan.
            
            # We need to add edges until min degree is met.
            # Number of edges to add approx n * min_deg / 2
            # For n=46, min_deg=7, edges ~ 161.
            # 161 iterations. Each iteration scans n candidates.
            # 46 * 46 ~ 2000 ops per edge. 161 * 2000 ~ 300k ops.
            # Very fast.
            
            while True:
                if time.time() > deadline:
                    success = False
                    break
                    
                current_min_deg = min(degrees)
                if current_min_deg >= min_deg:
                    success = True
                    break
                    
                # Pick u with min degree
                # To avoid bias, pick randomly among those with min degree?
                min_deg_val = min(degrees)
                candidates_u = [i for i in range(n) if degrees[i] == min_deg_val]
                u = random.choice(candidates_u)
                
                # Find valid v
                # Prefer v with low degree
                candidates_v = [v for v in range(n) if v != u and v not in neighbors[u]]
                if not candidates_v:
                    success = False
                    break
                    
                # Sort by degree
                candidates_v.sort(key=lambda v: degrees[v])
                
                added = False
                for v in candidates_v:
                    # Check C4
                    if len(neighbors[u] & neighbors[v]) <= 1:
                        neighbors[u].add(v)
                        neighbors[v].add(u)
                        degrees[u] += 1
                        degrees[v] += 1
                        edges.append([u, v])
                        added = True
                        break
                        
                if not added:
                    success = False
                    break
                    
            if success:
                found = True
                return {'n': n, 'edges': edges}
                
        if found:
            break
            
    return {'n': 39, 'edges': []}
