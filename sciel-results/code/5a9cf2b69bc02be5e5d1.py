import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 40
    # Use a base seed and increment for each attempt
    base_seed = int(time.time()) % 100000
    best_edges = []
    best_count = 0
    
    # Helper function
    def is_in_c6(u, v, adj):
        # Check if there is a path u-w-x-z-v of length 4 in adj
        # excluding the direct edge (u,v) if it exists
        nu = np.where(adj[u])[0]
        nu = nu[nu != v]
        
        nv = np.where(adj[v])[0]
        nv = nv[nv != u]
        
        if len(nu) == 0 or len(nv) == 0:
            return False
        
        for w in nu:
            nw = adj[w]
            for z in nv:
                # Check intersection of N(w) and N(z)
                # Exclude u and v
                intersect = np.logical_and(nw, adj[z])
                if intersect[u] or intersect[v]:
                    intersect[u] = False
                    intersect[v] = False
                if np.any(intersect):
                    return True
        return False

    for attempt in range(10):
        if time.time() > deadline:
            break
            
        seed = base_seed + attempt
        rng = np.random.default_rng(seed)
        
        adj = np.zeros((n, n), dtype=bool)
        # Start with p=0.45
        for u in range(n):
            for v in range(u+1, n):
                if rng.random() < 0.45:
                    adj[u, v] = True
                    adj[v, u] = True
                    
        # Phase 1: Remove
        while time.time() < deadline:
            removed = False
            for u in range(n):
                if time.time() > deadline:
                    break
                for v in range(u+1, n):
                    if not adj[u, v]:
                        continue
                    if is_in_c6(u, v, adj):
                        adj[u, v] = False
                        adj[v, u] = False
                        removed = True
                        break
                if removed:
                    break
            if not removed:
                break
        
        # Phase 2: Add
        for _ in range(5):
            if time.time() > deadline:
                break
            added = False
            for u in range(n):
                if time.time() > deadline:
                    break
                for v in range(u+1, n):
                    if adj[u, v]:
                        continue
                    if not is_in_c6(u, v, adj):
                        adj[u, v] = True
                        adj[v, u] = True
                        added = True
            if not added:
                break

        # Count
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if adj[u, v]:
                    edges.append([u, v])
                    
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges

    return best_edges
