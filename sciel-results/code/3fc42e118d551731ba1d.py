import time
import random

def _try_construct_greedy(n, target_min_deg, time_limit=10, restarts=1000):
    """
    Try to construct a C4-free graph on n vertices with min degree >= target_min_deg.
    """
    start_time = time.time()
    
    for restart in range(restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize
        adj = [set() for _ in range(n)]
        # c[u][v] is the number of common neighbors of u and v
        # Using a list of lists for speed, but n is small (47), so 47x47 is fine.
        c = [[0] * n for _ in range(n)]
        
        # We will greedily add edges.
        # To avoid bias, we shuffle the order in which we consider vertices.
        # However, we want to fix low-degree vertices first.
        
        # Strategy:
        # 1. While there are vertices with degree < target_min_deg:
        #    Pick one such vertex u (randomly among those with min degree).
        #    Try to connect u to any v such that c[u][v] == 0.
        #    If we can't connect u to anyone, we might need to remove some edges 
        #    incident to u's neighbors to free up space? 
        #    Or just fail this restart.
        
        # Let's try a simple greedy:
        # Iterate through vertices in random order.
        # For each u, try to add edges to all v where c[u][v]==0.
        # This might not maximize min degree if early vertices hog edges.
        
        # Better:
        # Maintain a list of "deficient" vertices.
        # While deficient list is not empty:
        #   Pick u from deficient list.
        #   candidates = [v for v in range(n) if v != u and v not in adj[u] and c[u][v] == 0]
        #   if not candidates:
        #       break # Stuck
        #   v = random.choice(candidates)
        #   # Add edge (u, v)
        #   adj[u].add(v)
        #   adj[v].add(u)
        #   # Update c
        #   for x in adj[u]:
        #       if x != v: # x is in adj[u] and x != v
        #           # u is a common neighbor of x and v
        #           c[x][v] += 1
        #           c[v][x] += 1
        #   for y in adj[v]:
        #       if y != u: # y is in adj[v] and y != u
        #           # v is a common neighbor of u and y
        #           c[u][y] += 1
        #           c[y][u] += 1
        #   # Check if u or v are no longer deficient
        #   if len(adj[u]) >= target_min_deg:
        #       # remove from deficient list if present
        #       pass # We'll just re-evaluate deficient list each time or keep a counter
        #   if len(adj[v]) >= target_min_deg:
        #       pass
        
        # To make it efficient, we just loop until no more edges can be added to deficient vertices.
        
        deficient = [i for i in range(n) if len(adj[i]) < target_min_deg]
        
        while deficient:
            # Pick a random deficient vertex
            u = random.choice(deficient)
            
            # Find valid neighbors
            candidates = []
            for v in range(n):
                if v == u:
                    continue
                if v in adj[u]:
                    continue
                if c[u][v] == 0:
                    candidates.append(v)
            
            if not candidates:
                # Cannot add any edge to u without creating C4
                # This restart failed
                break
            
            v = random.choice(candidates)
            
            # Add edge (u, v)
            adj[u].add(v)
            adj[v].add(u)
            
            # Update common neighbor counts
            # For each x in adj[u] (excluding v):
            # u becomes a common neighbor of x and v
            for x in adj[u]:
                if x != v:
                    c[x][v] += 1
                    c[v][x] += 1
                    
            # For each y in adj[v] (excluding u):
            # v becomes a common neighbor of u and y
            for y in adj[v]:
                if y != u:
                    c[u][y] += 1
                    c[y][u] += 1
            
            # Update deficient list
            # We can just rebuild it or update it. Rebuilding is O(n) which is fine.
            deficient = [i for i in range(n) if len(adj[i]) < target_min_deg]
            
        # Check if successful
        if not deficient:
            # All vertices have degree >= target_min_deg
            # Verify C4-free (should be, but good to be safe)
            # And build edge list
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append([u, v])
            return {'n': n, 'edges': edges}
            
    return None

def construct():
    start_time = time.time()
    deadline = 140  # Safety margin before 150s limit
    
    # Try n=47 first (min deg 8)
    # Then n=46 (min deg 7)
    # Then n=45 (min deg 6)
    
    for n in range(47, 38, -1):
        if time.time() - start_time > deadline:
            break
        
        remaining_time = deadline - (time.time() - start_time)
        if remaining_time < 1:
            break
            
        target_min_deg = n - 39
        if target_min_deg < 1:
            target_min_deg = 1
            
        result = _try_construct_greedy(n, target_min_deg, time_limit=remaining_time, restarts=5000)
        if result is not None:
            return result
            
    # Fallback
    return {'n': 39, 'edges': []}
