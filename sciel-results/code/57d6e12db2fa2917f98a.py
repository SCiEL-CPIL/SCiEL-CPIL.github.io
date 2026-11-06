"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s limit
    
    n = 20
    # Fallback: a star (19 edges, no 4-cycles)
    best_edges = [[0, v, 0] for v in range(1, n)]
    best_count = len(best_edges)
    
    def check_valid(adj, u, v, c):
        """
        Check if adding edge (u,v) with colour c creates any non-rainbow 4-cycle.
        A 4-cycle containing (u,v) is u-v-x-y-u.
        x must be neighbor of v, y must be neighbor of u, and x-y must be an edge.
        Colors c(u,v), c(v,x), c(x,y), c(y,u) must be distinct.
        """
        # Iterate x over neighbors of v
        for x in range(n):
            if x == u or x == v:
                continue
            cvx = adj[v][x]
            if cvx == -1:
                continue
            
            # Color conflict check: c(u,v) vs c(v,x)
            if c == cvx:
                return False
            
            # Iterate y over neighbors of u
            for y in range(n):
                if y == u or y == v or y == x:
                    continue
                
                cyu = adj[u][y]
                if cyu == -1:
                    continue
                
                # Color conflict check: c(u,v) vs c(y,u), c(v,x) vs c(y,u)
                if c == cyu or cvx == cyu:
                    return False
                
                # Check if x and y are connected
                cxy = adj[x][y]
                if cxy == -1:
                    continue
                
                # Final color conflict check: c, cvx, cyu, cxy must be distinct
                if c == cxy or cvx == cxy or cyu == cxy:
                    return False
        
        return True

    def local_search(adj, edges, time_limit):
        """
        Attempt to improve the graph by swapping colors or adding edges.
        Returns (adj, edges, improved_count).
        """
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # 1. Try to add missing edges
            # Iterate through all pairs not in adj
            for u in range(n):
                for v in range(u+1, n):
                    if adj[u][v] != -1:
                        continue
                    
                    # Try all colors
                    # Use a heuristic: prefer colors not used on neighbors
                    # For simplicity, just try random order or sorted
                    for c in range(n):
                        if check_valid(adj, u, v, c):
                            adj[u][v] = c
                            adj[v][u] = c
                            edges.append([u, v, c])
                            improved = True
                            break
                    if time.time() > time_limit:
                        return adj, edges, len(edges)
            
            # 2. Try color swaps to enable more edges
            # This is complex, so we rely primarily on the greedy add.
            # If we want to be smarter, we could identify "bottleneck" edges.
            # But for now, the greedy add with random restarts is the main driver.
            # Let's add a simple "shaking" step: if stuck, maybe remove a random edge and re-add?
            # That's risky. Let's stick to the greedy add for local search.
            
        return adj, edges, len(edges)

    all_pairs = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    # Strategy: 
    # 1. Run multiple random greedy constructions.
    # 2. Apply local search (greedy add of missing edges) to the result.
    # 3. Keep the best.
    
    # We can also try to bias the initial construction.
    # For example, start with a random spanning tree or a sparse random graph, then fill.
    
    max_attempts = 5000
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        adj = [[-1]*n for _ in range(n)]
        edges = []
        
        # Random edge ordering
        order = all_pairs[:]
        random.shuffle(order)
        
        # Greedy construction
        for u, v in order:
            if time.time() > deadline - 5:  # Leave time for local search
                break
            
            # Try colours in a random order
            colours = list(range(n))
            random.shuffle(colours)
            
            for c in colours:
                if check_valid(adj, u, v, c):
                    adj[u][v] = c
                    adj[v][u] = c
                    edges.append([u, v, c])
                    break
        
        # Local search: try to add any remaining edges
        # This is a second pass over all pairs
        for u, v in all_pairs:
            if time.time() > deadline - 2:
                break
            if adj[u][v] != -1:
                continue
            for c in range(n):
                if check_valid(adj, u, v, c):
                    adj[u][v] = c
                    adj[v][u] = c
                    edges.append([u, v, c])
                    break
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
            
            # If we find a very dense graph, we can stop early?
            # The maximum possible edges in a graph on 20 vertices is 190.
            # But the rainbow constraint is very tight.
            # If we get > 100 edges, it's good.
            # Let's not stop early to allow finding even denser ones.
    
    return best_edges
# EVOLVE-BLOCK-END
