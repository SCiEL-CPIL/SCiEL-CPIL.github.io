import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def verify(adj, n, min_deg):
        # Check min degree
        if min(adj[i].bit_count() for i in range(n)) < min_deg:
            return False
        # Check C4: for every pair u,v, they must have at most 1 common neighbor
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] & adj[j]).bit_count() > 1:
                    return False
        return True

    def get_edges(adj, n):
        edges = []
        for i in range(n):
            mask = adj[i]
            # Only consider j > i
            for j in range(i + 1, n):
                if mask & (1 << j):
                    edges.append([i, j])
        return edges

    def construct_bitset(n, min_deg, time_limit):
        """
        Construct C4-free graph on n vertices with min degree >= min_deg.
        Uses bitsets for adjacency.
        Combines:
        1. Greedy construction (from current/inspiration)
        2. Local repair (edge removal) when stuck (from inspiration)
        3. Bitset efficiency (from current)
        """
        start = time.time()
        
        for attempt in range(200):
            if time.time() - start > time_limit:
                break
            
            adj = [0] * n
            deg = [0] * n
            
            # Heuristic: Initialize with a random sparse graph to break symmetry?
            # Or start empty. Starting empty is safer for C4 avoidance initially.
            
            success = False
            stuck_count = 0
            max_stuck = 50  # Allow some repairs before restarting
            
            # Main loop
            while time.time() - start < time_limit:
                # Check if min degree is satisfied
                min_d = min(deg)
                if min_d >= min_deg:
                    success = True
                    break
                
                # Find vertex with minimum degree
                # Randomize among min-degree vertices to break symmetry
                min_vertices = [i for i in range(n) if deg[i] == min_d]
                u = random.choice(min_vertices)
                
                # Find valid neighbors for u
                # A neighbor v is valid if:
                # 1. v != u
                # 2. (u,v) not in adj
                # 3. adj[u] & adj[v] == 0 (no common neighbors, to avoid C4)
                
                mask_u = adj[u]
                candidates = []
                for v in range(n):
                    if v == u:
                        continue
                    if mask_u & (1 << v):
                        continue
                    if adj[u] & adj[v]:
                        continue
                    candidates.append(v)
                
                if candidates:
                    # Pick a random candidate
                    v = random.choice(candidates)
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    deg[u] += 1
                    deg[v] += 1
                    stuck_count = 0
                else:
                    # Stuck: no valid neighbor for u
                    stuck_count += 1
                    
                    # Try local repair: remove a random edge to potentially free up space
                    # Strategy: Remove an edge (x, y) where x is a neighbor of u or y is a neighbor of u?
                    # Actually, removing any edge might help if it's causing the C4 constraint.
                    # But we need to be careful not to drop min degree below 0.
                    
                    # If stuck too many times, restart
                    if stuck_count > max_stuck:
                        break
                    
                    # Try to remove a random edge
                    # Pick a random vertex x
                    x = random.randint(0, n - 1)
                    if deg[x] == 0:
                        continue
                    # Pick a random neighbor y of x
                    mask_x = adj[x]
                    # Get list of neighbors
                    neighbors = [i for i in range(n) if mask_x & (1 << i)]
                    if not neighbors:
                        continue
                    y = random.choice(neighbors)
                    
                    # Remove edge (x, y)
                    adj[x] &= ~(1 << y)
                    adj[y] &= ~(1 << x)
                    deg[x] -= 1
                    deg[y] -= 1
                    
                    # Note: This might create a new bottleneck, but it's a random walk.
            
            if success:
                # Final verification
                if verify(adj, n, min_deg):
                    return {'n': n, 'edges': get_edges(adj, n)}
        
        return None

    # Target n=46 first (min_deg=7), then 45 (min_deg=6), etc.
    # n=47 requires min_deg=8.
    # The Reiman bound suggests n=47 might be impossible, so we focus on 46.
    
    targets = [
        (46, 7),
        (45, 6),
        (44, 5),
        (43, 4),
    ]
    
    for n, min_deg in targets:
        remaining = deadline - time.time()
        if remaining < 5.0:
            break
        
        # Allocate time based on difficulty. n=46 is hardest.
        # Give most time to the first target.
        if n == 46:
            t_limit = remaining * 0.85
        else:
            t_limit = remaining * 0.4
            
        res = construct_bitset(n, min_deg, t_limit)
        if res:
            return res
    
    # Fallback: Return a small valid graph
    # n=1, min_deg=0
    return {'n': 1, 'edges': []}

# Entry point
# The function name and signature must be exactly `construct()`
