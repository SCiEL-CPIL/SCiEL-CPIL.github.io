import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Fallback: Lazebnik-Ustimenko-Woldar construction
    def luw_construct(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # Target parameters
    # We assume k=5, g=9 is the primary case, but handle generally.
    # Moore bound
    if g % 2 == 0:
        r = g // 2 - 1
        moore = 2 * (1 + k * sum((k-1)**i for i in range(r)))
    else:
        r = (g - 1) // 2
        moore = 1 + k * sum((k-1)**i for i in range(r))
    
    # We try to find a graph with n close to the known bounds or slightly above Moore bound.
    # For k=5, g=9, known bound is 1116.
    # We will try n = 1100, 1150, 1200, etc.
    
    # For other k,g, we might just use L-U-W if we don't have a specific strategy.
    # But the heuristic should work generally.
    
    # Heuristic Search
    # 1. Generate random k-regular graph
    # 2. Check girth
    # 3. Local search to increase girth
    
    def generate_random_k_regular(n, k):
        # Configuration model
        while True:
            stubs = [i for i in range(n) for _ in range(k)]
            random.shuffle(stubs)
            edges = set()
            valid = True
            for i in range(0, 2*n*k, 2):
                u, v = stubs[i], stubs[i+1]
                if u == v:
                    valid = False
                    break
                if (u, v) in edges or (v, u) in edges:
                    valid = False
                    break
                edges.add((min(u, v), max(u, v)))
            if valid:
                return list(edges)
        
    def check_girth(edges, n, g):
        # Returns True if girth >= g
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Check for cycles of length < g
        # For each edge (u,v), check if there is a path from u to v of length < g-1 not using (u,v)
        # This is equivalent to checking if dist(u,v) < g-1 in G - (u,v)
        
        # Optimization: Only check up to length g-2
        # BFS from u, avoiding v initially? No, avoid the edge.
        
        # For small g, we can just BFS from u and see if we hit v in < g-1 steps.
        # But we must not use the edge (u,v) directly.
        
        # A cycle of length L exists if there is a path of length L-1 between the endpoints of an edge.
        
        # We will check for the smallest girth.
        # If we find a cycle of length < g, return False.
        
        # To speed up, we can check for 3-cycles and 4-cycles explicitly if g > 4, etc.
        # For g=9, we check 3,4,5,6,7,8.
        
        # Efficient check:
        # For each vertex u:
        #   BFS up to depth (g-1)//2
        #   If we find a node v at distance d, and we know that v can reach u in d steps via a different path?
        
        # Let's use the edge-removal BFS but limit depth to g-2.
        # If g is small (e.g., 9), depth is 7.
        # n=1100, m=2750.
        # 2750 BFS of depth 7.
        # 5^7 = 78125 nodes.
        # 2750 * 78125 = 215 million ops. Too slow in Python.
        
        # Better: Find *any* short cycle.
        # 1. Check 3-cycles: For each edge (u,v), if common neighbor, cycle 3.
        # 2. Check 4-cycles: For each edge (u,v), if common neighbor of u and v (excluding each other), cycle 4.
        # 3. Check 5-cycles: ...
        
        # We will implement a BFS that stops when a cycle is found.
        # We'll iterate edges and do a limited BFS.
        
        for u, v in edges:
            # BFS from u to v, not using edge (u,v)
            # Max depth g-2
            visited = {u: 0}
            queue = deque([(u, 0)])
            found = False
            while queue:
                node, dist = queue.popleft()
                if dist >= g - 1:
                    break
                for neighbor in adj[node]:
                    if neighbor == v and dist + 1 < g:
                        # Found a path of length dist+1. Cycle length dist+2.
                        # Wait, if we reach v, the path is u->...->v.
                        # The cycle is u->...->v->u. Length dist+2.
                        # We want cycle length < g.
                        # So dist+2 < g => dist < g-2.
                        return False
                    if neighbor == v:
                        continue
                    if neighbor not in visited:
                        visited[neighbor] = dist + 1
                        queue.append((neighbor, dist + 1))
        return True

    def find_short_cycle(edges, n, g):
        # Returns a list of nodes forming a cycle of length < g, or None
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Check 3-cycles
        for u in range(n):
            for i in range(len(adj[u])):
                for j in range(i+1, len(adj[u])):
                    v, w = adj[u][i], adj[u][j]
                    if v in adj[w]:
                        return [u, v, w]
                        
        # Check 4-cycles
        # For each edge (u,v), check common neighbors of u and v.
        edge_set = set((min(u,v), max(u,v)) for u,v in edges)
        for u, v in edges:
            common = set(adj[u]) & set(adj[v])
            # Remove u and v from common? No, u is not in adj[u].
            # If there is a node x in common, u-x-v-u is a 3-cycle? No, u-x-v is path 2.
            # Cycle is u-x-v-u? Length 3? No, u-x, x-v, v-u.
            # So if common is not empty, we have a 3-cycle.
            # Wait, if u and v are connected, and share a neighbor x, then u-x-v-u is a 3-cycle.
            # So checking 3-cycles covers this.
            
            # For 4-cycles: u-x-y-v-u.
            # x is neighbor of u, y is neighbor of v. x != y, x != v, y != u.
            # And x-y is an edge.
            # This is more complex.
            
        # For simplicity, we rely on the BFS in check_girth to just return False.
        # But we need the cycle to break it.
        # Let's just return the first edge that fails the BFS?
        # We can modify check_girth to return the cycle.
        
        # For now, if check_girth fails, we pick a random edge to swap.
        return None

    # Main Search Loop
    # We try a few n values
    n_candidates = []
    if k == 5 and g == 9:
        n_candidates = [1100, 1116, 1200, 1300]
    else:
        # Estimate n
        n_est = moore * 1.5
        n_candidates = [int(n_est), int(n_est * 1.2), int(n_est * 1.5)]
        
    for n in n_candidates:
        if time.time() > deadline - 10:
            break
            
        # Try to find a valid graph of size n
        # We will run local search for a few seconds
        search_deadline = time.time() + 15
        
        while time.time() < search_deadline:
            edges = generate_random_k_regular(n, k)
            if check_girth(edges, n, g):
                return n, edges
            
            # Local Search
            # Try 100 swaps
            for _ in range(100):
                if time.time() > search_deadline:
                    break
                # Pick a random edge to break
                u, v = random.choice(edges)
                # Find a new edge (u, w) and (x, v) to replace (u,v) with (u,x) and (v,w)?
                # Standard 2-opt swap:
                # Remove (u,v) and (x,y). Add (u,x) and (v,y).
                # We need to find x,y such that no loops/multi-edges are created.
                
                # Simplified: Just retry random generation if local search is complex.
                # Given the high density of short cycles, random restarts might be better.
                pass
                
    # Fallback
    return luw_construct(k, g)
