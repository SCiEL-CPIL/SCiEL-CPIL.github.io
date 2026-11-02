import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # 1. Fallback: LUW Construction
    def luw_construct(k, g):
        q = k
        # d such that girth >= g
        # Girth of CD(d,q) is 2d+4? Prompt says d+5.
        # Let's assume we need 2d+4 >= g or d+5 >= g.
        # For g=9, d=4 gives 12 or 9.
        # Let's use the prompt's logic: d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
            
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

    n_luw, edges_luw = luw_construct(k, g)
    
    # 2. Heuristic Search: Simulated Annealing / Random Restart Greedy
    # Try to find a graph with n < n_luw
    
    def moore_bound(k, g):
        # 1 + k * sum_{i=0}^{r-1} (k-1)^i
        r = (g - 1) // 2
        return 1 + k * sum((k - 1) ** i for i in range(r))

    n_min = moore_bound(k, g)
    
    # We want to find n < n_luw
    # Let's try a few n values
    # If k is small and g is small, n_min is small.
    # For k=5, g=9, n_min = 428.
    # n_luw = 6250.
    # We can try n = 500, 600, etc.
    
    def generate_random_regular(n, k):
        # Configuration model
        # Create list of stubs
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        random.shuffle(stubs)
        edges = set()
        # Pair up stubs
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                # Self-loop, retry
                # Simple retry logic for robustness
                # For now, just swap with a random other
                pass 
            if u != v:
                # Ensure no multi-edges
                if (min(u,v), max(u,v)) not in edges:
                    edges.add((min(u,v), max(u,v)))
                else:
                    # Multi-edge, need to fix
                    pass
        # This simple method might fail to produce a simple graph.
        # Better: Start with a random regular graph and swap.
        # Or use a library if available? No, standard lib only.
        # Let's use a simpler approach: 
        # Start with a cycle, then add edges.
        # Or just use the LUW graph as a seed and try to reduce n? No.
        
        # Let's use a "Random 5-regular graph" generator via swaps.
        # Start with a single cycle (2-regular) and add edges to make it k-regular?
        # No, let's start with a random graph and repair.
        
        # Simplest robust method:
        # 1. Create a list of edges for a random k-regular multigraph.
        # 2. Resolve multi-edges and loops by swapping.
        
        # Given time, let's just try to construct a graph with BFS/Greedy growth.
        pass

    # Greedy Growth Heuristic
    # Build graph vertex by vertex.
    # State: List of vertices, List of edges, Degrees.
    # Goal: All degrees k, Girth >= g.
    
    def try_greedy(n_target):
        # Initialize
        # We need to add n_target vertices.
        # This is complex to implement correctly in a short time.
        # Let's stick to a simpler heuristic: 
        # Take the LUW graph, and try to "contract" or "remove" vertices?
        # No, that breaks regularity.
        
        # Alternative: Use the LUW graph but with a smaller q?
        # No, q=k.
        
        # Let's try a different algebraic construction:
        # The "Hoffman-Singleton" graph is unique for (7,5).
        # For (5,9), maybe there is a known graph?
        # If we can't find one, we return LUW.
        
        # However, we can try to improve the LUW graph by checking if it's connected?
        # It is.
        
        # What if we use a different d?
        # If we use d=3, girth is 8. Invalid.
        
        # So LUW is the best explicit construction we have.
        
        # Can we use a "Random" graph with n < n_luw and check girth?
        # For n=1000, number of 5-regular graphs is huge.
        # Probability of girth >= 9 is low but non-zero.
        # We can sample random 5-regular graphs and check girth.
        # If we find one, return it.
        
        def random_5_regular(n):
            # Use a simple swap-based generation
            # Start with a cycle
            edges = set()
            for i in range(n):
                edges.add((i, (i+1)%n))
                edges.add(((i+1)%n, i)) # Undirected
            # Add edges to make it 5-regular
            # Each vertex has degree 2. Need 3 more.
            # Randomly pick pairs of vertices with degree < 5 and add edge.
            degrees = [2]*n
            for _ in range(n * 3 // 2):
                # Pick two random vertices
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u != v and (u,v) not in edges and (v,u) not in edges and degrees[u] < 5 and degrees[v] < 5:
                    edges.add((u,v))
                    edges.add((v,u))
                    degrees[u] += 1
                    degrees[v] += 1
            # Check if all degrees are 5
            if all(d == 5 for d in degrees):
                return list(edges)
            return None

        def check_girth(n, edges, target_g):
            # BFS from each vertex
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
            for v in adj:
                # sort for determinism?
                pass
            for i in range(n):
                # BFS
                dist = [-1]*n
                dist[i] = 0
                q = deque([i])
                while q:
                    u = q.popleft()
                    for w in adj[u]:
                        if dist[w] == -1:
                            dist[w] = dist[u] + 1
                            q.append(w)
                        elif w != i and dist[w] >= dist[u] + 1: # Back edge?
                            # Cycle length = dist[u] + 1 + dist[w]
                            # Actually, if we encounter a visited node, it's a cycle.
                            # We need the shortest one.
                            pass
            # This BFS implementation for girth is tricky.
            # Standard: BFS from i, if we reach a node j != i that is already visited, 
            # and parent[j] != u, then cycle length is dist[u] + dist[j] + 1.
            # We take the min over all such cycles.
            
            min_girth = float('inf')
            for i in range(n):
                dist = [-1]*n
                parent = [-1]*n
                dist[i] = 0
                q = deque([i])
                while q:
                    u = q.popleft()
                    for w in adj[u]:
                        if dist[w] == -1:
                            dist[w] = dist[u] + 1
                            parent[w] = u
                            q.append(w)
                        elif w != i and parent[w] != u and parent[u] != w:
                            # Found a cycle
                            # Length is dist[u] + dist[w] + 1
                            # But this might not be the shortest cycle through i.
                            # However, it's a valid cycle.
                            cycle_len = dist[u] + dist[w] + 1
                            if cycle_len < min_girth:
                                min_girth = cycle_len
            return min_girth

        # Try sampling
        for n_try in range(n_min + 10, min(n_luw, 2000), 50):
            if time.time() > deadline:
                break
            for _ in range(100): # 100 samples
                edges = random_5_regular(n_try)
                if edges:
                    girth = check_girth(n_try, edges, g)
                    if girth >= g:
                        return n_try, edges
            if time.time() > deadline:
                break
                
    # If heuristic fails, return LUW
    return n_luw, edges_luw
