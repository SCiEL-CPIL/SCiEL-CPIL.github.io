import time
import random
from collections import deque

def construct(k, g):
    # The known smallest 4-regular graph with girth >= 10 has 320 vertices.
    # We will search for circulant graphs which are 4-regular.
    
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

    if k == 4 and g == 10:
        start_time = time.time()
        deadline = start_time + 110
        
        # Search for circulant graphs C_n(a, b)
        # n must be such that a 4-regular graph is possible.
        # Moore bound is 243.
        
        def fast_girth_check(n, steps, min_girth):
            """
            BFS from 0. Returns True if no cycle shorter than min_girth is found 
            involving vertex 0 or detected via non-tree edges from 0's BFS tree.
            Note: This is a necessary condition but not sufficient for global girth.
            """
            adj = [[] for _ in range(n)]
            for s in steps:
                for u in range(n):
                    v = (u + s) % n
                    adj[u].append(v)
                    # adj[v].append(u) is implicit in undirected, but we build full adj for simplicity
            
            # Actually, for circulant, we can just use steps.
            # BFS from 0
            dist = [-1] * n
            parent = [-1] * n
            dist[0] = 0
            parent[0] = -2
            queue = deque([0])
            
            min_cycle = float('inf')
            
            while queue:
                u = queue.popleft()
                if dist[u] >= min_girth: # No need to expand if we've already found a shorter cycle?
                    # Actually we need to check all edges.
                    pass
                
                for s in steps:
                    v = (u + s) % n
                    # Also need -s? 
                    # The steps list should contain all generators.
                    # If steps = [a, -a, b, -b], then we cover all neighbors.
                    
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        # Check for cycle
                        if parent[u] == v:
                            continue
                        # If v is already visited and not parent
                        # Cycle length = dist[u] + dist[v] + 1
                        # This is valid if the path from 0 to u and 0 to v are distinct 
                        # and the edge (u,v) closes the loop.
                        # In a BFS tree, if dist[v] != dist[u] + 1 and dist[v] != dist[u] - 1, 
                        # it's a cross edge.
                        # If dist[v] == dist[u], cycle is 2*dist[u] + 1.
                        # If dist[v] == dist[u] - 1, it's a back edge to parent (skipped) 
                        # or a cross edge to a grandparent?
                        # In unweighted graph BFS, edges only connect levels d, d+1 or d, d-1.
                        # So if dist[v] == dist[u] - 1 and parent[u] != v, it's a cross edge.
                        
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if min_cycle < min_girth:
                            return False, min_cycle
            
            return True, min_cycle

        def full_girth_check(n, steps, min_girth):
            """
            Verify girth >= min_girth by checking all vertices.
            """
            adj = [[] for _ in range(n)]
            for s in steps:
                for u in range(n):
                    v = (u + s) % n
                    adj[u].append(v)
            
            for start in range(n):
                dist = [-1] * n
                parent = [-1] * n
                dist[start] = 0
                parent[start] = -2
                queue = deque([start])
                
                while queue:
                    u = queue.popleft()
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            queue.append(v)
                        else:
                            if parent[u] == v:
                                continue
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < min_girth:
                                return False
            return True

        # Generate steps: {a, -a, b, -b}
        # We need to ensure the graph is simple (no loops, no multi-edges).
        # Loop: a = 0 or a = n/2 (if n even).
        # Multi-edge: a = -b mod n => 2b = 0 mod n => n divides 2b.
        
        found = False
        # Try n from 244 to 600
        for n in range(244, 600):
            if time.time() > deadline:
                break
            
            # Number of attempts per n
            # Smaller n are more constrained, try more?
            # Larger n are easier to satisfy, try fewer?
            attempts = 5000 if n < 300 else 2000
            
            for _ in range(attempts):
                if time.time() > deadline:
                    break
                
                # Random steps
                # a, b in 1..n//2
                # Ensure a != b, a != n-b, etc.
                a = random.randint(1, n // 2)
                b = random.randint(1, n // 2)
                
                # Check for degeneracy
                if a == b:
                    continue
                if (2 * a) % n == 0: # Loop
                    continue
                if (2 * b) % n == 0: # Loop
                    continue
                if (a + b) % n == 0: # Multi-edge a = -b
                    continue
                if (a - b) % n == 0: # Multi-edge a = b
                    continue
                # Also check if 2a = 2b mod n => a=b mod n/2?
                # If n is even, a = n/2 is a loop.
                # If n is even, 2a = 0 mod n is loop.
                
                steps = [a, n - a, b, n - b]
                
                # Fast check
                ok, _ = fast_girth_check(n, steps, g)
                if not ok:
                    continue
                
                # Full check
                if full_girth_check(n, steps, g):
                    # Construct edges
                    edges = []
                    for u in range(n):
                        for s in [a, b]: # Only positive steps to avoid duplicates
                            v = (u + s) % n
                            if u < v:
                                edges.append((u, v))
                    
                    return n, edges
        
        # If no circulant found, fallback
        return luw_construct(k, g)
    
    else:
        return luw_construct(k, g)
