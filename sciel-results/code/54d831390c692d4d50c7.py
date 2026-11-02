import numpy as np
import random
import time

def construct(k, g):
    # Fallback to seed if search fails
    def seed_construct():
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

    # Heuristic Search
    target_n = 320
    deadline = time.time() + 100 # 100s limit for search
    
    def get_girth(adj, n):
        # BFS from each vertex to find shortest cycle
        # For 4-regular, we can just check up to depth 4
        # If no cycle found, girth > 8 (so >= 10 for integer girth)
        # Actually, we need to ensure no 4, 6, 8 cycles.
        # BFS to depth 4 from each node.
        # If a node is reached at depth d1 and d2 with d1+d2 < 10?
        # Simpler: BFS to depth 4. If any node is visited twice (via different paths), cycle length <= 2*4 = 8?
        # No, if I visit u at depth 2 and depth 3, cycle length is 5 (odd).
        # We need girth >= 10. So no cycles < 10.
        # So no cycles of length 3, 4, 5, 6, 7, 8, 9.
        # BFS to depth 4 from each vertex.
        # If I find a back edge to a vertex at depth d' such that depth + d' + 1 < 10, then cycle < 10.
        
        for i in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[i] = 0
            queue = [i]
            for u in queue:
                if dist[u] >= 4: continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif dist[v] <= dist[u]:
                        # Cycle found?
                        # Length = dist[u] + dist[v] + 1
                        # If this is < 10, return it
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < 10:
                            return cycle_len
        return 10 # Or higher

    def random_4reg(n):
        # Generate random 4-regular graph
        # Use configuration model or swaps
        # Start with a ring and add chords
        # Or just random permutations
        adj = [set() for _ in range(n)]
        # Ensure degree 4
        # This is hard to do randomly simply.
        # Use a known generator:
        # 1. Create 2n pairs.
        # 2. Randomly connect.
        # 3. Remove multi-edges and loops.
        # 4. If fails, retry.
        
        # Better: Start with a 4-regular graph (e.g. circulant) and swap.
        # Circulant C_n(1, 2, n-1, n-2) is 4-regular for n > 4.
        # This graph has girth 4 (triangles? no, 4-cycles).
        # So we start with a 4-regular graph and swap.
        
        adj = [set() for _ in range(n)]
        for i in range(n):
            for j in [1, 2, n-1, n-2]:
                v = (i + j) % n
                if i != v and v not in adj[i]:
                    adj[i].add(v)
                    adj[v].add(i)
        
        # Verify
        for i in range(n):
            if len(adj[i]) != 4:
                return None
        return adj

    def swap_edges(adj, n):
        # Pick two edges (u,v) and (x,y) and swap to (u,x) and (v,y) or (u,y) and (v,x)
        # Must maintain simplicity.
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        
        if not edges:
            return False
            
        e1 = random.choice(edges)
        e2 = random.choice(edges)
        if e1 == e2:
            return False
            
        u, v = e1
        x, y = e2
        
        # Try swap 1: (u,x) and (v,y)
        # Remove (u,v) and (x,y)
        adj[u].remove(v)
        adj[v].remove(u)
        adj[x].remove(y)
        adj[y].remove(x)
        
        # Add (u,x) and (v,y)
        if x not in adj[u] and y not in adj[v]:
            adj[u].add(x)
            adj[x].add(u)
            adj[v].add(y)
            adj[y].add(v)
            return True
        else:
            # Revert
            adj[u].add(v)
            adj[v].add(u)
            adj[x].add(y)
            adj[y].add(x)
            
        # Try swap 2: (u,y) and (v,x)
        adj[u].remove(v)
        adj[v].remove(u)
        adj[x].remove(y)
        adj[y].remove(x)
        
        if y not in adj[u] and x not in adj[v]:
            adj[u].add(y)
            adj[y].add(u)
            adj[v].add(x)
            adj[x].add(v)
            return True
        else:
            # Revert
            adj[u].add(v)
            adj[v].add(u)
            adj[x].add(y)
            adj[y].add(x)
            
        return False

    best_n, best_edges = seed_construct()
    
    # Try to improve
    for n_target in [320, 400, 450]:
        if time.time() > deadline:
            break
        adj = random_4reg(n_target)
        if not adj:
            continue
            
        # Local search
        for _ in range(1000):
            if time.time() > deadline:
                break
            girth = get_girth(adj, n_target)
            if girth >= 10:
                # Convert to edge list
                edges = []
                for i in range(n_target):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                if n_target < best_n:
                    best_n, best_edges = n_target, edges
                break
            if not swap_edges(adj, n_target):
                continue
                
    return best_n, best_edges
