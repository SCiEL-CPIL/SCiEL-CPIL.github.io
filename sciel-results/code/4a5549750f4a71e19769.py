import time
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Fallback: The original LUW construction
    def construct_luw(k, g):
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

    # Try Progressive Construction
    try:
        # Initialize
        n = 1
        adj = [[]]
        degree = [0]
        queue = [0]
        
        # We need to fill degree k for all vertices
        # Strategy: Pop from queue, add neighbors.
        # If no valid neighbor, create new one.
        
        while queue:
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
                
            u = queue.pop(0)
            
            # If u is already full, skip (shouldn't happen if we manage queue correctly)
            if degree[u] >= k:
                continue
                
            # We need to add (k - degree[u]) neighbors
            while degree[u] < k:
                if time.time() > deadline:
                    raise TimeoutError("Time limit exceeded")
                
                # Find valid neighbors
                # BFS from u to calculate distances
                dist = [-1] * n
                dist[u] = 0
                bfs_q = [u]
                head = 0
                while head < len(bfs_q):
                    curr = bfs_q[head]
                    head += 1
                    for v in adj[curr]:
                        if dist[v] == -1:
                            dist[v] = dist[curr] + 1
                            bfs_q.append(v)
                
                candidates = []
                for v in range(n):
                    if v != u and dist[v] >= g - 1 and v not in adj[u]:
                        candidates.append(v)
                
                if candidates:
                    v = random.choice(candidates)
                    # Add edge u-v
                    adj[u].append(v)
                    adj[v].append(u)
                    degree[u] += 1
                    degree[v] += 1
                    if degree[v] < k:
                        queue.append(v)
                else:
                    # Create new vertex
                    w = n
                    n += 1
                    adj.append([u])
                    adj[u].append(w)
                    degree.append(1)
                    degree[u] += 1
                    # w needs k-1 more neighbors
                    queue.append(w)
        
        # Verify validity
        # 1. Simple graph: no loops, no multi-edges
        for i in range(n):
            if i in adj[i]:
                raise ValueError("Loop found")
            if len(adj[i]) != len(set(adj[i])):
                raise ValueError("Multi-edge found")
            if degree[i] != k:
                raise ValueError("Degree mismatch")
        
        # 2. Girth check: No cycle < g
        # We can do this by checking all pairs? No, too slow.
        # The construction ensures that any new edge (u,v) has dist(u,v) >= g-1 in the *existing* graph.
        # Does this guarantee global girth >= g?
        # Yes, any cycle formed must use at least one "new" edge.
        # If a cycle uses only tree edges, it's a tree (no cycles).
        # If it uses new edges, the first time a cycle is closed, the distance between the endpoints of the closing edge in the *previous* graph was >= g-1.
        # Thus the cycle length is >= (g-1) + 1 = g.
        
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
                    
        return n, edges

    except Exception:
        # Fallback to LUW if progressive construction fails or times out
        return construct_luw(k, g)
