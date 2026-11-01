"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0
    
    q = k
    
    def get_luw_graph(d, q):
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: (x * y) % q
            
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
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges = []
        
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
                    
        return len(order), edges

    def has_short_cycle(n, edges, g_limit):
        """
        Returns True if there is a cycle of length < g_limit.
        """
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Check for cycles of length 4, 6, ..., up to g_limit-2 (if even)
        # We only need to check up to g_limit - 1.
        # Since graph is bipartite, only even cycles.
        max_len = g_limit - 1
        if max_len % 2 != 0:
            max_len -= 1
            
        # BFS from each node
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            
            while queue:
                u = queue.pop(0)
                if dist[u] >= max_len // 2:
                    continue
                    
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        # Potential cycle
                        # Calculate cycle length
                        # LCA of u and v
                        # This is expensive.
                        # Instead, we can just check if dist[u] + dist[v] + 1 < g_limit
                        # is NOT sufficient because the paths might not be disjoint.
                        # However, for small girth, this is a good heuristic.
                        # To be exact, we should find the LCA.
                        
                        # Let's find LCA
                        a = u
                        b = v
                        da = dist[a]
                        db = dist[b]
                        while da != db:
                            if da > db:
                                a = parent[a]
                                da -= 1
                            else:
                                b = parent[b]
                                db -= 1
                        while a != b:
                            a = parent[a]
                            b = parent[b]
                            da -= 1
                            db -= 1
                        # LCA is a
                        # Cycle length = dist[u] + dist[v] - 2*dist[a] + 1
                        cycle_len = dist[u] + dist[v] - 2 * dist[a] + 1
                        if cycle_len < g_limit:
                            return True
                            
        return False

    best_n = None
    best_edges = None
    
    # Try d values
    for d in range(3, 10):
        if time.time() > deadline:
            break
        try:
            n, edges = get_luw_graph(d, q)
            if has_short_cycle(n, edges, g):
                continue
            if best_n is None or n < best_n:
                best_n = n
                best_edges = edges
            break # Found the smallest valid d
        except Exception:
            continue
            
    if best_n is None:
        # Fallback
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        best_n, best_edges = get_luw_graph(d, q)
        
    return best_n, best_edges

# EVOLVE-BLOCK-END
