"""De Bruijn-type finite-state construction for regular graphs of large girth."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(k, g):
    t0 = time.time()
    deadline = 120.0  # seconds; evaluator kills at 150s, keep margin

    # Fallback: Lazebnik-Ustimenko-Woldar graph
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

    # Helper: Generate a simple random 3-regular graph on n vertices
    def generate_random_3_reg_simple(n):
        if n < 4 or n % 2 != 0:
            return None
        
        # Configuration model
        stubs = [i for i in range(n) for _ in range(3)]
        
        for _ in range(100): # Try 100 times to get a simple graph
            random.shuffle(stubs)
            edges = set()
            valid = True
            # Pair up stubs
            for i in range(len(stubs) // 2):
                u, v = stubs[2*i], stubs[2*i+1]
                if u == v:
                    valid = False
                    break
                key = (min(u, v), max(u, v))
                if key in edges:
                    valid = False
                    break
                edges.add(key)
            
            if not valid:
                continue
            
            # Check degree (should be 3 if no loops/multi-edges and simple pairing)
            # Actually, simple pairing ensures degree 3 if no loops.
            # Multi-edges are handled by the set.
            # So if valid is True, we have a simple 3-regular graph.
            return list(edges)
        
        return None

    # Helper: Check girth
    def check_girth(n, edges, min_girth):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # BFS from each vertex
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            qi = 0
            while qi < len(queue):
                u = queue[qi]
                qi += 1
                if dist[u] >= min_girth // 2:
                    continue
                for v in adj[u]:
                    if v == parent[u]:
                        continue
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        # Cycle found
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            return False
        return True

    # Main logic
    if k == 3 and g == 16:
        # Search for smaller graph using random construction
        # Start from Moore bound 512
        # Step by 2 to ensure even n
        for n in range(512, 1200, 2):
            if time.time() - t0 > deadline - 5:
                break
            
            # Try a few random graphs
            # Increase attempts for smaller n?
            # For n=512, girth 16 is unlikely.
            # For n=936, it's possible.
            attempts = 20
            for _ in range(attempts):
                if time.time() - t0 > deadline - 5:
                    break
                edges = generate_random_3_reg_simple(n)
                if edges is None:
                    continue
                if check_girth(n, edges, g):
                    return n, edges
        
        # Fallback to LUW
        return luw_construct(k, g)
    else:
        return luw_construct(k, g)
# EVOLVE-BLOCK-END
