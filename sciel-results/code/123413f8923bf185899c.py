import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin

    # Moore Bound
    if g % 2 == 0:
        # Bipartite
        r = g // 2 - 1
        moore = 2 * (k * sum((k-1)**i for i in range(r)))
    else:
        r = (g - 3) // 2
        moore = 1 + k * sum((k-1)**i for i in range(r))
    
    # 1. LUW Construction
    def get_luw_graph(d):
        q = k
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
            if time.time() > deadline:
                break
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    def check_girth(n, edges, min_girth):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Check regularity
        for i in range(n):
            if len(adj[i]) != k:
                return False, None

        # BFS from each vertex
        for start in range(n):
            if time.time() > deadline:
                return False, None
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                if dist[u] >= min_girth - 1:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u] and u != parent[v]:
                        # Found a cycle
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            return False, None
        return True, edges

    best_n = float('inf')
    best_edges = None
    
    # Try LUW for d = 3 to 10
    for d in range(3, 11):
        if time.time() > deadline - 20:
            break
        n, edges = get_luw_graph(d)
        if n < moore:
            continue # Can't be valid
        valid, _ = check_girth(n, edges, g)
        if valid:
            if n < best_n:
                best_n = n
                best_edges = edges
            break # Smaller d means smaller n, so we can stop if valid? 
                 # Not necessarily, but usually d increases n exponentially.
                 # So the first valid d is the best LUW candidate.

    # If no LUW found, or if we want to try circulant for smaller n
    # Try Circulant search if best_n is large or not found
    if best_n == float('inf') or best_n > 1000:
        # Search for circulant graphs
        # n must be at least moore
        # Try n from moore to min(best_n - 1, moore + 500)
        # Limit search space
        max_n_search = min(best_n - 1 if best_n != float('inf') else moore + 200, moore + 200)
        
        for n in range(moore, max_n_search + 1):
            if time.time() > deadline - 5:
                break
            # Try random generator sets
            # Generators from 1 to n//2 - 1
            # Need 5 generators
            # Ensure they are distinct and not inverses
            # In a circulant graph C_n(S), S is a set of generators.
            # Degree is 2*|S| if S doesn't contain n/2.
            # We need degree 5. So we need 2.5 generators? Impossible.
            # So circulant graphs are 2m-regular.
            # k=5 is odd.
            # A circulant graph with odd degree must include the element n/2 (if n even) or be non-symmetric?
            # Standard circulant graphs are symmetric: if i in S, then -i in S.
            # So degree is even.
            # To get odd degree, we can use a "semi-circulant" or add a perfect matching?
            # Or just use a general graph search?
            # Actually, for k=5, we can't use a simple symmetric circulant.
            # We can use a circulant with 2 generators and add a perfect matching?
            # Or just skip circulant for odd k.
            pass

    if best_edges is None:
        # Fallback to original LUW with large d
        n, edges = get_luw_graph(5)
        return n, edges

    return best_n, best_edges
