"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    # Moore bound for odd girth
    def moore_bound(k, g):
        if g % 2 == 0:
            # even girth
            n = 2 * k
            for i in range(1, g // 2):
                n += 2 * k * (k - 1) ** i
            return n
        else:
            # odd girth
            n = k
            for i in range(1, (g + 1) // 2):
                n += k * (k - 1) ** i
            return n
    
    mb = moore_bound(k, g)
    
    # Try random regular graphs with girth check
    def check_girth(adj, n, min_girth):
        """Check if graph has girth >= min_girth using BFS from each vertex."""
        for s in range(n):
            # BFS to find shortest cycle containing s
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u]:
                        # Found a cycle
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            return False
        return True
    
    def random_regular_graph(n, k, seed=None):
        """Generate a random k-regular graph on n vertices using configuration model."""
        if seed is not None:
            random.seed(seed)
        # Configuration model
        if n * k % 2 != 0:
            return None
        half_edges = []
        for i in range(n):
            for _ in range(k):
                half_edges.append(i)
        random.shuffle(half_edges)
        adj = [[] for _ in range(n)]
        for i in range(0, len(half_edges), 2):
            u, v = half_edges[i], half_edges[i + 1]
            if u == v:
                continue
            if v not in adj[u]:
                adj[u].append(v)
                adj[v].append(u)
        # Check if regular
        for i in range(n):
            if len(adj[i]) != k:
                return None
        return adj
    
    # Try to find a valid random regular graph
    best_n = None
    best_edges = None
    
    for trial in range(50):
        if time.time() > deadline:
            break
        
        # Try sizes from Moore bound upward
        for n in range(mb, mb + 100):
            if time.time() > deadline:
                break
            if n * k % 2 != 0:
                continue
            
            adj = random_regular_graph(n, k, seed=trial * 1000 + n)
            if adj is None:
                continue
            
            if check_girth(adj, n, g):
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n, edges
    
    # Fallback: LUW construction
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
# EVOLVE-BLOCK-END
