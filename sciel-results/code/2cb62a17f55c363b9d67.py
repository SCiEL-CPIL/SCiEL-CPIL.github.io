"""Construct a 3-regular graph with girth >= g on as few vertices as possible."""
import time
import random
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    deadline = time.time() + 135  # safety margin for 150s limit
    
    def check_girth(n, adj, k, g):
        """Return True if girth >= g. Fast tree-level check."""
        radius = g // 2 - 1
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            count = [0] * (radius + 1)
            count[0] = 1
            while queue:
                u = queue.popleft()
                if dist[u] >= radius:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        if dist[v] <= radius:
                            count[dist[v]] += 1
                            queue.append(v)
            
            expected = [1] + [k * (k - 1) ** i for i in range(radius)]
            if count != expected:
                return False
        return True
    
    def random_regular_graph(n, k, rng):
        """Generate a random k-regular graph using configuration model."""
        if n * k % 2 != 0:
            return None
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        rng.shuffle(stubs)
        adj = [[] for _ in range(n)]
        for i in range(0, n * k, 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v:
                return None
            if v in adj[u]:
                return None
            adj[u].append(v)
            adj[v].append(u)
        return adj
    
    def is_connected(n, adj):
        visited = [False] * n
        queue = deque([0])
        visited[0] = True
        count = 1
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    count += 1
                    queue.append(v)
        return count == n
    
    def edge_swap(adj, n, k, rng):
        """Perform a random edge swap to modify the graph while preserving k-regularity."""
        # Pick two random vertices with the same degree
        u = rng.randint(0, n - 1)
        v = rng.randint(0, n - 1)
        if u == v:
            return False
        # Find an edge from u and an edge from v
        # Remove (u, a) and (v, b), add (u, b) and (v, a)
        a_candidates = list(adj[u])
        b_candidates = list(adj[v])
        if not a_candidates or not b_candidates:
            return False
        a = rng.choice(a_candidates)
        b = rng.choice(b_candidates)
        # Check constraints
        if a == b:
            return False
        if b in adj[u]:
            return False
        if a in adj[v]:
            return False
        # Perform swap
        adj[u].remove(a)
        adj[u].append(b)
        adj[a].remove(u)
        adj[a].append(v)
        adj[v].remove(b)
        adj[v].append(a)
        adj[b].remove(v)
        adj[b].append(u)
        return True
    
    # Moore bound for k=3, girth=16: n >= 2*3*sum(2^i for i=0..5) = 378
    # Known lower bound is 512. Start from there.
    n_start = max(512, 2 * k * sum((k - 1) ** i for i in range(g // 2 - 1)))
    n_max = 1000
    
    rng = random.Random(42)
    
    best_n = None
    best_edges = None
    
    for n in range(n_start, n_max + 1, 4):
        if time.time() > deadline:
            break
        if n * k % 2 != 0:
            continue
        
        attempts = 0
        max_attempts = 30
        while attempts < max_attempts:
            attempts += 1
            if time.time() > deadline:
                break
            
            adj = random_regular_graph(n, k, rng)
            if adj is None:
                continue
            
            if not is_connected(n, adj):
                continue
            
            if check_girth(n, adj, k, g):
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges
    
    # Fallback: LUW construction
    q = k
    d = g - 5 if (g - 5) % 2 else g - 4
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
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
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
            yield (1 - side, tuple(y))
    
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
# EVOLVE-BLOCK-END
