"""Rewritten program: construction for (k,g)-cages with girth verification."""
import time
import random
from collections import deque

def _time_left(deadline, margin=2.0):
    return time.time() < deadline - margin

def _moore_lower(k, g):
    """Lower bound on number of vertices for k-regular graph of girth g."""
    if g % 2 == 1:
        h = (g - 3) // 2
        s = 1
        for i in range(1, h + 1):
            s = s * (k - 1) + 1
        return 1 + k * s
    else:
        h = g // 2 - 1
        s = 0
        for i in range(h + 1):
            s += (k - 1) ** i
        return 2 * s

def _has_short_cycle(adj, n, g):
    """Return True if there exists a cycle of length < g."""
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            if dist[u] + 1 >= g:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u]:
                    # Found a cycle
                    cycle_len = dist[u] + 1 + dist[v]
                    if cycle_len < g:
                        return True
    return False

def _random_k_regular(n, k, seed=42):
    """Generate a random simple k-regular graph."""
    random.seed(seed)
    if k >= n or k == 0:
        return None
    if k % 2 != 0 and n * k % 2 != 0:
        return None
    
    # Configuration model with fixes
    max_iter = 100
    for _ in range(max_iter):
        stubs = []
        for i in range(n):
            stubs.extend([i] * k)
        random.shuffle(stubs)
        adj = [set() for _ in range(n)]
        valid = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v:
                valid = False
                break
            if v in adj[u]:
                valid = False
                break
            adj[u].add(v)
            adj[v].add(u)
        if valid:
            return [list(s) for s in adj]
    return None

def construct(k, g):
    """Construct a k-regular graph of girth at least g with as few vertices as possible."""
    deadline = time.time() + 118.0
    best_n = None
    best_edges = None
    
    moore = _moore_lower(k, g)
    
    # Try increasing n from Moore bound
    max_search = moore + 300
    
    for n in range(moore, min(max_search, moore + 150)):
        if not _time_left(deadline):
            break
        
        for seed in range(30):
            if not _time_left(deadline):
                break
            
            adj = _random_k_regular(n, k, seed=seed + n * 7919)
            if adj is None:
                continue
            
            if not _has_short_cycle(adj, n, g):
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                best_n = n
                best_edges = edges
                break
        
        if best_n is not None:
            break
    
    if best_n is not None:
        return best_n, best_edges
    
    # Fallback: Lazebnik-Ustimenko-Woldar
    q = k
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
