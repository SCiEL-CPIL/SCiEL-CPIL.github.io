import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    # 1. Try Local Search for small N
    # Target 936 for (3,16)
    if k == 3 and g == 16:
        target_n = 936
        # Try a few sizes
        for n in range(target_n, target_n + 100, 10):
            if time.time() > deadline - 5:
                break
            time_limit = min(5.0, deadline - time.time() - 2.0)
            if time_limit <= 0:
                break
            result = _ls_construct(n, k, g, time_limit)
            if result:
                return result
                
    # 2. Fallback to LUW (Valid, but large)
    return _luw_construct(k, g)

def _ls_construct(n, k, g, time_limit):
    if n % 2 != 0:
        n += 1
        
    edges = _gen_random_3reg(n)
    edge_set = set((min(u, v), max(u, v)) for u, v in edges)
    
    # Initial check
    girth = _get_girth(edges, n, g)
    if girth >= g:
        return n, edges
        
    start = time.time()
    
    while time.time() - start < time_limit:
        # Pick random edge
        u, v = random.choice(edges)
        
        # Try to swap
        improved = False
        for _ in range(20):
            x = random.randint(0, n - 1)
            y = random.randint(0, n - 1)
            if x == y or x in (u, v) or y in (u, v):
                continue
            if (min(x, y), max(x, y)) not in edge_set:
                continue
            
            e1 = (min(u, x), max(u, x))
            e2 = (min(v, y), max(v, y))
            
            if e1 in edge_set or e2 in edge_set or e1 == e2:
                continue
                
            # Swap
            edge_set.remove((min(u, v), max(u, v)))
            edge_set.remove((min(x, y), max(x, y)))
            edge_set.add(e1)
            edge_set.add(e2)
            
            new_edges = list(edge_set)
            new_girth = _get_girth(new_edges, n, g)
            
            if new_girth > girth:
                girth = new_girth
                edges = new_edges
                improved = True
                if girth >= g:
                    return n, edges
            else:
                # Revert
                edge_set.remove(e1)
                edge_set.remove(e2)
                edge_set.add((min(u, v), max(u, v)))
                edge_set.add((min(x, y), max(x, y)))
                
        if not improved:
            # Stagnation, try a different strategy or break
            pass

    return None

def _get_girth(edges, n, max_girth):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    min_g = max_girth + 1
    
    for s in range(n):
        visited = [-1] * n
        parent = [-1] * n
        depth = [0] * n
        visited[s] = 0
        q = deque([s])
        
        while q:
            u = q.popleft()
            for v in adj[u]:
                if visited[v] == -1:
                    visited[v] = 1
                    parent[v] = u
                    depth[v] = depth[u] + 1
                    q.append(v)
                elif parent[u] != v and parent[v] != u:
                    l = depth[u] + depth[v] + 1
                    if l < min_g:
                        min_g = l
        if min_g <= max_girth:
            return min_g
            
    return min_g

def _gen_random_3reg(n):
    stubs = list(range(n)) * 3
    random.shuffle(stubs)
    edges = set()
    for i in range(0, n * 3, 2):
        u = stubs[i]
        v = stubs[i + 1]
        if u == v:
            # Fix
            j = (i + 2) % (n * 3)
            while j == i or j == i + 1:
                j = (j + 2) % (n * 3)
            stubs[i + 1], stubs[j] = stubs[j], stubs[i + 1]
            v = stubs[i + 1]
        if (min(u, v), max(u, v)) in edges:
            # Fix multi
            j = (i + 2) % (n * 3)
            while j == i or j == i + 1:
                j = (j + 2) % (n * 3)
            stubs[i + 1], stubs[j] = stubs[j], stubs[i + 1]
            v = stubs[i + 1]
        edges.add((min(u, v), max(u, v)))
    return list(edges)

def _luw_construct(k, g):
    # Copy of seed logic
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
