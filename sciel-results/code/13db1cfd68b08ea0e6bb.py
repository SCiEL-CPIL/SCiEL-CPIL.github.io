import time
import random
from collections import deque

_DEADLINE = 140.0

def _luw_construct(k, g):
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

def _build_adjacency(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _compute_girth(adj, n):
    min_girth = float('inf')
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    clen = dist[u] + dist[v] + 1
                    if clen < min_girth:
                        min_girth = clen
        if min_girth <= 3:
            break
    return min_girth if min_girth != float('inf') else -1

def _short_cycle_exists(adj, n, g):
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    clen = dist[u] + dist[v] + 1
                    if clen < g:
                        return True
    return False

def _random_2switch(adj, n, g, max_checks=50):
    k = len(adj[0])
    for _ in range(max_checks):
        u = random.randrange(n)
        v = random.choice(list(adj[u]))
        w = random.randrange(n)
        if w == u or w == v:
            continue
        if w not in adj[u]:
            x = random.choice(list(adj[w]))
            if x == u or x == v or x == w:
                continue
            if x in adj[u] or w in adj[v]:
                continue
            # Check girth
            old_edges = [(u, v), (v, u), (w, x), (x, w)]
            # Remove old edges
            adj[u].discard(v)
            adj[v].discard(u)
            adj[w].discard(x)
            adj[x].discard(w)
            # Add new edges
            adj[u].add(w)
            adj[w].add(u)
            adj[v].add(x)
            adj[x].add(v)
            if _short_cycle_exists(adj, n, g):
                # Revert
                adj[u].discard(w)
                adj[w].discard(u)
                adj[v].discard(x)
                adj[x].discard(v)
                adj[u].add(v)
                adj[v].add(u)
                adj[w].add(x)
                adj[x].add(w)
            else:
                return True
    return False

def _greedy_switch(adj, n, g, time_limit):
    start_time = time.time()
    # Initial girth check
    cur_girth = _compute_girth(adj, n)
    if cur_girth >= g:
        return adj
    
    while time.time() - start_time < time_limit:
        # Find all short cycles (girth < g)
        # Try random 2-switches
        for _ in range(200):
            if time.time() - start_time > time_limit:
                break
            if _random_2switch(adj, n, g, max_checks=1):
                # Recheck girth
                if not _short_cycle_exists(adj, n, g):
                    return adj
    return None

def construct(k, g):
    if k == 5 and g == 9:
        # Use LUW as base, then try to improve via switching
        n_luw, edges_luw = _luw_construct(k, g)
        adj = _build_adjacency(edges_luw, n_luw)
        
        # Verify LUW girth
        girth = _compute_girth(adj, n_luw)
        if girth >= g:
            return n_luw, edges_luw
        
        # Try switching to improve
        time_left = _DEADLINE - time.time()
        if time_left > 30:
            improved = _greedy_switch(adj, n_luw, g, time_left - 20)
            if improved is not None:
                edges = []
                for i in range(n_luw):
                    for j in improved[i]:
                        if i < j:
                            edges.append((i, j))
                return n_luw, edges
        
        return n_luw, edges_luw
    
    return _luw_construct(k, g)
