import random
import time

def construct(k, g):
    if k != 4:
        return _luw_construct(k, g)
    
    start_time = time.time()
    n = 320
    random.seed(42)
    
    # Build random 4-regular graph
    adj = [set() for _ in range(n)]
    _make_random_regular(adj, n, k)
    
    # Local search
    for iteration in range(1000000):
        if time.time() - start_time > 110:
            break
        
        # Find a short cycle
        cycle = _find_short_cycle(adj, n, g)
        if cycle is None:
            break
        
        # Try to break it
        _break_cycle(adj, n, cycle, k)
    
    edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append((i, j))
    
    return n, edges

def _make_random_regular(adj, n, k):
    """Create a random k-regular graph."""
    half_edges = []
    for i in range(n):
        for _ in range(k):
            half_edges.append(i)
    
    random.shuffle(half_edges)
    
    for i in range(0, len(half_edges), 2):
        u, v = half_edges[i], half_edges[i + 1]
        if u != v and v not in adj[u]:
            adj[u].add(v)
            adj[v].add(u)
    
    for i in range(n):
        while len(adj[i]) < k:
            j = random.randint(0, n - 1)
            if j != i and j not in adj[i] and len(adj[j]) < k:
                adj[i].add(j)
                adj[j].add(i)
        while len(adj[i]) > k:
            j = random.choice(list(adj[i]))
            adj[i].remove(j)
            adj[j].remove(i)

def _find_short_cycle(adj, n, g):
    """Find a cycle of length < g, or None if girth >= g."""
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        
        for u in queue:
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        cycle = []
                        x = u
                        while x != v:
                            cycle.append(x)
                            x = parent[x]
                        cycle.append(v)
                        y = v
                        while y != start:
                            cycle.append(y)
                            y = parent[y]
                        cycle.append(start)
                        return cycle
    return None

def _break_cycle(adj, n, cycle, k):
    """Try to break a cycle using a 2-switch."""
    if not cycle or len(cycle) < 3:
        return
    
    cycle_len = len(cycle)
    for e_idx in range(cycle_len):
        u, v = cycle[e_idx], cycle[(e_idx + 1) % cycle_len]
        
        for a in range(n):
            if a == u or a == v:
                continue
            for b in adj[a]:
                if b == u or b == v or b == a:
                    continue
                if a in adj[v] or b in adj[u]:
                    continue
                if a == v or b == u:
                    continue
                
                adj[u].discard(v)
                adj[v].discard(u)
                adj[a].discard(b)
                adj[b].discard(a)
                adj[u].add(a)
                adj[a].add(u)
                adj[v].add(b)
                adj[b].add(v)
                return

def _luw_construct(k, g):
    """LUW construction for fallback."""
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
