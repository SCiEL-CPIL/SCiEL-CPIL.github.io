import time
import random
import sys

def _luw_construct(k, g):
    # Lazebnik-Ustimenko-Woldar construction for the 5-regular girth-9 case
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

def _generate_random_regular(n, k):
    """Generate a random simple k-regular graph on n vertices using configuration model."""
    if n * k % 2 != 0:
        return None
    half_edges = []
    for i in range(n):
        for _ in range(k):
            half_edges.append(i)
    
    attempts = 0
    max_attempts = 1000
    while attempts < max_attempts:
        attempts += 1
        random.shuffle(half_edges)
        edges = set()
        adj = [set() for _ in range(n)]
        valid = True
        for i in range(0, len(half_edges), 2):
            u = half_edges[i]
            v = half_edges[i+1]
            if u == v:
                valid = False
                break
            if v in adj[u]:
                valid = False
                break
            edges.add((u, v) if u < v else (v, u))
            adj[u].add(v)
            adj[v].add(u)
        if valid:
            return edges, adj
    return None

def _find_short_cycle(adj, max_len=8):
    """Find a cycle of length < 9 (i.e., <= 8) if one exists.
    Returns (u, v, L) where (u,v) is an edge in the cycle and L is the cycle length, or None."""
    n = len(adj)
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        qi = 0
        while qi < len(queue):
            u = queue[qi]
            qi += 1
            if dist[u] >= max_len - 1:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    if v == parent[u]:
                        continue
                    L = dist[u] + dist[v] + 1
                    if L <= max_len:
                        return (u, v, L)
    return None

def _try_switching(n, k, g, time_limit=100.0):
    start_time = time.time()
    
    # 1. Generate initial graph
    res = _generate_random_regular(n, k)
    if res is None:
        return None
    edges_set, adj = res
    
    # Check if girth is already >= 9
    if _find_short_cycle(adj, 8) is None:
        edge_list = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edge_list.append((i, j))
        return n, edge_list
    
    # 2. Switching loop
    while time.time() - start_time < time_limit:
        # Find a short cycle
        cycle_info = _find_short_cycle(adj, 8)
        if cycle_info is None:
            edge_list = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        edge_list.append((i, j))
            return n, edge_list
            
        u, v, L = cycle_info
        
        # Try to find a switch that breaks this cycle
        success = False
        for _ in range(200):
            x = random.randint(0, n-1)
            if x == u or x == v:
                continue
            if len(adj[x]) == 0:
                continue
            y = random.choice(list(adj[x]))
            if y == u or y == v or y == x:
                continue
            if y in adj[v]:
                continue
            if x in adj[u]:
                continue
            
            # Perform switch: remove (u,v), (x,y); add (u,x), (v,y)
            adj[u].remove(v)
            adj[v].remove(u)
            adj[x].remove(y)
            adj[y].remove(x)
            
            adj[u].add(x)
            adj[x].add(u)
            adj[v].add(y)
            adj[y].add(v)
            
            success = True
            break
            
        if not success:
            continue
            
    return None

def construct(k, g):
    if k == 5 and g == 9:
        target_ns = [1116, 1120, 1150, 1200]
        time_per_attempt = 25.0
        
        for n in target_ns:
            res = _try_switching(k, g, n, time_limit=time_per_attempt)
            if res is not None:
                return res
                
    # Fallback to LUW
    return _luw_construct(k, g)
