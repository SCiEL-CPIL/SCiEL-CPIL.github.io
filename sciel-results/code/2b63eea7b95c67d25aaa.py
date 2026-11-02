import random
import collections
import time

def construct(k, g):
    # Moore bound for k=3, g=16 is 512
    # We search for n in [512, 1200]
    
    def generate_random_regular(n, k):
        # Configuration model
        for _ in range(20): # Retry a few times if we get loops/multiedges
            stubs = [i for i in range(n) for _ in range(k)]
            random.shuffle(stubs)
            adj = {i: set() for i in range(n)}
            valid = True
            for i in range(0, n * k, 2):
                u = stubs[i]
                v = stubs[i + 1]
                if u == v or v in adj[u]:
                    valid = False
                    break
                adj[u].add(v)
                adj[v].add(u)
            if valid:
                # Final check
                for u in range(n):
                    if len(adj[u]) != k:
                        valid = False
                        break
                if valid:
                    return adj
        return None

    def find_short_cycle_edge(adj, n, g):
        # Returns (u, v) if a cycle of length < g is found, else None
        max_depth = g // 2 - 1
        for start in range(n):
            dist = {start: 0}
            parent = {start: -1}
            queue = collections.deque([start])
            
            while queue:
                u = queue.popleft()
                if dist[u] > max_depth:
                    continue
                for v in adj[u]:
                    if v == parent[u]:
                        continue
                    if v in dist:
                        length = dist[u] + dist[v] + 1
                        if length < g:
                            return (u, v)
                    else:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
        return None

    def local_search(n, k, g, time_limit):
        start_time = time.time()
        for attempt in range(20): # Limit restarts
            if time.time() - start_time > time_limit:
                break
            adj = generate_random_regular(n, k)
            if adj is None:
                continue
            
            for _ in range(5000): # Max iterations
                if time.time() - start_time > time_limit:
                    return None
                
                cycle_edge = find_short_cycle_edge(adj, n, g)
                if cycle_edge is None:
                    return adj
                
                u, v = cycle_edge
                swapped = False
                for _ in range(50): # Try 50 random swaps
                    x = random.randint(0, n - 1)
                    if x == u or x == v:
                        continue
                    neighbors = list(adj[x])
                    if not neighbors:
                        continue
                    y = random.choice(neighbors)
                    if y == u or y == v:
                        continue
                    if x in adj[u] or y in adj[v]:
                        continue
                    if u == x or v == y:
                        continue
                    
                    # Swap (u,v) and (x,y) with (u,x) and (v,y)
                    adj[u].remove(v)
                    adj[v].remove(u)
                    adj[x].remove(y)
                    adj[y].remove(x)
                    
                    adj[u].add(x)
                    adj[x].add(u)
                    adj[v].add(y)
                    adj[y].add(v)
                    
                    swapped = True
                    break
                if not swapped:
                    break
        
        return None

    min_n = 512
    max_n = 1200
    
    start_time = time.time()
    max_time = 115
    
    for n in range(min_n, max_n + 1, 2):
        if time.time() - start_time > max_time:
            break
        
        remaining = max_time - (time.time() - start_time)
        if remaining <= 0:
            break
        
        t_alloc = min(5, remaining / 2)
        
        adj = local_search(n, k, g, t_alloc)
        if adj:
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            return n, edges

    return _luw_fallback(k, g)

def _luw_fallback(k, g):
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
