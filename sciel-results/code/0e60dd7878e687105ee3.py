import time
from sympy import isprime, nextprime

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    if k != 4:
        # Fallback for other k if necessary, though task specifies k=4
        # Using LUW for generality
        q = k
        d = g - 5
        if d % 2 == 0:
            d += 1
        d = max(3, d)
        
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
        
        def neighbours_luw(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield (1 - side), tuple(y)
        
        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges = []
        
        for vertex in order:
            for other in neighbours_luw(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
            
            if time.time() > deadline:
                break
        
        return len(order), edges

    # Specific optimization for k=4, g=10
    # Try Cayley graphs on Z_p
    # Moore bound is 242
    
    def check_girth(n, adj, min_girth):
        # BFS from each node to find shortest cycle
        # Since graph is undirected, we can use BFS to find distance to all nodes
        # But we need to detect cycles.
        # A simple way: For each edge (u, v), check if there is a path of length < g-1 between u and v not using the edge.
        # Or just run BFS from every node and track parent.
        
        for start in range(n):
            # BFS
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
                        # Found a cycle
                        # Cycle length = dist[u] + dist[v] + 1
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            return False
        return True

    p = 243
    while True:
        if time.time() > deadline:
            break
            
        if not isprime(p):
            p = nextprime(p)
            continue
            
        n = p
        # Generators a, b
        # a < b
        # a, b in 1..p/2
        found = False
        for a in range(1, p // 2 + 1):
            for b in range(a + 1, p // 2 + 1):
                # Quick checks
                if 2 * a % p == 0 or 2 * b % p == 0:
                    continue
                if (a + b) % p == 0 or (a - b) % p == 0:
                    continue
                if (a + b) * 2 % p == 0:
                    continue
                
                # Build adjacency list
                adj = [[] for _ in range(n)]
                edges = []
                
                for u in range(n):
                    v1 = (u + a) % p
                    v2 = (u - a) % p
                    v3 = (u + b) % p
                    v4 = (u - b) % p
                    
                    adj[u] = [v1, v2, v3, v4]
                    if u < v1: edges.append((u, v1))
                    if u < v2: edges.append((u, v2))
                    if u < v3: edges.append((u, v3))
                    if u < v4: edges.append((u, v4))
                
                if check_girth(n, adj, g):
                    return n, edges
            if time.time() > deadline:
                break
        p = nextprime(p)
        
    # Fallback to LUW if no Cayley graph found
    q = 4
    d = 5
    table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
    add = lambda x, y: x ^ y
    sub = lambda x, y: x ^ y
    mul = lambda x, y: table[x][y]
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    def neighbours_luw(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield (1 - side), tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        for other in neighbours_luw(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
        
        if time.time() > deadline:
            break
    
    return len(order), edges
