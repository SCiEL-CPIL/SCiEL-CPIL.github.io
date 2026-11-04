import time
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0
    
    def check_girth_fast(n, edges, g, time_limit=None):
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        
        max_depth = (g - 1) // 2
        
        for s in range(n):
            if time_limit and time.time() > time_limit:
                return -1
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            qi = 0
            while qi < len(queue):
                node = queue[qi]
                qi += 1
                if dist[node] > max_depth:
                    continue
                for nb in adj[node]:
                    if nb == parent[node]:
                        continue
                    if dist[nb] != -1:
                        cycle_len = dist[node] + dist[nb] + 1
                        if cycle_len < g:
                            return cycle_len
                    else:
                        dist[nb] = dist[node] + 1
                        parent[nb] = node
                        queue.append(nb)
        return g

    def is_valid(n, edges, g, time_limit=None):
        if len(edges) != n * k // 2:
            return False
            
        edge_set = set()
        for u, v in edges:
            if u == v:
                return False
            if u > v:
                u, v = v, u
            if (u, v) in edge_set:
                return False
            edge_set.add((u, v))
            
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        for d in deg:
            if d != k:
                return False

        girth = check_girth_fast(n, edges, g, time_limit)
        if girth == -1:
            return False
        return girth >= g

    def generate_random_cubic(n):
        half_edges = []
        for i in range(n):
            half_edges.extend([i, i, i])
        random.shuffle(half_edges)
        edges = []
        for i in range(0, 3 * n, 2):
            edges.append((half_edges[i], half_edges[i + 1]))
        return edges

    def luw_construct(k, g):
        if time.time() > deadline - 20:
            return None
            
        q = k
        d = max(3, (g - 2) // 2)
        
        def build_luw_graph(q, d, time_limit=None):
            if q == 4:
                table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
                add = lambda x, y: x ^ y
                sub = lambda x, y: x ^ y
                mul = lambda x, y: table[x][y]
            else:
                add = lambda x, y: (x + y) % q
                sub = lambda x, y: (x - y) % q
                mul = lambda x, y: x * y % q
            
            def neighbours(vertex):
                side, x = vertex
                for first in range(q):
                    y = [None, first] + [0] * (d - 1)
                    for c in range(2, d + 1):
                        if c == 2:
                            a, b = 1, 1
                        elif c == 3:
                            a, b = 2, 1
                        elif c == 4:
                            a, b = 1, 2
                        else:
                            r, c0 = (c - 5) % 4, c - (c - 5) % 4
                            a, b = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
                        
                        y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                    yield 1 - side, tuple(y)
            
            start = (0, (None,) + (0,) * d)
            index = {start: 0}
            order = [start]
            edges = []
            
            for i in range(len(order)):
                if time_limit and time.time() > time_limit:
                    return None
                vertex = order[i]
                for other in neighbours(vertex):
                    if other not in index:
                        index[other] = len(order)
                        order.append(other)
                    if vertex[0] == 0:
                        edges.append((index[vertex], index[other]))
            
            return len(order), edges

        result = build_luw_graph(q, d, time_limit=deadline - 5)
        if result:
            n, edges = result
            if is_valid(n, edges, g, time_limit=deadline - 2):
                return n, edges
        return None

    if k != 3:
        return luw_construct(k, g)
    
    n = 512
    max_n_luw = 2 * (3 ** ((g - 2) // 2 + 1))
    
    # Random Search
    while n < max_n_luw and time.time() < deadline - 30:
        attempts = 0
        max_attempts = 500
        while attempts < max_attempts and time.time() < deadline - 30:
            edges = generate_random_cubic(n)
            if is_valid(n, edges, g, time_limit=deadline - 30):
                return n, edges
            attempts += 1
        n += 2
        
    # Fallback
    result = luw_construct(k, g)
    if result:
        return result
        
    return 0, []
