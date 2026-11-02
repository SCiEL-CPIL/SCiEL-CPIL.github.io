import time
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

    def is_prime_power(n):
        if n < 2:
            return False
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
            if p > n:
                break
            if n % p == 0:
                m = n
                while m % p == 0:
                    m //= p
                if m == 1:
                    return True
        return False

    def luw_construct(k, g):
        q = k
        if not is_prime_power(q) and q != 4:
            return None
        
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

    def check_girth(adj, g):
        n = len(adj)
        for start in range(n):
            visited = [-1] * n
            parent = [-1] * n
            queue = [(start, 0)]
            visited[start] = 0
            while queue:
                u, d = queue.pop(0)
                for v in adj[u]:
                    if v == parent[u]:
                        continue
                    if visited[v] == -1:
                        visited[v] = d + 1
                        parent[v] = u
                        queue.append((v, d + 1))
                    else:
                        cycle_len = d + visited[v] + 1
                        if cycle_len < g:
                            return False
        return True

    def random_regular_graph(n, k):
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        random.shuffle(stubs)
        adj = [[] for _ in range(n)]
        for i in range(0, n * k, 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                return None
            if v in adj[u]:
                return None
            adj[u].append(v)
            adj[v].append(u)
        return adj

    best_n = float('inf')
    best_edges = []
    
    luw_res = luw_construct(k, g)
    if luw_res:
        n_luw, edges_luw = luw_res
        adj_luw = [[] for _ in range(n_luw)]
        for u, v in edges_luw:
            adj_luw[u].append(v)
            adj_luw[v].append(u)
        if check_girth(adj_luw, g):
            best_n = n_luw
            best_edges = edges_luw

    depth = (g - 1) // 2
    moore_bound = 1 + k
    term = k
    for i in range(1, depth):
        term *= (k - 1)
        moore_bound += term
        
    max_n_to_try = best_n if best_n < float('inf') else moore_bound + 2000
    
    current_n = moore_bound
    while current_n < max_n_to_try and time.time() < deadline:
        success = False
        attempts = 0
        max_attempts = 50
        
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            adj = random_regular_graph(current_n, k)
            if adj is not None:
                if check_girth(adj, g):
                    edges = []
                    for u in range(current_n):
                        for v in adj[u]:
                            if u < v:
                                edges.append((u, v))
                    return current_n, edges
        current_n += 1
        
    if best_edges:
        return best_n, best_edges
        
    return 0, []
