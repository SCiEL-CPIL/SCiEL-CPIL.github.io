"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from collections import deque

def construct(k, g):
    # Random lift of Petersen graph + local search to achieve girth g
    
    base_edges = [
        (0,1), (1,2), (2,3), (3,4), (4,0),
        (5,7), (7,9), (9,6), (6,8), (8,5),
        (0,5), (1,6), (2,7), (3,8), (4,9)
    ]
    edge_keys = [(min(u,v), max(u,v)) for u, v in base_edges]
    
    def build_lift(m, twists):
        n = 10 * m
        edges = []
        for u, v in base_edges:
            key = (min(u,v), max(u,v))
            t = twists.get(key, 0)
            for i in range(m):
                u_lift = u * m + i
                v_lift = v * m + (i + t) % m
                edges.append((min(u_lift, v_lift), max(u_lift, v_lift)))
        return n, edges
    
    def compute_girth(n, edges):
        if n == 0:
            return float('inf')
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        min_girth = float('inf')
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            queue = deque([start])
            dist[start] = 0
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if v == parent[u]:
                        continue
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
        return min_girth
    
    def find_one_short_cycle(n, edges, target_g):
        """Find one cycle of length < target_g."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        for start in range(n):
            stack = [(start, [start], {start})]
            while stack:
                node, path, visited = stack.pop()
                if len(path) > target_g - 1:
                    continue
                for nb in adj[node]:
                    if nb == start and len(path) >= 3:
                        return list(path)
                    elif nb not in visited:
                        stack.append((nb, path + [nb], visited | {nb}))
        return None
    
    # Main search
    best_result = None
    best_n = float('inf')
    
    for m in range(60, 100, 2):
        n = 10 * m
        if n >= best_n:
            continue
        
        for attempt in range(50):
            random.seed(42 + m * 100 + attempt)
            twists = {key: random.randint(0, m-1) for key in edge_keys}
            n_lift, edges = build_lift(m, twists)
            
            girth = compute_girth(n_lift, edges)
            if girth >= g:
                if n_lift < best_n:
                    best_n = n_lift
                    best_result = (n_lift, edges)
                break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback: LUW construction
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
# EVOLVE-BLOCK-END
