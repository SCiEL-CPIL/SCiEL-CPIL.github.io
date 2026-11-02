import time
import random
import collections

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Target vertex count based on Moore bound or known constructions
    # Moore bound for k=5, g=9 is 322.
    # Try to find a graph with N=330 (hexagon size)
    target_n = 330
    
    best_result = None
    best_n = float('inf')
    
    # 1. Try Random Search for small N
    # We try N from Moore Bound up to target_n
    # But random search for girth 9 is hard.
    # We will just try N=330 for a while.
    
    def generate_random_regular_graph(n, k):
        # Configuration model
        while True:
            stubs = []
            for i in range(n):
                stubs.extend([i] * k)
            random.shuffle(stubs)
            
            adj = [[] for _ in range(n)]
            valid = True
            for i in range(0, n * k, 2):
                u, v = stubs[i], stubs[i+1]
                if u == v:
                    valid = False
                    break
                if v in adj[u]: # Check for multi-edge
                    valid = False
                    break
                adj[u].append(v)
                adj[v].append(u)
            
            if valid:
                return adj
        
    def check_girth(adj, n, g):
        # Check if girth >= g
        # For each vertex, BFS up to depth g-1
        # If we find a cycle of length < g, return False
        for start in range(n):
            # BFS
            visited = {start: 0}
            parent = {start: -1}
            queue = collections.deque([(start, 0)])
            
            while queue:
                u, depth = queue.popleft()
                if depth >= g - 1:
                    continue
                
                for v in adj[u]:
                    if v == parent[u]:
                        continue
                    if v in visited:
                        # Found a cycle
                        # Cycle length = depth + 1 + visited[v] ?
                        # Actually, the path from start to v in tree has length visited[v].
                        # The new path is depth + 1.
                        # The cycle closes.
                        # Length = depth + 1 + visited[v]?
                        # No, the cycle is formed by the tree path from v to LCA and the new edge.
                        # Simpler: If we encounter a visited node v that is not the parent,
                        # we have a cycle. The length is at most depth + 1 + visited[v].
                        # If this min possible length < g, fail?
                        # Actually, if we just want to know if girth < g,
                        # we can stop BFS at depth g-1.
                        # If we reach a node that was visited at depth d, and current depth is d',
                        # and d + d' + 1 < g, then we have a cycle < g.
                        if depth + 1 + visited[v] < g:
                            return False
                    else:
                        visited[v] = depth + 1
                        parent[v] = u
                        queue.append((v, depth + 1))
                        
        return True

    # Try random search
    attempts = 0
    while time.time() < deadline and attempts < 500:
        attempts += 1
        adj = generate_random_regular_graph(target_n, k)
        if check_girth(adj, target_n, g):
            # Convert adj to edges
            edges = []
            for i in range(target_n):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            return target_n, edges
        if time.time() > deadline:
            break
            
    # 2. Fallback to LUW Construction
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
