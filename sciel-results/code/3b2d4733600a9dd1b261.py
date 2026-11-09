"""Construct 5-regular graph with girth 9 using random regular graph generation."""
import random
import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin
    
    # Moore bound for odd girth
    moore = 1
    for i in range(1, (g - 1) // 2 + 1):
        moore += k * (k - 1) ** i
    # For g=9, k=5: 1 + 5 + 20 + 80 + 320 = 426
    
    # Try increasing n until we find a valid graph
    # Start near the known smallest (1116) and go down
    # But we need to be fast, so start with a reasonable n
    
    # For k=5, g=9, try n values
    # Random regular graphs with n ~ 1000+ can have girth >= 9
    
    def has_short_cycle(n, adj, max_len):
        """Check if any cycle of length < max_len exists using BFS from each vertex."""
        for start in range(n):
            # BFS up to depth max_len - 1, tracking parent to avoid backtracking
            visited = {}  # (node, depth) -> parent
            queue = deque([(start, 0, -1)])
            while queue:
                node, depth, parent = queue.popleft()
                if depth >= max_len - 1:
                    continue
                for nb in adj[node]:
                    if nb == parent:
                        continue
                    if nb == start and depth + 1 >= 3:
                        return True
                    key = (nb, depth + 1)
                    if key in visited:
                        continue
                    visited[key] = node
                    queue.append((nb, depth + 1, node))
        return False
    
    def generate_random_regular(n, degree):
        """Generate a random degree-regular graph using configuration model."""
        attempts = 0
        max_attempts = 50
        while attempts < max_attempts:
            attempts += 1
            # Create stubs
            stubs = []
            for v in range(n):
                for _ in range(degree):
                    stubs.append(v)
            random.shuffle(stubs)
            # Pair up stubs
            adj = [[] for _ in range(n)]
            valid = True
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i + 1]
                if u == v:
                    valid = False
                    break
                if v in adj[u]:
                    valid = False
                    break
                adj[u].append(v)
                adj[v].append(u)
            if valid:
                return adj
        return None
    
    # Try different n values
    # For k=5, g=9, the graph needs to be large enough
    # Start with n around 1000 and increase if needed
    base_n = 1000
    
    for offset in range(0, 500, 50):
        n = base_n + offset
        if time.time() > deadline - 10:
            break
        
        # Ensure n is compatible with k-regular (n*k must be even)
        if (n * k) % 2 != 0:
            continue
        
        for attempt in range(20):
            if time.time() > deadline - 10:
                break
            
            adj = generate_random_regular(n, k)
            if adj is None:
                continue
            
            # Verify girth >= g
            if not has_short_cycle(n, adj, g):
                # Build edge list
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n, edges
    
    # Fallback: use a known construction or return a simple valid graph
    # Use the LUW construction as fallback
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
