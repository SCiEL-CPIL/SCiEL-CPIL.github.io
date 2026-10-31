"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def construct(k, g):
    deadline = time.time() + 115
    
    # For k=3, g=16: try random 3-regular graphs and verify girth
    # Use a fast girth check via BFS from each vertex up to depth g//2 - 1
    # If no cycle found within that depth from any vertex, girth >= g
    
    def check_girth(n, edges, min_girth):
        """Check if graph has girth >= min_girth using non-backtracking BFS."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        max_depth = min_girth // 2 - 1  # for even girth, check up to this depth
        
        for start in range(n):
            # BFS with parent tracking (non-backtracking)
            # state: (vertex, parent)
            visited = set()
            queue = deque([(start, -1, 0)])
            visited.add((start, -1))
            while queue:
                node, parent, depth = queue.popleft()
                if depth == max_depth:
                    continue
                for nb in adj[node]:
                    if nb == parent:
                        continue
                    if (nb, node) in visited:
                        return False
                    visited.add((nb, node))
                    queue.append((nb, node, depth + 1))
        return True
    
    def random_3reg(n):
        """Generate random 3-regular graph on n vertices."""
        if n % 2 != 0 or n < 4:
            return None
        # Use configuration model
        stubs = []
        for i in range(n):
            stubs.extend([i] * 3)
        random.shuffle(stubs)
        edges = []
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u != v:
                edges.append((min(u,v), max(u,v)))
            else:
                return None  # loop, retry
        # Check for multi-edges
        if len(set(edges)) != len(edges):
            return None
        # Check degrees
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        if any(d != 3 for d in deg):
            return None
        return edges
    
    # Try increasing n from Moore bound
    # Moore bound for k=3, g=16: 2 * sum_{i=0}^{g/2-1} (k-1)^i = 2 * sum_{i=0}^{7} 2^i = 2*(2^8-1) = 510
    moore_bound = 2 * ((1 << (g//2)) - 1)
    
    # Try random 3-regular graphs starting from a reasonable size
    # The known best is 936, so try around there
    best_n = None
    best_edges = None
    
    # First try the LUW construction as fallback
    q = k
    d = g - 5 if (g - 5) % 2 else g - 4
    if d < 3:
        d = 3
    
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
    index, order, luw_edges = {start: 0}, [start], []
    for vertex in order:
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                luw_edges.append((index[vertex], index[other]))
    
    best_n = len(order)
    best_edges = luw_edges
    
    # Now try to find a smaller graph via random search
    # Try sizes from 512 upward
    for n in range(512, min(best_n, 1000), 2):
        if time.time() > deadline:
            break
        for attempt in range(50):
            if time.time() > deadline:
                break
            edges = random_3reg(n)
            if edges is None:
                continue
            if check_girth(n, edges, g):
                return n, edges
    
    return best_n, best_edges
# EVOLVE-BLOCK-END
