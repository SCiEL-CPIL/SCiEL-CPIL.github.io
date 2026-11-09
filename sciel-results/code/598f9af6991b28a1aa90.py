import time
import sys

# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Constructs a k-regular graph of girth at least g.
    Uses Lazebnik-Ustimenko-Woldar (LUW) graphs.
    For k=5, g=9, LUW(d=4, q=5) yields a 5-regular bipartite graph with girth 10.
    n = 2 * 5^4 = 1250.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # LUW graphs are q-regular.
    # We need k-regular. So q must be k.
    # LUW requires q to be prime or 4.
    if k not in [2, 3, 4, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
        # Fallback for unsupported k
        return _fallback_construct(k, g)
        
    q = k
    
    # Define field operations
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    def build_luw(d):
        """
        Builds the LUW graph CD(d, q).
        Returns (n, edges)
        """
        # Rule generation
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
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges = []
        
        for vertex in order:
            if time.time() > deadline:
                return None
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        
        return len(order), edges

    def calculate_girth(n, edges):
        """
        Calculates the girth of a simple graph using BFS from each vertex.
        """
        if n == 0: return 0
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        min_girth = float('inf')
        for s in range(n):
            if time.time() > deadline:
                return min_girth
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            for u in queue:
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        if parent[u] != v and parent[v] != u:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < min_girth:
                                min_girth = cycle_len
                            if min_girth <= g:
                                return min_girth
        return min_girth

    best_n = None
    best_edges = None
    
    # Try d values.
    # For k=5, g=9:
    # d=3 -> n=250, girth=8 (invalid)
    # d=4 -> n=1250, girth=10 (valid)
    
    for d in range(3, 12):
        if time.time() > deadline:
            break
            
        res = build_luw(d)
        if res is None:
            continue
        n, edges = res
        
        # Check girth
        girth = calculate_girth(n, edges)
        
        if girth >= g:
            if best_n is None or n < best_n:
                best_n = n
                best_edges = edges
            # Since size grows with d, we can break here
            break 
            
    if best_n is not None:
        return best_n, best_edges
    
    # Fallback
    return _fallback_construct(k, g)

def _fallback_construct(k, g):
    """
    Fallback for unsupported k or if LUW fails.
    """
    # Simple cycle if k=2
    if k == 2:
        n = max(g, 3)
        edges = [(i, (i+1)%n) for i in range(n)]
        return n, edges
    
    # For other k, return a large LUW-like graph or error
    # Since the task is specifically for k=5, this should rarely be hit
    return 0, []

# EVOLVE-BLOCK-END
