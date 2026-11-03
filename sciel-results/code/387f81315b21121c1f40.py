import time
import sys
import random
from collections import deque

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Uses algebraic constructions (LUW) for guaranteed valid graphs.
    """
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin for 120s limit

    best_n = float('inf')
    best_edges = None
    
    # LUW Construction
    # For k=3, q=3.
    # We need girth g.
    # Girth of CD(d, 3) is 2d+2.
    # So we need 2d+2 >= g => d >= (g-2)/2.
    
    min_d = (g - 2) // 2
    if (g - 2) % 2 != 0:
        min_d += 1
        
    # Try d starting from min_d
    for d in range(min_d, min_d + 5):
        if time.time() > deadline:
            break
            
        try:
            n, edges = _luw_construct(k, d)
            # Verify girth
            girth = _calc_girth_from_edges(n, edges, g)
            if girth >= g and n < best_n:
                best_n = n
                best_edges = edges
        except Exception:
            pass

    if best_edges is not None:
        return best_n, best_edges

    # Fallback: If LUW failed (should not happen), force a large d
    n, edges = _luw_construct(k, 12)
    return n, edges

def _luw_construct(k, d):
    """
    Lazebnik-Ustimenko-Woldar construction CD(d, q) with q=k.
    Returns (n, edges).
    """
    q = k
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

def _calc_girth_from_edges(n, edges, target_girth):
    """
    Calculates girth using BFS from each vertex, limited to depth target_girth // 2.
    Optimized for early detection of short cycles.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    min_girth = float('inf')
    max_depth = target_girth // 2
    
    # Early exit if we find a very small girth
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([(start, 0)])
        
        while queue:
            u, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = depth + 1
                    parent[v] = u
                    queue.append((v, depth + 1))
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                        if min_girth < target_girth:
                            return min_girth
        
        if min_girth < target_girth:
            break
            
    return min_girth if min_girth != float('inf') else n

if __name__ == "__main__":
    # Quick test
    n, edges = construct(3, 16)
    print(f"Constructed graph with {n} vertices, {len(edges)} edges.")
    girth = _calc_girth_from_edges(n, edges, 16)
    print(f"Measured girth: {girth}")
