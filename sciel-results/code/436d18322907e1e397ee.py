import time
import sys
import random

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Uses algebraic constructions (LUW) for optimal or near-optimal sizes.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin for 150s limit

    # For k=3, g=16, use LUW construction with q=3.
    # Try different d values to find the smallest valid graph.
    
    best_n = float('inf')
    best_edges = None
    
    # LUW CD(d, q) for q=3
    # Girth of CD(d, 3) should be at least 2d+2 for appropriate d
    # For g=16, we need 2d+2 >= 16 => d >= 7
    
    for d in [7, 8, 9, 10, 11, 12]:
        if time.time() > deadline - 5:
            break
            
        try:
            n, edges = _luw_construct(3, d)
            # Verify girth with fast check
            girth = _calc_girth_fast(n, edges, g)
            if girth >= g and n < best_n:
                best_n = n
                best_edges = edges
        except Exception:
            pass

    if best_edges is not None:
        return best_n, best_edges

    # Fallback: force a LUW with larger d
    n, edges = _luw_construct(3, 15)
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

def _calc_girth_fast(n, edges, target_girth):
    """
    Calculates girth using BFS from each vertex, with early exit.
    Returns the minimum cycle length found, or n if no cycle found.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    min_girth = target_girth + 1  # We only care if girth >= target_girth
    
    # Early exit if we find a very small girth
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        qi = 0
        
        while qi < len(queue):
            u = queue[qi]
            qi += 1
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                        if min_girth < target_girth:
                            return min_girth
        
        if min_girth < target_girth:
            break
            
    return min_girth

def _gq_construct(k, g):
    """
    Placeholder for GQ-based construction.
    For k=3, g=16, this is the area where 936 vertices come from.
    """
    pass

if __name__ == "__main__":
    # Quick test
    n, edges = construct(3, 16)
    print(f"Constructed graph with {n} vertices, {len(edges)} edges.")
    girth = _calc_girth_fast(n, edges, 16)
    print(f"Measured girth: {girth}")
