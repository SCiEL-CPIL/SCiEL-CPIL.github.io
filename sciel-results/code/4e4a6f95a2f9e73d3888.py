import time
import sys
from collections import deque
import random

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Uses Lazebnik-Ustimenko-Woldar (LUW) algebraic construction.
    """
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin

    best_n = float('inf')
    best_edges = None

    # Determine candidate d values for LUW
    # Girth of LUW CD(d, q) is at least 2d + 2.
    # We need 2d + 2 >= g => d >= (g - 2) / 2.
    min_d = max(2, (g - 2) // 2)
    
    # Try d from min_d to min_d + 10
    # Usually min_d is the most compact.
    for d in range(min_d, min_d + 10):
        if time.time() > deadline - 5:
            break
        try:
            n, edges = _luw_construct(k, d)
            # Verify girth
            girth = _verify_girth_fast(n, edges, g)
            if girth >= g:
                if n < best_n:
                    best_n = n
                    best_edges = edges
                    # If we found a graph with n close to Moore bound, stop?
                    # Moore bound for k=3, g=16 is 510.
                    # If n == 510, we can't do better.
                    if n <= 512:
                        break
        except Exception as e:
            # Ignore errors and try next d
            pass

    if best_edges is not None:
        return best_n, best_edges

    # Fallback: If LUW failed (shouldn't happen), try a larger d or a random search
    # For robustness, just return a large LUW graph
    try:
        n, edges = _luw_construct(k, min_d + 20)
        return n, edges
    except Exception:
        return 0, []

def _luw_construct(k, d):
    """
    Lazebnik-Ustimenko-Woldar construction CD(d, q) with q=k.
    Returns (n, edges).
    """
    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    # Build the rule table for CD(d, q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None] + [first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield (1 - side, tuple(y))

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

def _verify_girth_fast(n, edges, min_girth_needed):
    """
    Fast girth verification using BFS from each vertex.
    Returns the actual girth or a lower bound.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    half_g = (min_girth_needed - 1) // 2
    min_girth = min_girth_needed + 1
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            if dist[u] >= half_g:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    if dist[v] < half_g:
                        queue.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                        if min_girth < min_girth_needed:
                            return min_girth
    
    return min_girth

if __name__ == "__main__":
    n, edges = construct(3, 16)
    print(f"Constructed graph with {n} vertices, {len(edges)} edges.")
    girth = _verify_girth_fast(n, edges, 16)
    print(f"Measured girth: {girth}")
