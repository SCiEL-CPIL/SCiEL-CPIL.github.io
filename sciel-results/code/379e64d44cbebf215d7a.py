# EVOLVE-BLOCK-START
import time
import random
import math

def construct(k, g):
    """
    Construct a k-regular graph of girth at least g.
    Uses parameter sweep of Lazebnik-Ustimenko-Woldar construction.
    """
    start_time = time.time()
    deadline = start_time + 110
    
    if k != 3:
        # Fallback to original logic for non-3 regular
        return _original_construct(k, g)

    # Moore bound
    moore_bound = 2 * ((2 ** (g // 2)) - 1)
    
    # Try different 'd' parameters for LUW construction with q=3
    # Smaller d means smaller graph, but lower girth.
    # We want the smallest d such that girth >= g.
    best_n = float('inf')
    best_edges = None
    
    # d ranges from 6 to 15
    for d in range(6, 16):
        if time.time() > deadline - 10:
            break
            
        try:
            n, edges = _lnw_construct_params(3, d)
        except Exception:
            continue
            
        if n >= best_n:
            continue
            
        # Check girth
        adj = _edges_to_adj(edges, n)
        girth = _calculate_girth(adj, n)
        
        if girth >= g:
            best_n = n
            best_edges = edges
            # We can break if we find a very small one, but let's check smaller d first
            # Since we iterate d from low to high, the first valid one is the smallest n?
            # Not necessarily, n grows with d, so yes, smallest d gives smallest n.
            # So we can return immediately?
            # Yes, because n(d) is increasing.
            return n, edges

    if best_edges is not None:
        return best_n, best_edges
        
    # Fallback to original large d
    return _lnw_construct(3, g)

def _original_construct(k, g):
    # Copy of original construct logic if needed, or just call _lnw_construct
    return _lnw_construct(k, g)

def _lnw_construct_params(k, d):
    """
    LUW construction with fixed q=k and variable d.
    """
    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    
    # Rule generation depends on d
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

def _edges_to_adj(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _calculate_girth(adj, n):
    min_cycle = n + 1
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            if dist[u] >= min_cycle:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    if v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
    return min_cycle if min_cycle <= n else n + 1

def _lnw_construct(k, g):
    # Original fallback
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    return _lnw_construct_params(k, d)
# EVOLVE-BLOCK-END
