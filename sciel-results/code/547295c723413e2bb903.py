import time
import math
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s kill

    if k != 4 or g != 10:
        return _luw_construct(k, g)

    # Moore bound for (4,10): 2 + 4*(1+3+9) = 2+4*13 = 54? No.
    # For even girth g=10: n >= 2 * sum_{i=0}^{g/2-1} (k-1)^i = 2*(1+3+9+27+81) = 2*121 = 242? 
    # Actually Moore bound for girth g (even): n >= 2 * (1 + (k-1) + (k-1)^2 + ... + (k-1)^{g/2-1})
    # = 2 * ((k-1)^{g/2} - 1) / (k-2)
    # For k=4, g=10: 2 * (3^5 - 1)/(4-2) = 2 * (243-1)/2 = 242.
    # But the task says lower bound is 243. Let's use 243 as stated.
    
    # Get LUW baseline first (known to be valid)
    luw_n, luw_edges = _luw_construct(k, g)
    best_n = luw_n
    best_edges = luw_edges

    # Search generalized Petersen graphs G(n, s, t)
    # Vertices: outer 0..n-1, inner n..2n-1
    # Outer: (i, i+1) % n
    # Inner: (n+i, n+(i+s)%n)
    # Spokes: (i, n+(i+t)%n)
    # This gives 4-regular graph with 2n vertices.
    # We need 2n >= 243, so n >= 122.
    # We need 2n < best_n to improve.
    
    max_n_outer = min(250, (best_n - 1) // 2)
    min_n_outer = 122  # since 2*122 = 244 >= 243

    # Search over n_outer from min to max
    for n in range(min_n_outer, max_n_outer + 1):
        if time.time() > deadline:
            break
        
        # s and t are step parameters, 1 <= s,t < n/2
        # s != t, s+t != n, 2s != 0, 2t != 0 mod n
        # For efficiency, iterate over s and t
        for s in range(1, n // 2 + 1):
            if (2 * s) % n == 0:
                continue
            for t in range(1, n // 2 + 1):
                if (2 * t) % n == 0:
                    continue
                if s == t:
                    continue
                if s + t == n:
                    continue
                if math.gcd(math.gcd(n, s), t) != 1:
                    continue
                
                # Fast girth check
                if _check_petersen_girth(n, s, t, g):
                    # Build edges
                    edges = []
                    for i in range(n):
                        # Outer cycle
                        edges.append((i, (i + 1) % n))
                        # Inner cycle
                        edges.append((n + i, n + (i + s) % n))
                        # Spokes
                        edges.append((i, n + (i + t) % n))
                    
                    # Verify exact girth
                    if _verify_girth(2 * n, edges, g):
                        return 2 * n, edges

    return best_n, best_edges

def _check_petersen_girth(n, s, t, g):
    """
    Fast filter for girth of generalized Petersen graph G(n, s, t).
    BFS from vertex 0. If any edge (u,v) has dist[u]+dist[v]+1 < g, return False.
    This is a necessary condition: if girth < g, there exists a short cycle,
    and on that cycle there's an edge with dist[u]+dist[v]+1 < g.
    """
    total = 2 * n
    dist = [-1] * total
    dist[0] = 0
    q = [0]
    qi = 0
    
    # Precompute neighbors for speed
    def get_neighbors(v):
        if v < n:
            # Outer vertex
            return [(v - 1) % n, (v + 1) % n, n + (v + t) % n, n + (v - t) % n]
        else:
            # Inner vertex
            i = v - n
            return [n + (i - s) % n, n + (i + s) % n, (i - t) % n, (i + t) % n]
    
    max_depth = (g - 2) // 2
    while qi < len(q):
        u = q[qi]
        qi += 1
        du = dist[u]
        if du >= max_depth:
            continue
        
        for v in get_neighbors(u):
            if dist[v] == -1:
                dist[v] = du + 1
                if du + 1 <= max_depth:
                    q.append(v)
            else:
                if du + dist[v] + 1 < g:
                    return False
    return True

def _verify_girth(n, edges, g):
    """
    Exact girth check using BFS from every vertex.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        q = [s]
        qi = 0
        while qi < len(q):
            u = q[qi]
            qi += 1
            du = dist[u]
            if du >= (g - 1) // 2:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    if du + 1 <= (g - 1) // 2:
                        q.append(v)
                else:
                    if du + dist[v] + 1 < g:
                        return False
    return True

def _luw_construct(k, g):
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
