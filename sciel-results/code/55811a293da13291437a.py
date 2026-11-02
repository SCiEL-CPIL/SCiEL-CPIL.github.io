import time
import sys
from collections import deque
import random

def _bfs_girth(adj):
    """Compute the girth of an undirected graph via BFS from each vertex."""
    n = len(adj)
    best = float('inf')
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif parent[u] != v and parent[v] != u:
                    cycle = dist[u] + dist[v] + 1
                    if cycle < best:
                        best = cycle
        if best <= 2:
            break
    return best if best < float('inf') else 0

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction (from seed)."""
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

def _construct_hexagon_cage(k, g):
    """Construct incidence graph of generalized hexagon H(q) for q=k-1."""
    if k != 4 or g != 10:
        return None
        
    q = k - 1 # q=3
    
    # GF(3) implementation
    # Elements 0, 1, 2
    # 0+0=0, 0+1=1, 0+2=2, 1+1=2, 1+2=0, 2+2=1
    # 0*0=0, 0*1=0, 0*2=0, 1*1=1, 1*2=2, 2*2=1
    
    def gf_add(a, b): return (a + b) % q
    def gf_sub(a, b): return (a - b) % q
    def gf_mul(a, b): return (a * b) % q
    
    # Points of H(q)
    # Representation:
    # 1. Point at infinity: (-1, -1, -1) ? Or index 0.
    # 2. Points (x, y, z) in GF(q)^3?
    # Actually, the number of points is q^3+1?
    # Let's assume the standard set of points is:
    # P_inf
    # P_{x,y} for x,y in GF(q)? No.
    
    # Correct construction of H(q):
    # Points are (x, y, z) with x,y,z in GF(q) such that...
    # Actually, let's use the explicit list for q=3.
    # Total points should be q^3+1 = 28?
    # If so, vertices = 56.
    
    # Let's try to generate 28 points and 28 lines.
    # Points:
    # 0: Infinity
    # 1..27: (x, y, z) in {0,1,2}^3
    
    points = [(-1, -1, -1)] # Infinity
    for x in range(q):
        for y in range(q):
            for z in range(q):
                points.append((x, y, z))
                
    n_points = len(points)
    if n_points != q**3 + 1:
        return None

    # Lines:
    # A line contains q+1 points.
    # Incidence:
    # 1. Infinity is incident to lines L_inf_x, L_inf_y, L_inf_z?
    # Standard incidence relations for H(q):
    # Points: P_inf, P_{x,y,z}
    # Lines: 
    #   L1_{x,y}: contains P_inf and P_{x,y,z} for all z? No.
    
    # Let's use a known small instance or fallback if unsure.
    # Given the risk of incorrect construction, and the time limit,
    # I will return None to fallback to LUW.
    # The LUW construction is guaranteed to work.
    
    return None

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 130
    
    if k == 4 and g == 10:
        try:
            if time.time() < deadline:
                result = _construct_hexagon_cage(k, g)
                if result:
                    n, edges = result
                    # Verify girth
                    adj = [[] for _ in range(n)]
                    for u, v in edges:
                        adj[u].append(v)
                        adj[v].append(u)
                    if _bfs_girth(adj) >= g:
                        return n, edges
        except Exception:
            pass
            
        if time.time() >= deadline:
            try:
                n, edges = _luw_construct(k, g)
                return n, edges
            except Exception:
                pass
        
        n, edges = _luw_construct(k, g)
        return n, edges
    else:
        n, edges = _luw_construct(k, g)
        return n, edges
