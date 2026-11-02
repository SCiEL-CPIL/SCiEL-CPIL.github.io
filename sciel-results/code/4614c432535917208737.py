import time
import random
from collections import deque

def luw_construct(k, g):
    """
    Construct a k-regular graph with girth >= g using LUW method.
    Returns (n, edges).
    """
    q = k
    # d is the parameter for LUW. Girth is at least d + 5.
    # We want girth >= g.
    # So d + 5 >= g => d >= g - 5.
    # Also d must be odd? The formula in the prompt uses:
    # d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    # This ensures d is odd if g-5 is odd, or even?
    # If g=16, g-5=11 (odd). d=11.
    # If g=8, g-5=3 (odd). d=3.
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

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

def build_cover(n_base, edges_base, voltages, M):
    """
    Build the M-cover of the base graph.
    vertices are (v, g) where v in 0..n_base-1, g in 0..M-1.
    edge (u, v) in base with voltage w lifts to (u, g) -- (v, (g+w)%M).
    """
    n_cover = n_base * M
    edges_cover = []
    for i, (u, v) in enumerate(edges_base):
        w = voltages[i]
        for g in range(M):
            u_lift = u * M + g
            v_lift = v * M + (g + w) % M
            edges_cover.append((u_lift, v_lift))
    return n_cover, edges_cover

def check_girth(n, edges, g):
    """
    Check if graph has girth >= g.
    Uses BFS from each vertex.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            if dist[u] >= g - 1:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
                elif dist[v] <= dist[u]:
                    # Cycle found of length dist[u] + dist[v] + 1
                    # But BFS finds shortest path.
                    # If we find a neighbor v that is already visited and dist[v] < dist[u],
                    # it's a back edge.
                    # Cycle length = dist[u] + dist[v] + 1?
                    # Actually, if v is visited and v != parent, we have a cycle.
                    # But we don't track parent.
                    # Simpler: if dist[v] < dist[u], cycle length = dist[u] + dist[v] + 1.
                    # We need to ensure this is >= g.
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

def construct(k, g):
    deadline = time.time() + 115
    rng = random.Random(42)
    
    # Strategy: Lifting for k=3, g=16
    if k == 3 and g == 16:
        # Base: LUW for g=8
        base_n, base_edges = luw_construct(3, 8)
        
        # Try lifts with increasing M
        # Start with small M to minimize N
        for M in range(10, 200):
            if time.time() > deadline:
                break
            for attempt in range(100):
                if time.time() > deadline:
                    break
                voltages = [rng.randint(0, M - 1) for _ in base_edges]
                cover_n, cover_edges = build_cover(base_n, base_edges, voltages, M)
                
                # Quick check: is it 3-regular?
                # Yes, by construction.
                
                # Check girth
                if check_girth(cover_n, cover_edges, g):
                    return cover_n, cover_edges

    # Fallback: LUW for target g
    n, edges = luw_construct(k, g)
    return n, edges
