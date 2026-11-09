import random
import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    if k == 3 and g == 16:
        # Strategy: Random Lift of K3,3
        # K3,3 has 6 vertices. Lift size N. Total vertices 6N.
        # Target n ~ 936 => N = 156.
        
        # Try a few N values around 156
        for N in [156, 157, 158, 159, 160, 161, 162, 163, 164, 165]:
            if time.time() > deadline - 10:
                break
            
            n = 6 * N
            # Try up to 20 random lifts for this N
            for attempt in range(20):
                if time.time() > deadline - 5:
                    break
                
                edges = _generate_random_lift_k33(N)
                if edges is None:
                    continue
                
                if _check_girth_bfs(edges, n, g):
                    return n, edges
    
    # Fallback: LUW
    return _luw_construct(k, g)

def _generate_random_lift_k33(N):
    """
    Generate a random lift of K3,3 with group Z_N.
    Returns list of edges or None.
    """
    # Base graph K3,3 vertices 0-2 (left), 3-5 (right)
    # Edges: (0,3), (0,4), (0,5), (1,3), (1,4), (1,5), (2,3), (2,4), (2,5)
    base_edges = [
        (0, 3), (0, 4), (0, 5),
        (1, 3), (1, 4), (1, 5),
        (2, 3), (2, 4), (2, 5)
    ]
    
    # Assign random voltages
    voltages = {}
    for u, v in base_edges:
        voltages[(u, v)] = random.randint(0, N - 1)
        voltages[(v, u)] = (-voltages[(u, v)]) % N
    
    # Build lift
    edges = []
    # Vertices are (base_vertex, group_element)
    # We only need to add edges from left to right to avoid duplicates
    for u in range(3): # Left side
        for v in range(3, 6): # Right side
            if (u, v) in voltages:
                v_uv = voltages[(u, v)]
                for g in range(N):
                    u_lift = u * N + g
                    v_lift = v * N + (g + v_uv) % N
                    edges.append((u_lift, v_lift))
                    
    return edges

def _check_girth_bfs(edges, n, g):
    """Fast girth check using BFS - returns True if girth >= g."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    half_g = g // 2
    
    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        queue = deque([src])
        
        while queue:
            u = queue.popleft()
            d = dist[u]
            if d >= half_g:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = d + 1
                    queue.append(v)
                elif dist[u] + dist[v] + 1 < g:
                    # Found a cycle shorter than g
                    return False
    return True

def _luw_construct(k, g):
    """LUW construction - produces a valid graph but potentially large."""
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
