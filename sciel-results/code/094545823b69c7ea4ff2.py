import time
import random
import math
from collections import deque

def _luw_construct(k, g):
    # Fallback to LUW
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

def _check_girth(n, edges, target_girth):
    # Check if graph has girth >= target_girth
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # For each edge, check shortest path between u and v excluding the edge itself
    # This finds the shortest cycle containing that edge.
    # If all such cycles are >= target_girth, then girth >= target_girth.
    
    for u, v in edges:
        # BFS from u to v
        # visited[i] = distance from u to i
        visited = [-1] * n
        queue = deque()
        queue.append((u, 0))
        visited[u] = 0
        
        found_dist = -1
        while queue:
            node, dist = queue.popleft()
            if node == v:
                found_dist = dist
                break
            for neighbor in adj[node]:
                if neighbor == u and node == v: # Skip the edge (u,v) if coming from v to u? 
                    # Actually, we are doing BFS from u.
                    # We should not use the edge (u,v) to go from u to v directly.
                    pass
                if visited[neighbor] == -1:
                    # Do not traverse the edge (u,v) in the forward direction from u
                    if node == u and neighbor == v:
                        continue
                    visited[neighbor] = dist + 1
                    queue.append((neighbor, dist + 1))
        
        if found_dist != -1:
            cycle_len = found_dist + 1
            if cycle_len < target_girth:
                return False
    return True

def _try_voltage_graph(k, g, time_limit):
    if k != 3 or g != 16:
        return None
        
    start_time = time.time()
    m = 234
    # Base graph K4
    # Vertices 0,1,2,3
    # Edges: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    base_edges = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    
    # We assign voltages to these 6 edges
    # voltages[i] corresponds to base_edges[i]
    
    while time.time() - start_time < time_limit:
        # Random voltages
        voltages = [random.randint(0, m-1) for _ in range(6)]
        
        # Check connectivity: gcd of all voltages and m must be 1
        # Actually, the generated group must be the whole Z_m.
        # This requires gcd(v1, ..., v6, m) == 1
        g_val = m
        for v in voltages:
            g_val = math.gcd(g_val, v)
            if g_val == 1:
                break
        if g_val != 1:
            continue
            
        # Construct the lift
        n = 4 * m
        edges = []
        # Map (vertex_in_base, offset) to index
        # index = base_vertex * m + offset
        for i, (u, v) in enumerate(base_edges):
            vol = voltages[i]
            for offset in range(m):
                u_idx = u * m + offset
                v_idx = v * m + (offset + vol) % m
                edges.append((u_idx, v_idx))
                
        # Check girth
        if _check_girth(n, edges, g):
            return n, edges
            
    return None

def construct(k, g):
    # Try the voltage graph for the specific case (3, 16)
    # Allow 100 seconds for search
    result = _try_voltage_graph(k, g, 100)
    if result:
        return result
        
    # Fallback to LUW
    return _luw_construct(k, g)
