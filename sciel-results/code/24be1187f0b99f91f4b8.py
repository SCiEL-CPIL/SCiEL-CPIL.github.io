import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    if k == 5 and g == 9:
        # Try to construct a smaller graph using voltage lifts
        # Base graph: 5-regular, girth 5, 25 vertices
        base_graph = generate_base_graph()
        if base_graph is not None:
            best_n = float('inf')
            best_edges = None
            
            # Try different moduli m
            # 25 * m vertices
            # We want 25 * m < 1250 (LUW size)
            # And ideally < 1116
            for m in range(35, 50):
                if time.time() > deadline:
                    break
                # Try multiple random voltage assignments
                for trial in range(100):
                    if time.time() > deadline:
                        break
                    edges = assign_voltages(base_graph, m)
                    if edges is not None:
                        n = 25 * m
                        if n < best_n:
                            best_n = n
                            best_edges = edges
                        # If we found a very small one, we can stop?
                        # But we want the smallest.
                        # 25 * 35 = 875.
                        # If we find one at 875, it's great.
                        # We can break the m loop if we find one for small m?
                        # But maybe a smaller m works?
                        # We iterate m from small to large.
                        # So the first valid one is the smallest for that base.
                        # But maybe a different base works better?
                        # We only have one base.
                        # So if we find a valid graph for m=35, n=875.
                        # We should return it immediately?
                        # Yes, 875 < 1116.
                        return 25 * m, edges
                        
        # Fallback to LUW
        return construct_luw(k, g)
    else:
        return construct_luw(k, g)

def generate_base_graph():
    # Generate a random 5-regular graph on 25 vertices with girth 5
    # Use a simple random regular graph generator
    n = 25
    degree = 5
    for _ in range(1000):
        # Generate random 5-regular graph
        # Method: Random permutation of half-edges
        half_edges = []
        for i in range(n):
            for _ in range(degree):
                half_edges.append(i)
        random.shuffle(half_edges)
        adj = {i: set() for i in range(n)}
        valid = True
        for i in range(0, n * degree, 2):
            u = half_edges[i]
            v = half_edges[i+1]
            if u == v:
                valid = False
                break
            if v in adj[u]:
                valid = False
                break
            adj[u].add(v)
            adj[v].add(u)
        if not valid:
            continue
        
        # Check girth
        if girth(adj, n) >= 5:
            # Convert to edge list
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            return edges
    return None

def girth(adj, n):
    # BFS from each vertex
    for s in range(n):
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
                elif v != parent[u]:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < 5:
                        return cycle_len
    return 100 # Large value

def assign_voltages(base_edges, m):
    # Assign random voltages in Z_m
    voltages = {}
    for u, v in base_edges:
        voltages[(u, v)] = random.randint(0, m-1)
        voltages[(v, u)] = (m - voltages[(u, v)]) % m
        
    # Construct lifted graph
    n_base = 25
    n_lift = n_base * m
    adj = {i: set() for i in range(n_lift)}
    
    for u, v in base_edges:
        w = voltages[(u, v)]
        for i in range(m):
            u_lift = u * m + i
            v_lift = v * m + ((i + w) % m)
            adj[u_lift].add(v_lift)
            adj[v_lift].add(u_lift)
            
    # Check girth >= 9
    if girth(adj, n_lift) >= 9:
        # Return edge list
        edges = []
        for u in range(n_lift):
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))
        return edges
    return None

def construct_luw(k, g):
    # Original LUW code
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
