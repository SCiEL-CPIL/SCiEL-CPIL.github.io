import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Moore bound for k=3, g=16 is 512.
    # We try to find a graph with n close to 512.
    
    # Helper: Check girth of a graph given as adjacency list
    def get_girth(adj):
        n = len(adj)
        for _ in range(100): # Max girth check? No, we need exact.
            pass
        # BFS from each vertex
        min_girth = float('inf')
        for i in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[i] = 0
            queue = [i]
            while queue:
                u = queue.pop(0)
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
            if min_girth < g:
                break
        return min_girth

    # Helper: Generate random 3-regular graph
    def random_3_regular(n):
        # Fisher-Yates shuffle of half-edges
        half_edges = [i for i in range(n) for _ in range(3)]
        random.shuffle(half_edges)
        adj = [[] for _ in range(n)]
        used = [False] * n
        # This simple pairing might create multi-edges or loops.
        # Better: Configuration model with rejection or repair.
        # For simplicity, use a known generator or just try random swaps.
        # Let's use a simple random graph and repair.
        pass

    # Voltage Graph Approach
    # Base: Petersen Graph
    base_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    base_adj = [[] for _ in range(10)]
    for u, v in base_edges:
        base_adj[u].append(v)
        base_adj[v].append(u)
        
    def build_lift(m, voltages):
        # voltages: list of 15 integers in 0..m-1
        n = 10 * m
        adj = [[] for _ in range(n)]
        for i, (u, v) in enumerate(base_edges):
            w = voltages[i] % m
            for g in range(m):
                u_lift = u * m + g
                v_lift = v * m + ((g + w) % m)
                adj[u_lift].append(v_lift)
                # The reverse edge in base has voltage -w?
                # In a voltage graph, if (u,v) has voltage w, (v,u) has voltage -w.
                # But we are building an undirected graph.
                # The edge (v, u) in base corresponds to (v_lift, u_lift) with voltage -w.
                # We just add the undirected edge.
                # adj[v_lift].append(u_lift) is already handled if we iterate all edges?
                # No, base_edges are unique undirected edges.
                # So we add the edge once.
        return adj

    def get_girth_lift(m, voltages):
        adj = build_lift(m, voltages)
        return get_girth(adj)

    # Search for voltage assignment
    best_n = None
    best_edges = None
    
    # Try different m values
    # Moore bound 512. 10*m ~ 512 => m ~ 51.
    for m in range(51, 60):
        if time.time() > deadline - 10:
            break
        # Random restarts
        for restart in range(100):
            if time.time() > deadline - 5:
                break
            voltages = [random.randint(0, m - 1) for _ in range(15)]
            girth = get_girth_lift(m, voltages)
            if girth >= g:
                adj = build_lift(m, voltages)
                edges = []
                for i in range(len(adj)):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return len(adj), edges
            
            # Local search on voltages
            # Try to improve girth
            for step in range(100):
                if time.time() > deadline - 5:
                    break
                # Pick a random voltage index
                idx = random.randint(0, 14)
                old_v = voltages[idx]
                new_v = random.randint(0, m - 1)
                voltages[idx] = new_v
                new_girth = get_girth_lift(m, voltages)
                if new_girth >= g:
                    adj = build_lift(m, voltages)
                    edges = []
                    for i in range(len(adj)):
                        for j in adj[i]:
                            if i < j:
                                edges.append((i, j))
                    return len(adj), edges
                elif new_girth > girth:
                    girth = new_girth
                else:
                    # Revert
                    voltages[idx] = old_v
    
    # Fallback: LUW Construction (from seed)
    # Re-implementing the seed's LUW logic to ensure valid output
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
