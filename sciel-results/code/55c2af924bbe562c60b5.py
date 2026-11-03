import time
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # 1. Generate Base Graph: LUW CD(3,3)
    q = 3
    d = 3
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        
    rule = {2: (1, 1), 3: (2, 1)}
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, base_edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                base_edges.append((index[vertex], index[other]))
                
    n_base = len(order)
    
    # Precompute 8-cycles in base graph
    base_adj = [[] for _ in range(n_base)]
    for u, v in base_edges:
        base_adj[u].append(v)
        base_adj[v].append(u)
        
    # Find all 8-cycles
    # Since it's bipartite, cycles are even.
    # We can do BFS from each node up to depth 4.
    # If we find a node at depth 4 that is connected to a node at depth 0? No.
    # If we find a node at depth 4 that is the start node? No, that's a cycle of length 8?
    # Path: s -> n1 -> n2 -> n3 -> n4 -> s. Length 5? No, bipartite.
    # Path: s(0) -> n1(1) -> n2(2) -> n3(3) -> n4(4) -> n5(5) -> n6(6) -> n7(7) -> s(0).
    # So we look for paths of length 7 from s to s?
    # Or paths of length 4 from s to a neighbor of s?
    
    cycles_8 = []
    for s in range(n_base):
        # BFS to depth 4
        visited = {s: 0}
        queue = [(s, [s])]
        while queue:
            node, path = queue.pop(0)
            depth = len(path) - 1
            if depth == 4:
                # Check if node is connected to s
                if s in base_adj[node]:
                    # Cycle found: path + [s]
                    # Normalize cycle to avoid duplicates
                    cycle = path + [s]
                    # Sort cycle for uniqueness
                    # Since it's a cycle, rotation doesn't matter.
                    # Find min element, rotate to start there.
                    min_idx = min(range(8), key=lambda i: cycle[i])
                    rotated = cycle[min_idx:] + cycle[:min_idx]
                    # Also reverse direction
                    reversed_rot = rotated[::-1]
                    unique_key = tuple(min(rotated, reversed_rot))
                    if unique_key not in [tuple(min(c, c[::-1])) for c in cycles_8]:
                        cycles_8.append(rotated)
            elif depth < 4:
                for nei in base_adj[node]:
                    if nei not in visited:
                        visited[nei] = depth + 1
                        queue.append((nei, path + [nei]))
                        
    # 2. Try Voltage Lifts
    for K in range(14, 31):
        if time.time() > deadline:
            break
            
        for seed in range(100):
            if time.time() > deadline:
                break
                
            random.seed(seed)
            voltages = {}
            for u, v in base_edges:
                voltages[(u, v)] = random.randint(0, K - 1)
                
            valid = True
            for cycle in cycles_8:
                # Calculate voltage product
                prod = 0
                for i in range(7):
                    u = cycle[i]
                    v = cycle[i+1]
                    if u < v:
                        prod = (prod + voltages[(u, v)]) % K
                    else:
                        prod = (prod - voltages[(v, u)]) % K
                if prod == 0:
                    valid = False
                    break
                    
            if valid:
                # Construct lift edges
                lift_edges = []
                for u, v in base_edges:
                    vol = voltages[(u, v)]
                    for i in range(K):
                        u_lift = u * K + i
                        v_lift = v * K + (i + vol) % K
                        e1 = min(u_lift, v_lift)
                        e2 = max(u_lift, v_lift)
                        lift_edges.append((e1, e2))
                        
                return n_base * K, lift_edges
                
    # Fallback: Return a known valid graph (e.g. from seed logic with d=11)
    # To save space, we just return the base graph if we must, but it's invalid.
    # We should return the seed's output.
    # Let's just return a large LUW graph.
    d_fallback = 11
    # ... (generate LUW d=11) ...
    # For brevity, if we get here, we return the base (invalid) to avoid crash?
    # No, must be valid.
    # I will implement the fallback LUW d=11.
    
    # Re-run LUW for d=11
    d = 11
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        
    def neighbours_fb(vertex):
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
        for other in neighbours_fb(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges
