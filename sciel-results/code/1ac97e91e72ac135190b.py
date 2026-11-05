import time
import random

def construct(k, g):
    # Petersen graph
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    
    # Precompute cycles of length < g
    # g is 16
    def get_cycles(edges, max_len):
        adj = {i: [] for i in range(10)}
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        cycles = set()
        
        def dfs(start, current, path, visited):
            if len(path) > max_len:
                return
            if len(path) >= 3 and current == start:
                cyc = path[:-1]
                min_idx = cyc.index(min(cyc))
                norm = tuple(cyc[min_idx:] + cyc[:min_idx])
                rev = norm[::-1]
                if norm < rev:
                    cycles.add(norm)
                else:
                    cycles.add(rev)
                return
            
            for neighbor in adj[current]:
                if neighbor == start and len(path) < 3:
                    continue
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    dfs(start, neighbor, path, visited)
                    path.pop()
                    visited.remove(neighbor)
        
        for i in range(10):
            dfs(i, i, [i], {i})
            
        return list(cycles)

    cycles = get_cycles(petersen_edges, g - 1)
    
    # Map edges to indices for voltage assignment
    edge_to_idx = {}
    for i, (u, v) in enumerate(petersen_edges):
        edge_to_idx[(u, v)] = i
        edge_to_idx[(v, u)] = i
        
    # Precompute edge indices for each cycle
    cycle_edges = []
    for cyc in cycles:
        edge_idxs = []
        for i in range(len(cyc) - 1):
            u, v = cyc[i], cyc[i+1]
            edge_idxs.append(edge_to_idx[(u, v)])
        cycle_edges.append(edge_idxs)
        
    start_time = time.time()
    deadline = start_time + 110
    
    # Try different M
    for M in range(52, 150):
        if time.time() > deadline:
            break
            
        # Search for voltages
        found = False
        attempts = 0
        # Increase attempts as M increases? Or keep constant?
        # Let's try 5000 attempts per M.
        while attempts < 5000 and time.time() < deadline:
            attempts += 1
            voltages = [random.randint(0, M - 1) for _ in range(len(petersen_edges))]
            
            valid = True
            for edge_idxs in cycle_edges:
                s = 0
                for idx in edge_idxs:
                    s += voltages[idx]
                s %= M
                
                # Check if lifted length is >= g
                # Lifted length = L * (M / gcd(s, M))
                # We need L * (M / gcd(s, M)) >= g
                # L is len(edge_idxs)
                L = len(edge_idxs)
                if s == 0:
                    # Lifted length is L
                    if L < g:
                        valid = False
                        break
                else:
                    import math
                    g_val = math.gcd(s, M)
                    lifted_len = L * (M // g_val)
                    if lifted_len < g:
                        valid = False
                        break
            
            if valid:
                # Construct graph
                n = 10 * M
                edges = []
                for u, v in petersen_edges:
                    idx = edge_to_idx[(u, v)]
                    val = voltages[idx]
                    for i in range(M):
                        u_id = u * M + i
                        v_id = v * M + (i + val) % M
                        edges.append((u_id, v_id))
                return n, edges
        
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
