import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115
    
    def luw_construction():
        # Fallback construction, known to be valid for k=4, g=10
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

    def verify_girth(n, edges, g):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        for i in range(n):
            if len(adj[i]) != k:
                return False
        
        edge_set = set()
        for u, v in edges:
            if u == v:
                return False
            key = (min(u, v), max(u, v))
            if key in edge_set:
                return False
            edge_set.add(key)
            
        # Check girth
        for L in range(3, g):
            for start in range(n):
                visited = set()
                stack = [(start, -1, 0)]
                while stack:
                    v, prev, l = stack.pop()
                    if l == L:
                        if v == start:
                            return False
                        continue
                    for w in adj[v]:
                        if w != prev:
                            if (v, prev, l) not in visited:
                                visited.add((v, prev, l))
                                stack.append((w, v, l + 1))
        return True

    def get_luw_result():
        n, edges = luw_construction()
        return n, edges

    # Specific optimization for k=4, g=10
    if k == 4 and g == 10:
        # Try to find a 320-vertex graph (5 * 64)
        # Base graph: K5
        # Voltages in Z_64
        
        base_vertices = 5
        M = 64
        n_target = base_vertices * M
        
        # Edges in K5
        base_edges = []
        for i in range(base_vertices):
            for j in range(i + 1, base_vertices):
                base_edges.append((i, j))
        
        # Initialize random voltages
        # voltages[u][v] = sigma
        voltages = [[0] * base_vertices for _ in range(base_vertices)]
        for u, v in base_edges:
            voltages[u][v] = random.randint(0, M - 1)
            voltages[v][u] = (-voltages[u][v]) % M
            
        def build_graph(vol):
            adj = [[] for _ in range(n_target)]
            for u in range(base_vertices):
                for v in range(base_vertices):
                    if u == v:
                        continue
                    sigma = vol[u][v]
                    for h in range(M):
                        u_idx = u * M + h
                        v_h = (h + sigma) % M
                        v_idx = v * M + v_h
                        adj[u_idx].append(v_idx)
            return adj

        def compute_girth(adj, n, g_max=10):
            # Returns the length of the shortest cycle, or infinity if none < g_max
            # We only care if it's < 10
            min_girth = g_max
            for s in range(n):
                dist = [-1] * n
                parent = [-1] * n
                dist[s] = 0
                q = deque([s])
                while q:
                    u = q.popleft()
                    if dist[u] >= g_max // 2: # Pruning: if dist is large, cycle will be large
                        # Actually, if dist[u] + dist[v] + 1 >= g_max, we don't care
                        # But we need to explore to find if there's a shorter one?
                        # If we are looking for cycles < 10, we only need to explore up to depth 4 or 5?
                        # A cycle of length L has radius L/2.
                        # So if L < 10, radius <= 4.
                        # So we only need to BFS up to depth 4.
                        pass 
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            q.append(v)
                        elif parent[v] != u and parent[u] != v:
                            # Found a cycle
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < min_girth:
                                min_girth = cycle_len
            return min_girth

        best_voltages = [row[:] for row in voltages]
        best_girth = compute_girth(build_graph(voltages), n_target, 10)
        
        while time.time() < deadline - 5:
            # Pick a random edge and change voltage
            u, v = random.choice(base_edges)
            old_val = voltages[u][v]
            # Random step
            delta = random.choice([-1, 1, random.randint(1, 10)])
            new_val = (old_val + delta) % M
            
            voltages[u][v] = new_val
            voltages[v][u] = (-new_val) % M
            
            adj = build_graph(voltages)
            new_girth = compute_girth(adj, n_target, 10)
            
            if new_girth > best_girth:
                best_girth = new_girth
                best_voltages = [row[:] for row in voltages]
                if best_girth >= 10:
                    break
            elif new_girth == best_girth:
                # Accept with some probability to explore
                if random.random() > 0.5:
                    best_girth = new_girth
                    best_voltages = [row[:] for row in voltages]
                else:
                    # Revert
                    voltages[u][v] = old_val
                    voltages[v][u] = (-old_val) % M
            else:
                # Revert
                voltages[u][v] = old_val
                voltages[v][u] = (-old_val) % M
        
        if best_girth >= 10:
            # Construct edges from best_voltages
            adj = build_graph(best_voltages)
            edges = []
            seen = set()
            for u in range(n_target):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
                        seen.add((u, v))
            return n_target, edges
            
    # Fallback
    return get_luw_result()
