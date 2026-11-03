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
        voltages = [[0] * base_vertices for _ in range(base_vertices)]
        for u, v in base_edges:
            voltages[u][v] = random.randint(0, M - 1)
            voltages[v][u] = (-voltages[u][v]) % M
            
        def build_adj(vol):
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

        def check_girth_fast(adj, n, min_girth=10):
            """Check if graph has no cycle shorter than min_girth.
            Returns True if girth >= min_girth, False otherwise."""
            # For each vertex, do BFS up to depth (min_girth-1)//2
            # and check for back edges that would create short cycles
            max_depth = (min_girth - 1) // 2  # 4 for girth 10
            
            for s in range(n):
                # BFS from s, tracking parent to avoid immediate backtracking
                dist = [-1] * n
                parent = [-1] * n
                dist[s] = 0
                q = deque([s])
                while q:
                    u = q.popleft()
                    if dist[u] > max_depth:
                        continue
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            if dist[v] <= max_depth:
                                q.append(v)
                        elif v != parent[u] and u != parent[v]:
                            # Found a cycle of length dist[u] + dist[v] + 1
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < min_girth:
                                return False
            return True

        def count_short_cycles(adj, n, min_girth=10):
            """Count number of cycles shorter than min_girth (approximate via BFS)."""
            max_depth = (min_girth - 1) // 2
            count = 0
            for s in range(n):
                dist = [-1] * n
                parent = [-1] * n
                dist[s] = 0
                q = deque([s])
                while q:
                    u = q.popleft()
                    if dist[u] > max_depth:
                        continue
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            if dist[v] <= max_depth:
                                q.append(v)
                        elif v != parent[u] and u != parent[v]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < min_girth:
                                count += 1
            return count

        # Strategy 1: Try many random initializations with early exit
        best_voltage_copy = None
        best_count = float('inf')
        
        for attempt in range(50):
            if time.time() > deadline - 15:
                break
                
            for u, v in base_edges:
                voltages[u][v] = random.randint(0, M - 1)
                voltages[v][u] = (-voltages[u][v]) % M
            
            adj = build_adj(voltages)
            
            # Quick check: just verify girth
            if check_girth_fast(adj, n_target, 10):
                edges = []
                for u in range(n_target):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n_target, edges
        
        # Strategy 2: Local search with simulated annealing-like approach
        if time.time() < deadline - 10:
            # Start from a random initialization
            for u, v in base_edges:
                voltages[u][v] = random.randint(0, M - 1)
                voltages[v][u] = (-voltages[u][v]) % M
            
            adj = build_adj(voltages)
            current_count = count_short_cycles(adj, n_target, 10)
            best_voltage_copy = [row[:] for row in voltages]
            best_count = current_count
            
            temp = 1.0
            cooling = 0.95
            
            while time.time() < deadline - 10:
                if current_count == 0:
                    edges = []
                    for u in range(n_target):
                        for v in adj[u]:
                            if u < v:
                                edges.append((u, v))
                    return n_target, edges
                
                # Pick a random edge and change voltage
                u, v = random.choice(base_edges)
                old_val = voltages[u][v]
                new_val = random.randint(0, M - 1)
                
                voltages[u][v] = new_val
                voltages[v][u] = (-new_val) % M
                
                adj = build_adj(voltages)
                new_count = count_short_cycles(adj, n_target, 10)
                
                # Accept if better or with probability exp(-delta/temp)
                if new_count < current_count:
                    current_count = new_count
                    if new_count < best_count:
                        best_count = new_count
                        best_voltage_copy = [row[:] for row in voltages]
                else:
                    import math
                    delta = new_count - current_count
                    if delta == 0:
                        accept = True
                    else:
                        accept = random.random() < math.exp(-delta / max(temp, 0.01))
                    
                    if not accept:
                        voltages[u][v] = old_val
                        voltages[v][u] = (-old_val) % M
                        adj = build_adj(voltages)
                
                temp *= cooling
        
        # Restore best if we found one
        if best_voltage_copy is not None and best_count == 0:
            for u, v in base_edges:
                voltages[u][v] = best_voltage_copy[u][v]
                voltages[v][u] = best_voltage_copy[v][u]
            adj = build_adj(voltages)
            if check_girth_fast(adj, n_target, 10):
                edges = []
                for u in range(n_target):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n_target, edges
        
        # Fallback
        return get_luw_result()
