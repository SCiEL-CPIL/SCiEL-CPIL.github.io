import time
import sys
from collections import deque
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before the 150s kill

    # --- Helper: Check girth using BFS from every vertex (optimized) ---
    def check_girth_fast(adj, n, g):
        half_g = g // 2
        max_depth_allowed = (g - 1) // 2
        
        for start in range(n):
            # BFS
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            
            while queue:
                u = queue.popleft()
                if dist[u] >= half_g:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        # Found a visited node. 
                        # Check if it's a cross edge within the radius.
                        if dist[u] <= max_depth_allowed and dist[v] <= max_depth_allowed:
                            # Check if it's a tree edge
                            if parent[u] == v or parent[v] == u:
                                continue
                            # It's a cross edge within the radius.
                            return False
        return True

    # --- Strategy: Voltage Graph Lifting with Deterministic Patterns ---
    
    def get_petersen_edges():
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
            (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
            (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
        ]
        return edges

    def build_lift(base_edges, m, voltages):
        n_base = 10
        n = n_base * m
        adj = [[] for _ in range(n)]
        
        for idx, (u, v) in enumerate(base_edges):
            vol = voltages[idx] % m
            for i in range(m):
                u_lift = u * m + i
                v_lift = v * m + (i + vol) % m
                adj[u_lift].append(v_lift)
                adj[v_lift].append(u_lift)
        
        return adj, n

    # Fallback: LUW construction (Baseline)
    def construct_luw(k, g):
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

    # Get baseline
    baseline_n, baseline_edges = construct_luw(k, g)
    best_n = baseline_n
    best_edges = baseline_edges

    if k == 3:
        base_edges = get_petersen_edges()
        n_base_edges = len(base_edges)
        
        # Search for smaller n via voltage lifting
        # n = 10 * m
        # We want n < best_n
        max_m = best_n // 10
        if max_m < 10:
            max_m = 10
            
        search_deadline = deadline - 5.0
        
        # We iterate m from small to large.
        # For each m, we try deterministic voltage patterns.
        # Pattern 1: Arithmetic Progression v[i] = (a*i + b) % m
        # Pattern 2: Random (fallback)
        
        for m in range(10, max_m + 1):
            if time.time() > search_deadline:
                break
                
            found = False
            # Try deterministic patterns first
            # We only need to check a few 'a' and 'b' values.
            # If 'a' is coprime to m, it's a permutation.
            
            # Limit attempts per m to keep time down
            # For small m, try more patterns.
            max_patterns = 20 if m < 50 else 5
            
            for pat in range(max_patterns):
                if time.time() > search_deadline:
                    break
                    
                if pat == 0:
                    # All 1s
                    voltages = [1] * n_base_edges
                elif pat == 1:
                    # All 2s
                    voltages = [2] * n_base_edges
                elif pat == 2:
                    # Arithmetic: i+1
                    voltages = [(i + 1) % m for i in range(n_base_edges)]
                elif pat == 3:
                    # Arithmetic: 2*i+1
                    voltages = [(2 * i + 1) % m for i in range(n_base_edges)]
                elif pat == 4:
                    # Arithmetic: 3*i+1
                    voltages = [(3 * i + 1) % m for i in range(n_base_edges)]
                else:
                    # Random
                    voltages = [random.randint(1, m - 1) for _ in range(n_base_edges)]
                
                adj, n = build_lift(base_edges, m, voltages)
                
                if check_girth_fast(adj, n, g):
                    if n < best_n:
                        edges = []
                        for u in range(n):
                            for v in adj[u]:
                                if u < v:
                                    edges.append((u, v))
                        best_n = n
                        best_edges = edges
                        found = True
                        break # Break pattern loop, this m works
                        
            if found:
                # Since m is increasing, and we found a valid graph at this m,
                # and we started from small m, this is the smallest m we found.
                # However, we might have missed a valid m < current m if we didn't try enough patterns.
                # But given the deterministic patterns cover the most "structured" cases,
                # it is likely that if a small m works, one of these patterns will find it.
                # We break to save time, as larger m gives larger n.
                break

    return best_n, best_edges
