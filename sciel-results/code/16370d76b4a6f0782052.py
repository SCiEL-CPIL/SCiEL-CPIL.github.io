import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0
    
    # --- Step 1: LUW Baseline ---
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
    
    baseline_n = len(order)
    baseline_edges = edges
    
    # --- Step 2: Search ---
    if k == 3 and g == 16:
        best_n = baseline_n
        best_edges = baseline_edges
        
        def check_girth_fast(edges, n, target_g):
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            min_girth = float('inf')
            for start in range(n):
                dist = [-1] * n
                parent = [-1] * n
                dist[start] = 0
                queue = deque([start])
                while queue:
                    u = queue.popleft()
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            queue.append(v)
                        elif v != parent[u]:
                            l = dist[u] + dist[v] + 1
                            if l < min_girth:
                                min_girth = l
                            if l < target_g:
                                return False
            return True

        def greedy_construct(n, target_g):
            # Try to build a graph with n vertices, degree 3, girth >= target_g
            edges = []
            adj = [[] for _ in range(n)]
            edge_set = set()
            
            # List of all possible edges
            all_edges = []
            for u in range(n):
                for v in range(u+1, n):
                    all_edges.append((u, v))
            random.shuffle(all_edges)
            
            for u, v in all_edges:
                if len(adj[u]) >= 3 or len(adj[v]) >= 3:
                    continue
                if (u, v) in edge_set or (v, u) in edge_set:
                    continue
                if u == v:
                    continue
                
                # Check if adding (u, v) creates a cycle < target_g
                # Temporarily add
                adj[u].append(v)
                adj[v].append(u)
                edge_set.add((u, v))
                
                # Check girth
                # We can do a BFS from u and v to see if they connect in < target_g - 1 steps
                # Actually, just run full check if graph is small?
                # For n=1000, full check is O(N^2). Too slow for every edge.
                # We need a fast check.
                # Check if there is a path between u and v of length < target_g - 1 in the graph WITHOUT (u,v).
                # If yes, cycle length = path_len + 1 < target_g.
                
                # Remove for check
                adj[u].remove(v)
                adj[v].remove(u)
                edge_set.remove((u, v))
                
                # BFS from u to v
                visited = {u}
                queue = deque([(u, 0)])
                found_short = False
                while queue:
                    node, dist = queue.popleft()
                    if dist >= target_g - 1:
                        break
                    for nei in adj[node]:
                        if nei == v:
                            if dist + 1 < target_g - 1: # Cycle length = dist + 1 + 1
                                found_short = True
                                break
                        if nei not in visited:
                            visited.add(nei)
                            queue.append((nei, dist + 1))
                    if found_short:
                        break
                
                if not found_short:
                    # Safe to add
                    adj[u].append(v)
                    adj[v].append(u)
                    edge_set.add((u, v))
                    edges.append((u, v))
                
            # Check if all degrees are 3
            for i in range(n):
                if len(adj[i]) != 3:
                    return None
            return edges

        # Search loop
        current_n = 512
        while time.time() < deadline - 10.0:
            # Try greedy construction
            edges = greedy_construct(current_n, g)
            if edges:
                # Verify
                if check_girth_fast(edges, current_n, g):
                    best_n = current_n
                    best_edges = edges
                    break
            
            current_n += 16
            if current_n > baseline_n:
                break

        if best_n < baseline_n:
            return best_n, best_edges

    return baseline_n, baseline_edges
