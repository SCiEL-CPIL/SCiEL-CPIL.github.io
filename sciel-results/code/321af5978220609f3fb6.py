import time
import random
import math
import numpy as np
from collections import deque

def construct(k, g):
    # Fallback: Lazebnik-Ustimenko-Woldar graph
    def luw_construct(k, g):
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

    # If k=4 and g=10, try to find a smaller cage
    if k == 4 and g == 10:
        deadline = time.time() + 110
        
        def generate_random_4_regular(n):
            if n % 2 != 0:
                return None
            # Configuration model
            half_edges = [i for i in range(n) for _ in range(4)]
            random.shuffle(half_edges)
            adj = [[] for _ in range(n)]
            for i in range(0, len(half_edges), 2):
                u, v = half_edges[i], half_edges[i+1]
                if u == v:
                    return None
                adj[u].append(v)
                adj[v].append(u)
            # Check for multi-edges
            for u in range(n):
                if len(adj[u]) != 4 or len(set(adj[u])) != 4:
                    return None
            return adj

        def check_girth_numpy(adj, n, max_girth=9):
            """
            Returns True if girth > max_girth.
            Uses matrix powers to detect short cycles.
            """
            A = np.zeros((n, n), dtype=int)
            for u in range(n):
                for v in adj[u]:
                    A[u, v] = 1
            # Check cycles of length 3 to max_girth
            for L in range(3, max_girth + 1):
                A = A @ A
                # If any diagonal element is non-zero, there is a cycle of length L
                if np.any(np.diag(A) > 0):
                    return False
            return True

        def find_short_cycle(adj, n):
            """
            Find a short cycle (length < 10) using BFS.
            Returns a list of nodes in the cycle, or None.
            """
            for start in range(n):
                dist = [-1] * n
                parent = [-1] * n
                dist[start] = 0
                q = deque([start])
                while q:
                    u = q.popleft()
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            q.append(v)
                        elif dist[v] < dist[u]:
                            # Found a cycle
                            path = []
                            cur = u
                            while cur != -1:
                                path.append(cur)
                                cur = parent[cur]
                            path.append(v)
                            # Verify it's a simple cycle and length < 10
                            if len(path) < 10:
                                return path
            return None

        def swap_edges(adj, a, b, c, d):
            """
            Swap edges (a,b) and (c,d) with (a,c) and (b,d).
            Returns True if swap was successful (no multi-edges/loops).
            """
            # Check validity
            if a == c or a == d or b == c or b == d:
                return False
            if c in adj[a] or d in adj[a] or a in adj[c] or a in adj[d]:
                return False
            
            # Perform swap
            adj[a].remove(b)
            adj[b].remove(a)
            adj[c].remove(d)
            adj[d].remove(c)
            
            adj[a].append(c)
            adj[c].append(a)
            adj[b].append(d)
            adj[d].append(b)
            return True

        def revert_swap(adj, a, b, c, d):
            adj[a].remove(c)
            adj[c].remove(a)
            adj[b].remove(d)
            adj[d].remove(b)
            
            adj[a].append(b)
            adj[b].append(a)
            adj[c].append(d)
            adj[d].append(c)

        # Search for N from 244 to 320
        for n_target in range(244, 321, 2):
            if time.time() > deadline - 5:
                break
            for attempt in range(500):
                if time.time() > deadline - 5:
                    break
                adj = generate_random_4_regular(n_target)
                if not adj:
                    continue
                
                # Quick check with NumPy
                if check_girth_numpy(adj, n_target, 9):
                    # Found a valid graph
                    edges = []
                    for u in range(n_target):
                        for v in adj[u]:
                            if u < v:
                                edges.append((u, v))
                    return n_target, edges
                
                # Try to repair
                for repair_step in range(20):
                    if time.time() > deadline - 5:
                        break
                    cycle = find_short_cycle(adj, n_target)
                    if not cycle:
                        # Re-verify with NumPy (might have missed it or fixed it)
                        if check_girth_numpy(adj, n_target, 9):
                            edges = []
                            for u in range(n_target):
                                for v in adj[u]:
                                    if u < v:
                                        edges.append((u, v))
                            return n_target, edges
                        break
                    
                    # Pick an edge in the cycle to swap
                    # Cycle: c0-c1-...-ck-c0
                    u, v = cycle[0], cycle[1]
                    # Find neighbors x of u (not v) and y of v (not u)
                    neighbors_u = [x for x in adj[u] if x != v]
                    neighbors_v = [y for y in adj[v] if y != u]
                    
                    if not neighbors_u or not neighbors_v:
                        continue
                    
                    x = random.choice(neighbors_u)
                    y = random.choice(neighbors_v)
                    
                    if swap_edges(adj, u, v, x, y):
                        # Check if girth improved or maintained
                        # We don't need to check full girth, just if we broke the specific cycle
                        # But to be safe, we can check if the cycle is gone
                        # For now, just accept and re-evaluate in next loop
                        pass
                    else:
                        continue

        # Fallback to LUW
        return luw_construct(k, g)
    else:
        return luw_construct(k, g)
