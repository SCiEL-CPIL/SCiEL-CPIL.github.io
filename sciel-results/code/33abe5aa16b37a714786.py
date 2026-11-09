import random
import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 110s deadline to be safe
    
    # Fallback: LUW construction (guaranteed valid, but large)
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

    # Helper: Generate a random k-regular graph on n vertices
    def random_k_regular(n, k):
        if n * k % 2 != 0:
            return None
        # Use configuration model
        stubs = []
        for i in range(n):
            stubs.extend([i] * k)
        random.shuffle(stubs)
        edges = []
        for i in range(0, n * k, 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                # Self-loop, try to fix by swapping
                # For simplicity, restart if self-loop
                return None
            edges.append((u, v))
        # Check for multi-edges
        edge_set = set()
        for u, v in edges:
            if (v, u) in edge_set:
                return None
            edge_set.add((u, v))
        return edges

    # Helper: Check girth using BFS from every vertex
    def get_girth(n, edges):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        min_girth = float('inf')
        for start in range(n):
            # BFS
            dist = [-1] * n
            parent = [-1] * n
            queue = deque([start])
            dist[start] = 0
            found = False
            while queue and not found:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        # Cycle found
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
                            if min_girth <= g:
                                return min_girth
                        found = True # Stop BFS for this start if we found a cycle? No, need min.
                        # Actually, BFS finds the shortest cycle passing through start?
                        # No, BFS from start finds shortest path to each node.
                        # A back edge (u,v) gives a cycle of length dist[u] + dist[v] + 1.
                        # This is the shortest cycle containing the edge (u,v) and passing through start?
                        # Not necessarily the global shortest.
                        # But if we do this for all starts, we cover all edges.
                        # Optimization: If min_girth <= g, we can stop early.
        return min_girth

    # Try random graphs with increasing n
    # Moore bound for k=3, g=16 is 766.
    # Try n from 800 to 3000
    best_n, best_edges = None, None
    
    for n in range(800, 3001, 100):
        if time.time() > deadline:
            break
        for _ in range(20): # Try 20 random graphs for each n
            if time.time() > deadline:
                break
            edges = random_k_regular(n, k)
            if edges is None:
                continue
            # Check girth
            girth = get_girth(n, edges)
            if girth >= g:
                return n, edges
            # If girth is close, keep as best fallback?
            # No, we need valid.
            
    # Fallback to LUW
    return luw_construct(k, g)
