import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    
    # LUW construction
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

    # Get LUW result as baseline
    luw_n, luw_edges = luw_construct(k, g)
    
    # If not the specific case, return LUW
    if k != 4 or g != 10:
        return luw_n, luw_edges
    
    # For k=4, g=10, try to find a smaller graph
    # LUW gives n=2048, smallest known is 320
    # Try random generation with girth checking
    
    def check_girth(adj, n, min_girth):
        """Check if graph has girth >= min_girth using BFS from each vertex."""
        max_depth = (min_girth - 1) // 2
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                v = queue.popleft()
                if dist[v] >= max_depth:
                    continue
                for u in adj[v]:
                    if dist[u] == -1:
                        dist[u] = dist[v] + 1
                        parent[u] = v
                        queue.append(u)
                    elif u != parent[v] and parent[u] != v:
                        cycle_len = dist[v] + dist[u] + 1
                        if cycle_len < min_girth:
                            return False
        return True

    def generate_random_k_regular(n, k, rng):
        """Generate a random k-regular graph using configuration model with rejection."""
        for attempt in range(100):
            adj = [set() for _ in range(n)]
            stubs = []
            for i in range(n):
                stubs.extend([i] * k)
            rng.shuffle(stubs)
            
            valid = True
            for i in range(0, len(stubs), 2):
                u = stubs[i]
                v = stubs[i + 1]
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
            
            # Verify all degrees are exactly k
            all_ok = True
            for i in range(n):
                if len(adj[i]) != k:
                    all_ok = False
                    break
            if all_ok:
                return adj
        
        return None

    def adj_to_edges(adj, n):
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))
        return edges

    # Try to find a graph with n < luw_n
    # Start from a reasonable n and go up
    # For k=4, g=10, try n from 320 upward
    best_n = luw_n
    best_edges = luw_edges
    
    rng = random.Random(42)
    
    # Try various n values, starting small
    # The theoretical lower bound is around 243
    # Smallest known is 320
    for target_n in range(320, min(luw_n, 400)):
        if time.time() > deadline - 30:
            break
        
        # Try a few random graphs for this n
        for trial in range(20):
            if time.time() > deadline - 30:
                break
            
            adj = generate_random_k_regular(target_n, 4, rng)
            if adj is None:
                continue
            
            if check_girth(adj, target_n, 10):
                best_n = target_n
                best_edges = adj_to_edges(adj, target_n)
                break  # Found one for this n, try smaller n
    
    if best_n < luw_n:
        return best_n, best_edges
    
    return luw_n, luw_edges
