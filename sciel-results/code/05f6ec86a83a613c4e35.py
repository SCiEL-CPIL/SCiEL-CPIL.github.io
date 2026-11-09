import time
import random
import numpy as np
from itertools import combinations

def construct(k, g):
    start_time = time.time()
    deadline = 110.0

    # Fallback: LUW Graph
    def get_luw(k, g):
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

    def check_circulant_girth(n, steps, g):
        # BFS from 0
        # dist[v] = min distance
        # prev[v] = parent
        # We want to know if there is a cycle of length < g.
        # In a circulant graph, a cycle starting and ending at 0 corresponds to a sequence of steps summing to 0 mod n.
        # We do BFS on the Cayley graph.
        dist = [-1] * n
        dist[0] = 0
        queue = [0]
        
        for u in queue:
            if dist[u] >= g - 1:
                continue
            for s in steps:
                v1 = (u + s) % n
                v2 = (u - s) % n
                for v in (v1, v2):
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
                    else:
                        # Found a non-tree edge
                        # The cycle length is dist[u] + 1 + dist[v]? No.
                        # The edge (u,v) connects two nodes in the BFS tree.
                        # The cycle length is dist[u] + 1 + dist[v] only if the LCA is 0?
                        # No, the cycle length is dist[u] + dist[v] + 1 - 2 * dist[lca(u,v)].
                        # We don't track LCA easily here.
                        # However, if we just want to know if girth < g, we can check if any edge connects nodes with dist sum + 1 < g?
                        # This is a necessary condition for a short cycle but not sufficient?
                        # Actually, if dist[u] + 1 + dist[v] < g, it doesn't guarantee a cycle of that length.
                        # But if we find an edge (u,v) such that dist[u] + dist[v] + 1 < g, it's a strong candidate.
                        # To be exact, we should reconstruct the path or use a different method.
                        
                        # Simpler exact check for small g:
                        # Check all non-backtracking walks of length < g from 0.
                        pass
                        
        # Let's use the non-backtracking walk check for exactness, it's fast enough for n<320.
        # State: (node, prev)
        # Start: (s, 0) and (-s, 0) for each step s.
        # If we reach (0, s') at depth L, we have a cycle of length L+1.
        
        visited = set()
        queue = []
        for s in steps:
            v1 = s % n
            v2 = (-s) % n
            if v1 != 0:
                queue.append((v1, 0, 1))
                visited.add((v1, 0))
            if v2 != 0 and v2 != v1:
                queue.append((v2, 0, 1))
                visited.add((v2, 0))
        
        while queue:
            u, prev, depth = queue.pop(0)
            if depth >= g - 1:
                continue
            for s in steps:
                v1 = (u + s) % n
                v2 = (u - s) % n
                for v in (v1, v2):
                    if v == prev:
                        continue
                    if v == 0:
                        if depth + 1 < g:
                            return False
                    elif (v, u) not in visited:
                        visited.add((v, u))
                        queue.append((v, u, depth + 1))
        return True

    if k == 4 and g == 10:
        # Search for circulant graphs
        # n must be even for 4-regular? No, sum of degrees 4n is even.
        # Moore bound 161.
        for n in range(162, 320):
            if time.time() > deadline:
                break
            
            # Steps must be in 1..n//2
            # We need 4 steps.
            # To avoid multi-edges, steps must be distinct and not negatives of each other.
            # In 1..n//2, they are automatically distinct and not negatives (unless n/2 is in steps and n is even, -n/2 = n/2).
            # So if n is even, we cannot use step n/2 twice? No, we just pick 4 distinct values from 1..n//2.
            # If n is odd, 1..(n-1)/2.
            
            max_step = n // 2
            if n % 2 == 0:
                # If we pick n/2, it's its own inverse.
                # We can pick at most one n/2.
                # Let's just iterate combinations.
                pass
            
            # Optimization: Randomly sample steps first, then exhaustive if time permits?
            # Or just iterate.
            # Number of combinations is large.
            # Let's try a heuristic: small steps are bad.
            # Try steps from larger end?
            
            # Actually, let's just run a limited random search over steps for each n.
            # If we find one, return it.
            
            for _ in range(1000):
                if time.time() > deadline:
                    break
                steps = random.sample(range(1, max_step + 1), 4)
                if check_circulant_girth(n, steps, g):
                    edges = []
                    for u in range(n):
                        for s in steps:
                            v = (u + s) % n
                            if u < v:
                                edges.append((u, v))
                    return n, edges
                    
    # Fallback
    return get_luw(k, g)
