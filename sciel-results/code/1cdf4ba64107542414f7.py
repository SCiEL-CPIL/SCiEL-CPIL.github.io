import time
import random
import sys
from collections import deque

def construct(k, g):
    deadline = time.time() + 115

    if k != 3:
        # Fallback: LUW construction for general k
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)

        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q

        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours_luw(vertex):
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
            for other in neighbours_luw(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))

        return len(order), edges

    # k = 3 case: use non-backtracking walk enumeration
    # For girth g (even), r = g/2
    # Moore bound: n >= 2 * (1 + (k-1) + (k-1)^2 + ... + (k-1)^(r-1)) = 2 * ((k-1)^r - 1)/(k-2)
    # For k=3, g=16: r=8, n >= 2 * (2^8 - 1) = 510
    # n must be even for 3-regular, so n >= 512

    r = g // 2
    moore_bound = 2 * ((k - 1) ** r - 1) // (k - 2)
    if moore_bound % 2 != 0:
        moore_bound += 2

    # Strategy: Build a random 3-regular graph on n vertices and check girth
    # Use efficient girth checking via BFS from each vertex

    def check_girth_valid(n, adj, target_girth):
        """Check if graph has girth >= target_girth using BFS from each vertex"""
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                if dist[u] >= target_girth - 1:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        if dist[v] < target_girth - 1:
                            queue.append(v)
                    else:
                        # Cycle found through start
                        cycle_len = dist[u] + 1 + dist[v]
                        if cycle_len < target_girth:
                            return False
        return True

    def random_3_regular(n, rng):
        """Generate a random 3-regular graph on n vertices (n must be even)"""
        if n < 4 or n % 2 != 0:
            return None
        # Use configuration model with rejection
        for _ in range(100):
            # Create stubs
            stubs = []
            for i in range(n):
                stubs.extend([i, i, i])
            rng.shuffle(stubs)
            adj = {i: set() for i in range(n)}
            valid = True
            for i in range(0, n * 3, 2):
                u, v = stubs[i], stubs[i + 1]
                if u == v or v in adj[u]:
                    valid = False
                    break
                adj[u].add(v)
                adj[v].add(u)
            if valid and all(len(adj[i]) == 3 for i in range(n)):
                return adj
        return None

    rng = random.Random(12345)
    
    # Try n values starting from Moore bound
    # For k=3, g=16: Moore bound = 510, so try 512, 514, ...
    # Random 3-regular graphs have girth ~ 2*log_{k-1}(n) = 2*log_2(n)
    # For n=512: expected girth ~ 2*9 = 18 > 16, so it's plausible
    # For n=1024: expected girth ~ 2*10 = 20
    
    # But we need to be careful: the actual girth distribution matters
    # Let's try a range of n values with multiple random attempts
    
    best_result = None
    
    # First, try small n values with more attempts
    for n_try in range(moore_bound, moore_bound + 400, 2):
        if time.time() > deadline - 10:
            break
        max_attempts = 200 if n_try < moore_bound + 100 else 50
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            adj = random_3_regular(n_try, rng)
            if adj is None:
                continue
            if check_girth_valid(n_try, adj, g):
                edges = []
                for u in range(n_try):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n_try, edges
    
    # If random search fails, try a more structured approach
    # Build a (k-1)-ary tree of depth r and try to close it
    
    # For now, use LUW as fallback (gives ~1280 vertices for k=3, g=16)
    q = 3
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r_idx, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r_idx]

    def neighbours_luw(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = (x[c] + y[a] * x[b]) % q
                else:
                    y[c] = (x[c] - x[a] * y[b]) % q
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours_luw(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges
