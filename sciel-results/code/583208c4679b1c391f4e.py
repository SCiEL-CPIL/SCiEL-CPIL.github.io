import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin
    
    # Moore bound for (k, g)
    # For girth g (odd), Moore bound: 1 + k * sum_{i=0}^{(g-3)/2} (k-1)^i
    # For g=9, k=5: 1 + 5*(1 + 4 + 16 + 64) = 1 + 5*85 = 426
    if g % 2 == 1:
        r = (g - 3) // 2
        moore = 1 + k * sum((k-1)**i for i in range(r+1))
    else:
        r = g // 2 - 1
        moore = k * sum((k-1)**i for i in range(r+1))
    
    moore = max(moore, k + 1)
    
    def check_girth(adj, n, min_girth):
        """Check if graph has girth >= min_girth using BFS from each vertex."""
        for s in range(n):
            # BFS tracking (vertex, parent) to avoid backtracking
            visited = {}  # (vertex, parent) -> distance
            queue = deque()
            for nb in adj[s]:
                visited[(nb, s)] = 1
                queue.append((nb, s, 1))
            while queue:
                v, p, dist = queue.popleft()
                if dist >= min_girth:
                    continue
                for nb in adj[v]:
                    if nb == p:
                        continue
                    key = (nb, v)
                    if key in visited:
                        if visited[key] + 1 < min_girth:
                            return False  # found cycle shorter than min_girth
                    else:
                        visited[key] = dist + 1
                        queue.append((nb, v, dist + 1))
        return True
    
    def build_circulant(m, S):
        """Build circulant graph on m vertices with connection set S."""
        adj = [[] for _ in range(m)]
        edges = []
        for i in range(m):
            for s in S:
                j = (i + s) % m
                if j > i:
                    edges.append((i, j))
                adj[i].append(j)
        return adj, edges
    
    def girth_of_circulant(m, S):
        """Compute exact girth of circulant graph."""
        adj = [[] for _ in range(m)]
        for i in range(m):
            for s in S:
                adj[i].append((i + s) % m)
        
        for s in range(m):
            visited = {}
            queue = deque()
            for nb in adj[s]:
                visited[(nb, s)] = 1
                queue.append((nb, s, 1))
            while queue:
                v, p, dist = queue.popleft()
                if dist >= 9:
                    continue
                for nb in adj[v]:
                    if nb == p:
                        continue
                    key = (nb, v)
                    if key in visited:
                        return visited[key] + 1
                    else:
                        visited[key] = dist + 1
                        queue.append((nb, v, dist + 1))
        return 999  # no cycle found (shouldn't happen for connected graph)
    
    # Try circulant graphs
    best_n = None
    best_edges = None
    
    for m in range(moore, moore + 2000):
        if time.time() > deadline:
            break
        
        # Try several random connection sets
        max_attempts = 50 if m < 500 else 20
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Generate random symmetric 5-element set
            # Pick 2 or 3 positive values, add their negatives
            # Case 1: 2 positive + 3 negative? No, symmetric means if s in S, -s in S
            # For |S|=5 (odd), one element must be 0 or m/2 (if m even)
            # Actually, for a simple graph, 0 not in S. If m is even, m/2 is its own negative.
            
            if m % 2 == 0:
                # One element can be m//2 (self-inverse)
                # Pick 2 positive values from 1..m//2-1, plus m//2
                candidates = list(range(1, m//2))
                if len(candidates) < 2:
                    continue
                picks = random.sample(candidates, 2)
                S = picks + [m - picks[0], m - picks[1], m//2]
            else:
                # All elements come in pairs, so |S| must be even for symmetric set without 0
                # |S|=5 is odd, so we can't have a purely symmetric set without 0 or m/2
                # This means circulant graphs with odd m and degree 5 need 0 in S (self-loop) 
                # or m even with m/2 in S
                # Skip odd m for now
                continue
            
            # Check no duplicates and 0 not in S
            S_set = set(S)
            if len(S_set) != 5 or 0 in S_set:
                continue
            
            # Quick girth check
            girth = girth_of_circulant(m, S)
            if girth >= 9:
                adj, edges = build_circulant(m, S)
                # Verify
                if check_girth(adj, m, 9):
                    best_n = m
                    best_edges = edges
                    break
        
        if best_n is not None:
            break
    
    if best_n is not None:
        return best_n, best_edges
    
    # Fallback: use LUW construction
    q = k
    d = 5  # gives girth >= 9 for the bipartite version
    # Use the original LUW code as fallback
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
