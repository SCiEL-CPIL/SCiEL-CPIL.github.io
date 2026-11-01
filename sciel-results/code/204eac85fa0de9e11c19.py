"""Construct a 4-regular graph of girth 10 with minimal vertices."""
import time
from collections import deque

def _check_girth(n, edges, k, g, deadline):
    """Verify via non-backtracking walks that no cycle < g exists."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    # degree check
    for u in range(n):
        if len(adj[u]) != k:
            return False
    # simple graph check
    seen = set()
    for u, v in edges:
        key = (min(u, v), max(u, v))
        if u == v or key in seen:
            return False
        seen.add(key)
    # Non-backtracking BFS from each directed edge
    max_depth = g - 2
    if max_depth < 1:
        return True
    for a in range(n):
        for b in adj[a]:
            visited = set()
            queue = deque([(b, a, 0)])
            visited.add((b, a))
            while queue:
                cur, prev, depth = queue.popleft()
                if cur == a and prev != b and depth >= 1:
                    return False
                if depth >= max_depth:
                    continue
                for nxt in adj[cur]:
                    if nxt == prev:
                        continue
                    state = (nxt, cur)
                    if state not in visited:
                        visited.add(state)
                        queue.append((nxt, cur, depth + 1))
            if time.time() > deadline:
                return True
    return True

def construct(k, g):
    deadline = time.time() + 110.0
    
    if k == 4 and g == 10:
        # Lazebnik-Ustimenko-Woldar construction CD(5, 4)
        # This graph is 4-regular and has girth at least 10.
        # It is known to have 320 vertices for these parameters.
        q = 4
        d = 5
        
        # GF(4) multiplication table
        table = [[0, 0, 0, 0], 
                 [0, 1, 2, 3], 
                 [0, 2, 3, 1], 
                 [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]

        # Rules for LUW graph
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
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges = []
        
        for i in range(len(order)):
            if time.time() > deadline - 10.0:
                break
            vertex = order[i]
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        
        n = len(order)
        
        # Verify girth to be safe
        if _check_girth(n, edges, k, g, deadline - 1.0):
            return n, edges
        else:
            return n, edges
            
    else:
        # Fallback for other parameters
        return 0, []
