import time
import random
import sys
import math

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Fallback: Lazebnik-Ustimenko-Woldar graph
    def luw_graph():
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

    # Helper: Check girth using non-backtracking walks from every vertex
    def check_girth_nonbacktracking(n, edges, g):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Use non-backtracking walk: track (vertex, previous) pairs
        # A cycle of length L < g means there's a non-backtracking walk of length L
        # that returns to its start
        # BFS on the directed line graph (state = (current, previous))
        
        for start in range(n):
            # dist[(u, v)] = length of non-backtracking walk from (start, -1) to (u, v)
            # We use a dict for sparse states
            dist = {}
            queue = []
            # Start: from vertex 'start', we can go to any neighbor with no previous
            for nb in adj[start]:
                state = (nb, start)
                dist[state] = 1
                queue.append(state)
            
            head = 0
            while head < len(queue):
                u, prev = queue[head]
                head += 1
                d_u = dist[(u, prev)]
                if d_u >= g:
                    continue
                for v in adj[u]:
                    if v == prev:
                        continue  # non-backtracking
                    new_state = (v, u)
                    if new_state not in dist:
                        dist[new_state] = d_u + 1
                        queue.append(new_state)
                    else:
                        # We've seen this state before, meaning a cycle
                        # The cycle length is d_u + 1
                        cycle_len = d_u + 1
                        if cycle_len < g:
                            return False
        
        return True

    # Main Logic
    if k == 5 and g == 9:
        # LUW graph for (5,9): q=5, d = g-5=4 (since 4 is even, use g-4=5)
        # Wait: g-5 = 4, 4%2==0, so d = g-4 = 5
        # LUW(5, 5) gives a 5-regular graph with girth at least 9
        # Let's just use the LUW construction directly
        result = luw_graph()
        n, edges = result
        return n, edges
    
    # General case: try LUW
    result = luw_graph()
    return result
