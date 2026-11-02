import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # 110s limit, leaving 10s margin

    # 1. Helper: LUW Construction
    def luw_construct(k, g):
        q = k
        base_d = max(3, g - 5)
        
        for d in range(base_d, base_d + 10):
            if time.time() > deadline:
                break
                
            # Build LUW graph with parameter d
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
            
            n = len(order)
            
            # Quick Moore bound check
            if g % 2 == 1:
                r = (g - 1) // 2
                moore = 1 + k * sum(k**i for i in range(r))
            else:
                r = g // 2
                moore = 2 * sum(k**i for i in range(r))
            
            if n < moore:
                continue
            
            # Verify Girth
            if verify_girth(n, edges, g):
                return n, edges

        # Fallback: Return the last attempted graph (largest d)
        d_final = base_d + 9
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d_final + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d_final - 1)
                for c in range(2, d_final + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d_final)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        
        return len(order), edges

    def verify_girth(n, edges, g):
        if g <= 3:
            return True
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Non-backtracking walk BFS from every vertex.
        for s in range(n):
            dist_state = {}
            queue = deque()
            for v in adj[s]:
                state = (v, s)
                dist_state[state] = 1
                queue.append(state)
            
            while queue:
                u, p = queue.popleft()
                d = dist_state[(u, p)]
                if d >= g - 1:
                    continue
                for w in adj[u]:
                    if w == p:
                        continue
                    new_state = (w, u)
                    if new_state not in dist_state:
                        dist_state[new_state] = d + 1
                        if w == s:
                            if d + 1 < g:
                                return False
                        queue.append(new_state)
        return True

    # 2. Small Case Optimizations (Known Cages)
    if k == 3:
        if g <= 3:
            return 4, [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
        elif g <= 4:
            return 6, [(0,3),(0,4),(0,5),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5)]
        elif g <= 5:
            edges = []
            for i in range(5):
                edges.append((i, (i+1)%5))
                edges.append((5+i, 5+(i+2)%5))
                edges.append((i, 5+i))
            return 10, edges
        elif g <= 6:
            edges = []
            lines = [
                (0,1,2), (0,3,4), (0,5,6),
                (1,3,5), (1,4,6), (2,3,6), (2,4,5)
            ]
            for i, line in enumerate(lines):
                line_idx = 7 + i
                for p in line:
                    edges.append((p, line_idx))
            return 14, edges

    elif k == 4:
        if g <= 3:
            return 5, [(i,j) for i in range(5) for j in range(i+1,5)]
        elif g <= 4:
            return 8, [(i,j) for i in range(4) for j in range(4,8)]

    elif k == 5:
        if g <= 3:
            return 6, [(i,j) for i in range(6) for j in range(i+1,6)]
        elif g <= 4:
            return 10, [(i,j) for i in range(5) for j in range(5,10)]
        elif g <= 5:
            vertices = []
            for i in range(16):
                if bin(i).count('1') % 2 == 0:
                    vertices.append(i)
            edges = []
            for idx_i, vi in enumerate(vertices):
                for idx_j, vj in enumerate(vertices):
                    if idx_j <= idx_i:
                        continue
                    diff = vi ^ vj
                    if bin(diff).count('1') == 2:
                        edges.append((idx_i, idx_j))
            n = len(vertices)
            from collections import Counter
            deg = Counter()
            for u, v in edges:
                deg[u] += 1
                deg[v] += 1
            if all(deg[v] == 5 for v in range(n)):
                if verify_girth(n, edges, g):
                    return n, edges

    # 3. General Case: LUW
    return luw_construct(k, g)
