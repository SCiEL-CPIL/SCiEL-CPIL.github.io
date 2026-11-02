import time
from collections import deque
import random

def construct(k, g):
    t0 = time.time()
    deadline = 115.0

    def certify_girth(n, edges, g):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        N2 = n * n
        INF = g
        dist = [INF] * N2
        q = deque()
        
        for a in range(n):
            for b in adj[a]:
                dist[a * n + b] = 1
                q.append(a * n + b)
                
        while q:
            s = q.popleft()
            d = dist[s]
            if d >= g - 1:
                continue
            p = s // n
            c = s % n
            rev = c * n + p
            dr = dist[rev]
            if dr != INF:
                if d + dr + 1 < g:
                    return False
            for nb in adj[c]:
                if nb == p:
                    continue
                ns = c * n + nb
                nd = d + 1
                if nd < dist[ns]:
                    dist[ns] = nd
                    q.append(ns)
        return True

    best_n, best_edges = None, None

    if k == 3 and g == 16:
        n_target = 936
        
        # Use Cayley graph of Z_3 x Z_312 with 3 generators
        # Elements: (i, j) where i in {0,1,2}, j in Z_312
        # Map to vertex: i * 312 + j
        # Generators: (a1, b1), (a2, b2), (a3, b3) with inverses also in the set
        # For 3-regular: need exactly 3 generators where one is self-inverse or we pair them
        
        # For 3-regular Cayley: generators S = {s1, s2, s3} where s_i^{-1} is also in S
        # So either all self-inverse, or one self-inverse and two that are inverses of each other
        
        # Option: S = {s, s^{-1}, t} where t = t^{-1} (self-inverse)
        # In Z_3 x Z_312: self-inverse elements have order 1 or 2
        # Order 2 elements: (0, 0) [identity, skip], (1, 0), (2, 0), (0, 156)
        
        random.seed(99)
        
        tried = set()
        
        # Try various generator sets
        for t_choice in [(1, 0), (2, 0), (0, 156), (1, 156), (2, 156)]:
            for a1 in range(3):
                for b1 in range(1, 312):
                    if time.time() - t0 > deadline - 20:
                        break
                    s = (a1, b1)
                    s_inv = ((3 - a1) % 3, (312 - b1) % 312)
                    if s == s_inv:
                        continue
                    if s == t_choice:
                        continue
                    if s_inv == t_choice:
                        continue
                    
                    key = (t_choice, a1, b1)
                    if key in tried:
                        continue
                    tried.add(key)
                    
                    # Build Cayley graph
                    adj = [[] for _ in range(n_target)]
                    edges = []
                    eset = set()
                    gens = [s, s_inv, t_choice]
                    for i in range(3):
                        for j in range(312):
                            u = i * 312 + j
                            for (ga, gb) in gens:
                                ni = (i + ga) % 3
                                nj = (j + gb) % 312
                                v = ni * 312 + nj
                                adj[u].append(v)
                                if u < v:
                                    kk = (u, v)
                                    if kk not in eset:
                                        eset.add(kk)
                                        edges.append(kk)
                    
                    # Check degree
                    if any(len(adj[u]) != 3 for u in range(n_target)):
                        continue
                    
                    if certify_girth(n_target, edges, g):
                        best_n, best_edges = n_target, edges
                        break
                if best_n is not None:
                    break
            if best_n is not None:
                break
        
        # Fallback: LUW
        if best_n is None and time.time() - t0 < deadline - 5:
            def luw_construct(k, g, d):
                q = k
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

            for d in [8, 9, 10]:
                if time.time() - t0 > deadline - 3:
                    break
                try:
                    n, edges = luw_construct(k, g, d)
                    if certify_girth(n, edges, g):
                        best_n, best_edges = n, edges
                        break
                except Exception:
                    continue

    if best_n is None:
        def luw_construct(k, g, d):
            q = k
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

        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        best_n, best_edges = luw_construct(k, g, d)
    
    return best_n, best_edges
