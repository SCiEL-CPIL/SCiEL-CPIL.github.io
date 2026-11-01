import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    if k != 3 or g != 16:
        return _luw_construct(k, g)

    # Base graph: K_{3,3}
    # Generate all closed non-backtracking walks of length 4 to g-1 (14) in K_{3,3}
    # Coefficients for the 9 edges of K_{3,3}
    
    def get_closed_nonbacktracking_walks(max_len):
        all_walks = []
        
        def dfs2(curr, prev, coeff, length):
            if length == max_len:
                if curr == 0:
                    all_walks.append(tuple(coeff))
                return
            if length >= max_len:
                return
            
            if curr < 3:
                nexts = [3, 4, 5]
            else:
                nexts = [0, 1, 2]
                
            for nxt in nexts:
                if nxt == prev:
                    continue
                
                new_coeff = coeff[:]
                if curr < 3:
                    idx = curr * 3 + (nxt - 3)
                    new_coeff[idx] += 1
                else:
                    idx = nxt * 3 + (curr - 3)
                    new_coeff[idx] -= 1
                    
                dfs2(nxt, curr, new_coeff, length + 1)
                
        for L in range(4, max_len, 2):
            dfs2(0, -1, [0]*9, 0)
            
        return list(set(all_walks))

    if time.time() > deadline:
        return _luw_construct(k, g)

    walks = get_closed_nonbacktracking_walks(g)
    
    if not walks:
        return _luw_construct(k, g)

    W = np.array(walks, dtype=np.int32)
    
    found = False
    best_m = None
    best_a = None
    
    rng = np.random.default_rng(42)
    
    for m in [156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170]:
        if time.time() > deadline - 5:
            break
            
        for _ in range(2000):
            if time.time() > deadline - 5:
                break
                
            a = rng.integers(0, m, size=9)
            a[0] = 0
            a[1] = 1
            a[2] = 2
            
            sums = (W @ a) % m
            
            if not np.any(sums == 0):
                found = True
                best_m = m
                best_a = a
                break
        if found:
            break
            
    if not found:
        return _luw_construct(k, g)

    m = best_m
    a = best_a
    n = 6 * m
    edges = []
    for i in range(3):
        for j in range(3):
            for fiber in range(m):
                u = i * m + fiber
                v = (3+j) * m + (fiber + a[i*3+j]) % m
                edges.append((u, v))
                
    return n, edges

def _luw_construct(k, g):
    # Standard LUW fallback
    if k != 3:
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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
    else:
        q = 3
        d = 11 # g-5
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
