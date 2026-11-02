import time
import math
from collections import deque

def construct(k, g):
    deadline = time.time() + 115  # Safety margin
    
    # Moore bound
    if k == 1:
        return g, [(0, 1) for _ in range(g // 2)] # Just a path? No, 1-regular is matching.
    # For k=3, g=16
    # M = 2 * sum_{i=0}^{g/2 - 1} (k-1)^i
    r = g // 2
    moore_bound = 2 * sum((k - 1) ** i for i in range(r))
    if moore_bound % 2 != 0:
        moore_bound += 1
        
    # Search for circulant graph C(n, {a, -a, n/2})
    # n must be even
    max_n = 3000 
    # If k != 3, the specific circulant form C(n, {a, -a, n/2}) is 3-regular.
    # For general k, this strategy is less direct. 
    # The task specifies k=3, g=16 in the description, but interface takes k, g.
    # The inspirations focus on k=3.
    # If k != 3, we might skip this specific search or adapt it.
    # However, the LUW fallback works for any prime q=k.
    # Let's restrict the circulant search to k=3 for safety, or check if k allows it.
    # C(n, S) is |S|-regular. So this specific form is only for k=3.
    
    if k == 3:
        for n in range(moore_bound, max_n + 1, 2):
            if time.time() > deadline:
                break
            h = n // 2
            for a in range(1, h):
                if math.gcd(h, a) != 1:
                    continue
                if _check_girth_circulant(n, a, g):
                    edges = []
                    for u in range(n):
                        for s in (a, n - a, h):
                            v = (u + s) % n
                            if u < v:
                                edges.append((u, v))
                    return n, edges
    else:
        # For other k, we could try other circulants, but let's stick to LUW
        # or a simple search if feasible.
        # Given the prompt focuses on 3-regular, we prioritize that.
        pass

    return _luw_fallback(k, g)

def _check_girth_circulant(n, a, g):
    h = n // 2
    S = [a, n - a, h]
    # Inverses: 0<->1, 2<->2
    inv = [1, 0, 2]
    
    # BFS states: (v, p)
    # Start: take one step from 0
    # v = S[p], p = p
    visited = [[False] * 3 for _ in range(n)]
    
    q = deque()
    for p in range(3):
        v = S[p]
        if not visited[v][p]:
            visited[v][p] = True
            q.append((v, p, 1))
            
    while q:
        v, p, d = q.popleft()
        if v == 0:
            # Found a cycle of length d
            return d >= g
        if d >= g - 1:
            # We have explored all walks of length < g
            # If we didn't hit 0, girth >= g
            # Note: we break out if we finish a level
            # Actually, we can just return True if we finish BFS without hitting 0
            # But we need to ensure we explored all depths < g
            # Since we are doing BFS, if we pop a node at depth d, all nodes at depth < d are processed.
            # If we reach depth g-1 and don't hit 0, we are safe.
            # However, we might have nodes in queue at depth g-1.
            # We just need to ensure no node at depth < g is 0.
            # So if we are about to process depth g, we can stop.
            # But we check v==0 on pop.
            # So if we pop (0, p, d) with d < g, we return False.
            # If queue becomes empty or min depth >= g, return True.
            pass
        
        for np_ in range(3):
            if np_ == inv[p]:
                continue
            nv = (v + S[np_]) % n
            if not visited[nv][np_]:
                visited[nv][np_] = True
                q.append((nv, np_, d + 1))
                
    return True

def _luw_fallback(k, g):
    # The original LUW construction
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
