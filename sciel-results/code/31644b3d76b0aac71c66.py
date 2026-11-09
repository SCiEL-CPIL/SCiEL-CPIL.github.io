import time
import random
import math
from collections import deque

def construct(k, g):
    t0 = time.time()
    deadline = t0 + 115

    def moore_bound(k, g):
        if g % 2 == 0:
            h = g // 2
            return 2 * ((k - 1)**h - 1) // (k - 2) if k > 2 else 2 * h
        else:
            h = (g - 1) // 2
            return 1 + k * ((k - 1)**h - 1) // (k - 2) if k > 2 else 1 + k * h

    mb = moore_bound(k, g)
    if mb % 2 != 0:
        mb += 1

    def girth_cayley(n, a, c, g_max):
        # Generators: a, -a, c (where c = n//2)
        # Check for non-backtracking cycles of length < g_max
        seen = set()
        frontier = []
        for start_step in [a, -a, c]:
            if start_step == 0:
                continue
            ns = start_step % n
            frontier.append((ns, 0))
            seen.add((ns, 0))
        
        for length in range(1, g_max):
            new_frontier = []
            for (u, p) in frontier:
                for nb in [a, -a, c]:
                    if nb == 0:
                        continue
                    ns = (u + nb) % n
                    if ns == p:
                        continue
                    if ns == 0:
                        return length + 1
                    state = (ns, u)
                    if state not in seen:
                        seen.add(state)
                        new_frontier.append(state)
            frontier = new_frontier
            if not frontier:
                break
        return g_max

    def make_edges(n, a, c):
        edges = []
        for i in range(n):
            for j in [(i + a) % n, (i - a) % n, (i + c) % n]:
                if i < j:
                    edges.append((i, j))
        return edges

    # Search
    candidates = list(range(mb, 937, 2))
    
    for n_try in candidates:
        if time.time() > deadline - 5:
            break
        
        c = n_try // 2
        valid_as = []
        for a in range(1, c):
            if a == c:
                continue
            if (2 * a) % n_try == 0:
                continue
            if math.gcd(a, n_try) != 1:
                continue
            valid_as.append(a)
        
        if not valid_as:
            continue
        
        random.shuffle(valid_as)
        
        # Limit attempts per n to save time for larger n
        max_attempts = 50 if n_try < 700 else 20
        for a in valid_as[:max_attempts]:
            if time.time() > deadline - 5:
                break
            girth = girth_cayley(n_try, a, c, g)
            if girth >= g:
                return n_try, make_edges(n_try, a, c)

    # Fallback: LUW construction
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add, sub, mul = (lambda x,y: x^y), (lambda x,y: x^y), (lambda x,y: table[x][y])
    else:
        add, sub, mul = (lambda x,y: (x+y)%q), (lambda x,y: (x-y)%q), (lambda x,y: x*y%q)
    rule = {2:(1,1), 3:(2,1), 4:(1,2)}
    for c in range(5, d+1):
        r, c0 = (c-5)%4, c-(c-5)%4
        rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0]*(d-1)
            for c in range(2, d+1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1-side, tuple(y)
    
    start = (0, (None,)+(0,)*d)
    index, order, edges = {start:0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
