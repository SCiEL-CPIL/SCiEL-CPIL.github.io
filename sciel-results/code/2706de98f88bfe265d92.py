"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

_DEADLINE = 150.0
_SAFETY = 10.0

def _time_left():
    return _DEADLINE - _SAFETY - time.time()

def _lazebnik_construct(k, g):
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
                y[c] = add(x[c], mul(y[a], x[b])) if side==0 else sub(x[c], mul(x[a], y[b]))
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

def _girth_ok(adj, n, g):
    """Check girth >= g via non-backtracking BFS from each directed edge."""
    if g <= 2:
        return True
    max_d = g - 2
    if max_d < 1:
        return True
    
    for u in range(n):
        for v in adj[u]:
            seen = set()
            q = deque()
            q.append((v, u, 0))
            seen.add((v, u))
            while q:
                node, prev, dep = q.popleft()
                if dep >= max_d:
                    continue
                for nxt in adj[node]:
                    if nxt == prev:
                        continue
                    if nxt == u and dep + 2 < g:
                        return False
                    st = (nxt, node)
                    if st not in seen:
                        seen.add(st)
                        q.append((nxt, node, dep+1))
    return True

def _random_kreg(n, k, seed):
    if n*k % 2 != 0 or k >= n:
        return None
    rng = random.Random(seed)
    stubs = [v for v in range(n) for _ in range(k)]
    rng.shuffle(stubs)
    adj = [set() for _ in range(n)]
    for i in range(0, n*k, 2):
        u, v = stubs[i], stubs[i+1]
        if u == v or v in adj[u]:
            return None
        adj[u].add(v)
        adj[v].add(u)
    adj = [list(s) for s in adj]
    edges = [(u,v) for u in range(n) for v in adj[u] if u < v]
    return adj, edges

def construct(k, g):
    t0 = time.time()
    luw_n, luw_edges = _lazebnik_construct(k, g)
    best_n, best_edges = luw_n, luw_edges
    
    # Moore bound
    if g % 2 == 1:
        r = (g-3)//2
        moore = 1 + k * sum((k-1)**i for i in range(r+1))
    else:
        r = g//2 - 1
        moore = 2 * sum((k-1)**i for i in range(r+1))
        
    start_n = max(moore, k+1)
    
    # Try randomized search for smaller graphs
    # Limit the number of n values to avoid running out of time
    # We check n values in increasing order
    for n in range(start_n, best_n):
        if time.time() - t0 > _DEADLINE - _SAFETY:
            break
        if (n * k) % 2 != 0:
            continue
        # Try a few seeds
        for seed in range(3):
            if time.time() - t0 > _DEADLINE - _SAFETY:
                break
            res = _random_kreg(n, k, seed)
            if res is None:
                continue
            adj, edges = res
            if _girth_ok(adj, n, g):
                return n, edges
    return best_n, best_edges
# EVOLVE-BLOCK-END
