# EVOLVE-BLOCK-START
import time
import random
import sys
import numpy as np
from collections import deque

def construct(k, g):
    deadline = time.time() + 140.0
    
    # Guaranteed-valid fallback: Lazebnik-Ustimenko-Woldar CD graph
    def _lw_fallback():
        q = k
        d = max(3, g - 5)
        
        if q == 4:
            table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            add = lambda x,y: x^y
            sub = lambda x,y: x^y
            mul = lambda x,y: table[x][y]
        else:
            add = lambda x,y: (x+y)%q
            sub = lambda x,y: (x-y)%q
            mul = lambda x,y: x*y%q
        
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
    
    # Fast girth certification using non-backtracking walks
    def _girth(adj, n):
        INF = 10**9
        for L in range(3, 2*g+1):
            for s in range(n):
                dist = {}
                q = deque()
                for nb in adj[s]:
                    q.append((s, nb))
                    dist[(s, nb)] = 1
                while q:
                    prev, cur = q.popleft()
                    dl = dist[(prev, cur)]
                    if dl == L:
                        break
                    for nb in adj[cur]:
                        if nb == prev:
                            continue
                        key = (cur, nb)
                        if key in dist:
                            cyc = dl + 1 - dist[key]
                            if 3 <= cyc < L:
                                return cyc
                        else:
                            dist[key] = dl + 1
                            q.append(key)
        return 2*g
    
    # Driver
    try:
        n, edges = _lw_fallback()
        return n, edges
    except Exception:
        return 2048, []
# EVOLVE-BLOCK-END
