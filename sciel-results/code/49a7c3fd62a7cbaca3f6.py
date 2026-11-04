"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

def construct(k, g):
    deadline = time.time() + 110
    
    q = k
    d = max(2, g - 5)
    
    if q == 4:
        t4 = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add = lambda x,y: x^y
        sub = lambda x,y: x^y
        mul = lambda x,y: t4[x][y]
    else:
        add = lambda x,y: (x+y)%q
        sub = lambda x,y: (x-y)%q
        mul = lambda x,y: (x*y)%q
    
    # LUW(q,d): points and lines are d-tuples over GF(q)
    # Point p=(p1..pd) ~ Line l=(l1..ld) iff for i=2..d: li - pi = l(i-1)*p(i-1)
    # with l(0)=0, p(0)=0
    
    start = (0, tuple([0]*d))
    index = {start: 0}
    order = [start]
    edges = []
    edge_set = set()
    
    queue = deque([start])
    
    def nbrs(v):
        side, x = v
        result = []
        for c0 in range(q):
            if side == 0:
                # x is point, find lines l with l1=c0, and li = xi + l(i-1)*x(i-1) for i=2..d
                y = [0]*(d+1)
                y[1] = c0
                for i in range(2, d+1):
                    y[i] = add(x[i-1], mul(y[i-1], x[i-2])) if i-2 >= 1 else x[i-1]
                result.append((1, tuple(y[1:d+1])))
            else:
                # x is line, find points p with p1=c0, and pi = li - l(i-1)*p(i-1) for i=2..d
                y = [0]*(d+1)
                y[1] = c0
                for i in range(2, d+1):
                    if i-2 >= 1:
                        y[i] = sub(x[i-1], mul(x[i-2], y[i-1]))
                    else:
                        y[i] = x[i-1]
                result.append((0, tuple(y[1:d+1])))
        return result
    
    while queue and time.time() < deadline:
        v = queue.popleft()
        vi = index[v]
        for w in nbrs(v):
            if w not in index:
                index[w] = len(order)
                order.append(w)
                queue.append(w)
            wj = index[w]
            if v[0] == 0:
                key = (vi, wj)
                if key not in edge_set:
                    edge_set.add(key)
                    edges.append(key)
    
    return len(order), edges
# EVOLVE-BLOCK-END
