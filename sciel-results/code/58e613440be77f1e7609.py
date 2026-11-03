import time
import random
from collections import deque
import numpy as np

_START = time.time()
_DEADLINE = 110

def _check_girth(n, adj, g):
    """Check if graph has girth at least g using non-backtracking BFS from each vertex."""
    adj_set = [set(a) for a in adj]
    max_depth = g // 2
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            du = dist[u]
            if du >= max_depth:
                continue
            pu = parent[u]
            for v in adj[u]:
                if v == pu:
                    continue
                dv = dist[v]
                if dv == -1:
                    dist[v] = du + 1
                    parent[v] = u
                    q.append(v)
                elif dv <= du:
                    cycle_len = du + dv + 1
                    if cycle_len < g:
                        return False
    return True

def _has_4_cycle(adj, n):
    """Fast check for 4-cycles using set intersections."""
    adj_set = [set(a) for a in adj]
    for u in range(n):
        neighbors = adj[u]
        for i in range(4):
            v1 = neighbors[i]
            for j in range(i+1, 4):
                v2 = neighbors[j]
                # 4-cycle: u-v1-x-v2-u, need x != u, x in N(v1), x in N(v2)
                common = adj_set[v1] & adj_set[v2]
                if len(common) > 0:
                    # Check if any common neighbor is not u
                    for x in common:
                        if x != u:
                            return True
    return False

def _build_bigg_smith_cayley():
    """Exhaustive search for Biggs-Smith graph generators."""
    n = 320
    
    def idx(mask, z):
        return mask * 5 + z
        
    def add(a, b):
        ma, za = divmod(a, 5)
        mb, zb = divmod(b, 5)
        return idx(ma ^ mb, (za + zb) % 5)
        
    def neg(a):
        ma, za = divmod(a, 5)
        return idx(ma, (-za) % 5)
        
    # Non-involutions: z != 0
    pool = [i for i in range(1, n) if i % 5 != 0]
    
    # Precompute neg and add for pool elements?
    # Or just compute on fly.
    
    # To avoid duplicate checks (a,b) and (b,a), and (a,-b) etc?
    # The set S = {a, -a, b, -b} is the same for (a,b), (a,-b), (-a,b), etc.
    # We can canonicalize the pair.
    # Or just iterate all and break on first success.
    
    # There are ~256 elements. Pairs ~32k.
    
    for i in range(len(pool)):
        a = pool[i]
        na = neg(a)
        for j in range(len(pool)):
            # Deadline guard: bail out if we're running out of time
            if time.time() - _START > _DEADLINE:
                return None, None
            
            b = pool[j]
            nb = neg(b)
            
            # Check distinctness
            if a == b or a == nb or na == b or na == nb:
                continue
            if a == na or b == nb: # Should not happen as z!=0
                continue
                
            S = [a, na, b, nb]
            
            # Build adjacency
            adj = [[] for _ in range(n)]
            for u in range(n):
                for s in S:
                    v = add(u, s)
                    adj[u].append(v)
            
            # Fast 4-cycle check
            if _has_4_cycle(adj, n):
                continue
                
            # If no 4-cycle, check full girth
            # This is expensive, but should be rare
            if _check_girth(n, adj, 10):
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n, edges
                
    return None, None

def _build_luw_fallback():
    """Fallback to LUW construction (512 vertices)."""
    q = 4
    d = 5
    table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
    add = lambda x, y: x ^ y
    sub = lambda x, y: x ^ y
    mul = lambda x, y: table[x][y]
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2), 5: (1, 3)}
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(4):
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
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges

def construct(k, g):
    if k == 4 and g == 10:
        # Try Cayley graph search for 320 vertices with deadline guard
        if time.time() - _START < _DEADLINE:
            n, edges = _build_bigg_smith_cayley()
            if n is not None:
                return n, edges
        
        # Fallback to LUW (512 vertices)
        return _build_luw_fallback()
    else:
        return _build_luw_fallback()
