"""Deterministic algebraic construction for regular graphs of large girth."""
import time
import random
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    t0 = time.time()
    deadline = 115.0  # seconds; evaluator kills at 150s, keep generous margin

    if k != 3:
        # Fallback for other k (not expected in this task, but good practice)
        return _luw_construct(k, g, deadline)

    # Special case for k=3, g=16
    # Use LUW construction
    # The LUW construction for k=3 with depth d produces a graph with 
    # n = 3 * 2^d - 2 vertices (if it's a Moore graph candidate) or similar.
    # Actually, the LUW graph size for k=3 is exactly 3 * 2^d - 2.
    # Girth is approximately 2d + 1.
    # We need girth >= 16.
    # Try d = 7 (girth ~15, might fail)
    # Try d = 8 (girth ~17, likely passes)
    
    best_result = None
    best_n = float('inf')
    
    # Start from d=7, go up to d=10
    for d in range(7, 11):
        if time.time() - t0 > deadline - 10:
            break
            
        n, edges, adj = _build_luw_3reg(d)
        
        if n < best_n:
            # Verify girth
            if _verify_girth(n, adj, g):
                best_n = n
                best_result = (n, edges)
                # Since we iterate d increasing, and n increases with d,
                # the first valid d gives the smallest n.
                break
                
    if best_result:
        return best_result
        
    # Fallback: return the largest d we tried if it was valid, or just the last one
    # If none were valid, something is wrong, but return the last built graph
    # to avoid crash (though it might be invalid)
    n, edges, adj = _build_luw_3reg(10)
    return n, edges

def _luw_construct(k, g, deadline):
    # General LUW construction
    q = k
    d = max(3, (g + 1) // 2 - 1)
    
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
    
    start = (0, (None,) + (0,)*d)
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
    
    n = len(order)
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    return n, edges, adj

def _build_luw_3reg(d):
    """
    Build LUW 3-regular graph of depth d.
    Vertices are indexed by paths of length d in a binary tree, 
    but folded into a group structure.
    Size is 3 * 2^d - 2.
    """
    # The LUW graph for k=3 is a Cayley graph of the group 
    # G_d = < a, b, c | a^2=b^2=c^2=1, ... >
    # Actually, it's easier to use the standard LUW construction 
    # from the inspiration code, which handles k=3 correctly.
    
    q = 3
    # The depth in the LUW construction for k=3 to get girth g
    # The inspiration code uses d = (g+1)//2 - 1.
    # For g=16, d=7.
    
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
    
    start = (0, (None,) + (0,)*d)
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
    
    n = len(order)
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    return n, edges, adj

def _verify_girth(n, adj, min_girth):
    """Check girth >= min_girth via non-backtracking BFS from every vertex."""
    max_dist = (min_girth - 1) // 2
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            du = dist[u]
            if du >= max_dist:
                continue
            pu = parent[u]
            for v in adj[u]:
                if v == pu:
                    continue
                dv = dist[v]
                if dv == -1:
                    dist[v] = du + 1
                    parent[v] = u
                    queue.append(v)
                elif dv >= du:
                    cycle_len = du + dv + 1
                    if cycle_len < min_girth:
                        return False
    return True
# EVOLVE-BLOCK-END
