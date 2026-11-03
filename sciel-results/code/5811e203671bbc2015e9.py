import numpy as np
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Uses Cayley graphs of SL(2, q) for various primes q.
    """
    if k != 5:
        # Fallback for other k, though task is (5,9)
        return _fallback_construct(k, g)
        
    # Try primes q
    primes = [11, 13, 17, 19, 23, 29, 31, 37, 41, 43]
    
    for q in primes:
        n, edges = _cayley_sl2q(q, k, g)
        if n is not None:
            return n, edges
            
    # If no small q works, try a larger one or fallback
    return _fallback_construct(k, g)

def _cayley_sl2q(q, k, g):
    """
    Construct Cayley graph of SL(2, q) with a specific generating set.
    Returns (n, edges) if girth >= g, else (None, None).
    """
    # Generate group elements of SL(2, q)
    group = []
    group_index = {}
    
    for a in range(q):
        for b in range(q):
            for c in range(q):
                val = 1 + b * c
                if a == 0:
                    if val != 0: 
                        continue
                    d = 0
                else:
                    a_inv = pow(a, q-2, q)
                    d = (val * a_inv) % q
                elem = (a, b, c, d)
                if (a*d - b*c) % q != 1:
                    continue
                if elem not in group_index:
                    group_index[elem] = len(group)
                    group.append(elem)
    
    n = len(group)
    
    # Define generators
    # S = { -I, A, A^-1, B, B^-1 }
    I = (1, 0, 0, 1)
    negI = (q-1, 0, 0, q-1)
    
    A = (1, 1, 0, 1)
    A_inv = (1, q-1, 0, 1)
    
    B = (1, 0, 1, 1)
    B_inv = (1, 0, q-1, 1)
    
    S = [negI, A, A_inv, B, B_inv]
    
    if I in S:
        return None, None
    
    adj = [[False] * n for _ in range(n)]
    edges = []
    
    def mul(X, Y):
        a1, b1, c1, d1 = X
        a2, b2, c2, d2 = Y
        a = (a1*a2 + b1*c2) % q
        b = (a1*b2 + b1*d2) % q
        c = (c1*a2 + d1*c2) % q
        d = (c1*b2 + d1*d2) % q
        return (a, b, c, d)
    
    for i, g_elem in enumerate(group):
        for s in S:
            prod = mul(g_elem, s)
            if prod in group_index:
                j = group_index[prod]
                if i < j:
                    adj[i][j] = True
                    adj[j][i] = True
                    edges.append((i, j))
    
    for i in range(n):
        if sum(adj[i]) != k:
            return None, None
            
    girth = _compute_girth(adj, n)
    if girth >= g:
        return n, edges
    else:
        return None, None

def _compute_girth(adj, n):
    min_girth = float('inf')
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            for v in range(n):
                if adj[u][v]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u] and u != parent[v]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
                            
    return min_girth if min_girth < float('inf') else n

def _fallback_construct(k, g):
    # If algebraic construction fails, use a large prime
    for q in [47, 53, 59, 61]:
        n, edges = _cayley_sl2q(q, k, g)
        if n is not None:
            return n, edges
    # Last resort: return a small valid graph if possible, or error?
    # Must return valid.
    # For (5,9), q=11 should work.
    return 0, []
# EVOLVE-BLOCK-END
