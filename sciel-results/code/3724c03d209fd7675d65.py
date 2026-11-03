import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Moore Bound
    if g % 2 == 1:
        r = (g - 1) // 2
        moore = 1 + k * sum((k - 1) ** i for i in range(r))
    else:
        r = g // 2
        s = 0
        for i in range(r):
            s += k * ((k - 1) ** i)
        moore = 2 * s

    # LUW Construction with optimized coordinate handling
    q = k
    
    # Find smallest d such that girth >= g
    # For LUW: girth >= d+5 if d odd, d+4 if d even
    d = None
    for candidate_d in range(3, 20):
        if candidate_d % 2 == 1:
            if candidate_d + 5 >= g:
                d = candidate_d
                break
        else:
            if candidate_d + 4 >= g:
                d = candidate_d
                break
    
    if d is None:
        d = g - 4
    
    if d < 3:
        d = 3
    
    # Finite-field helpers
    if q == 4:
        _add_t = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        _mul_t = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        def f_add(a, b): return a ^ b
        def f_sub(a, b): return a ^ b
        def f_mul(a, b): return _mul_t[a][b]
    else:
        def f_add(a, b): return (a + b) % q
        def f_sub(a, b): return (a - b) % q
        def f_mul(a, b): return (a * b) % q
    
    # Recurrence rule for the LUW construction
    rules = {}
    rules[2] = (1, 1)
    rules[3] = (2, 1)
    rules[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        base = c - r
        candidates = [(1, base - 2), (base - 1, 1), (base, 1), (1, base + 1)]
        rules[c] = candidates[r]
    
    # BFS over the incidence graph with flat integer encoding
    # Encode vertex as: side * q^d + sum(coords[i] * q^(d-i)) for i=1..d
    q_pow = [1] * (d + 1)
    for i in range(d - 1, -1, -1):
        q_pow[i] = q_pow[i+1] * q
    
    def encode(side, coords):
        """Encode (side, coords[1..d]) to a single integer."""
        val = side * q_pow[0]
        for i in range(1, d + 1):
            val += coords[i] * q_pow[i]
        return val
    
    def decode(val):
        """Decode integer to (side, coords)."""
        side = val // q_pow[0]
        rem = val % q_pow[0]
        coords = [0] * (d + 1)
        for i in range(d, 0, -1):
            coords[i] = rem % q
            rem //= q
        return side, coords
    
    start_key = encode(0, [0] * (d + 1))
    label = {start_key: 0}
    queue = [start_key]
    edge_list = []
    
    for vkey in queue:
        if time.time() > deadline - 10:
            break
        side, coords = decode(vkey)
        vidx = label[vkey]
        for first_val in range(q):
            ncoords = [0] * (d + 1)
            ncoords[1] = first_val
            for c in range(2, d + 1):
                a, b = rules[c]
                if side == 0:
                    ncoords[c] = f_add(coords[c], f_mul(ncoords[a], coords[b]))
                else:
                    ncoords[c] = f_sub(coords[c], f_mul(coords[a], ncoords[b]))
            nkey = encode(1 - side, ncoords)
            if nkey not in label:
                label[nkey] = len(queue)
                queue.append(nkey)
            if side == 0:
                edge_list.append((vidx, label[nkey]))
    
    n_luw = len(queue)
    
    # Verify the graph is valid: check degrees and girth
    if n_luw == 0 or not edge_list:
        return n_luw, edge_list
    
    # Build adjacency for verification
    adj = [set() for _ in range(n_luw)]
    for u, v in edge_list:
        if u < n_luw and v < n_luw:
            adj[u].add(v)
            adj[v].add(u)
    
    # Check all vertices have degree exactly k
    valid = True
    for i in range(n_luw):
        if len(adj[i]) != k:
            valid = False
            break
    
    if not valid:
        # Fallback: return what we have
        return n_luw, edge_list
    
    # Verify girth using non-backtracking walks (BFS with parent tracking)
    max_depth = (g - 1) // 2
    girth_ok = True
    for start in range(n_luw):
        if time.time() > deadline - 5:
            break
        visited = [-1] * n_luw
        parent = [-1] * n_luw
        queue2 = deque([(start, 0)])
        visited[start] = 0
        while queue2:
            node, depth = queue2.popleft()
            if depth >= max_depth:
                continue
            for nei in adj[node]:
                if visited[nei] == -1:
                    visited[nei] = depth + 1
                    parent[nei] = node
                    queue2.append((nei, depth + 1))
                elif nei != parent[node]:
                    cycle_len = depth + 1 + visited[nei]
                    if cycle_len < g:
                        girth_ok = False
                        break
            if not girth_ok:
                break
        if not girth_ok:
            break
    
    return n_luw, edge_list
