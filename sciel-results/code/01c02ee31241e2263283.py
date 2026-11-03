import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

    # Target: k=5, g=9
    # Try LPS construction for p=11
    if k == 5 and g == 9:
        res = _try_lps_5_9()
        if res is not None:
            return res

    # Fallback to LUW
    return _luw_construct(k, g)

def _try_lps_5_9():
    p = 11
    # Generate SL(2, p)
    # Elements are (a,b,c,d) with ad - bc = 1
    # We can iterate a,b,c and solve for d?
    # ad - bc = 1 => d = (1+bc)/a mod p (if a != 0)
    # If a = 0, then -bc = 1 => c = -b^{-1}, d can be anything?
    # No, if a=0, then -bc=1, so c = -b^{-1}. d is free?
    # No, the matrix is [[a,b],[c,d]].
    # If a=0, then -bc=1.
    
    group = []
    for a in range(p):
        for b in range(p):
            for c in range(p):
                for d in range(p):
                    if (a*d - b*c) % p == 1:
                        group.append((a,b,c,d))
    
    # Map group element to index
    idx_map = {g: i for i, g in enumerate(group)}
    n = len(group)
    
    # Candidate generators
    # Standard LPS generators for p=11
    # We need 5 generators.
    # Let's try a few sets.
    
    # Set 1: Standard + 1 random
    g1 = (1, 1, 0, 1)
    g2 = (1, 0, 1, 1)
    g3 = (0, -1, 1, 0)
    g4 = (1, 1, 1, 0)
    g5 = (1, -1, 0, 1)
    
    gens = [g1, g2, g3, g4, g5]
    
    # Check if they are distinct and not inverses
    # Inverse of (a,b,c,d) is (d,-b,-c,a)
    
    edges = []
    adj = [[] for _ in range(n)]
    
    for i, g in enumerate(group):
        for j, gen in enumerate(gens):
            # Multiply g * gen
            a1,b1,c1,d1 = g
            a2,b2,c2,d2 = gen
            a3 = (a1*a2 + b1*c2) % p
            b3 = (a1*b2 + b1*d2) % p
            c3 = (c1*a2 + d1*c2) % p
            d3 = (c1*b2 + d1*d2) % p
            ng = (a3,b3,c3,d3)
            ni = idx_map[ng]
            if i < ni:
                edges.append((i, ni))
                adj[i].append(ni)
                adj[ni].append(i)
            # Note: This adds edges for each generator.
            # Since we have 5 generators, and we iterate all i,
            # each vertex will have degree 5 (if all generators are distinct and not inverses).
            # We must ensure we don't add duplicate edges.
            # The set of neighbors is { g * gen | gen in gens }.
            # If g * gen1 == g * gen2, then gen1 == gen2.
            # If g * gen1 == g * gen2^{-1}, then gen1 == gen2^{-1}.
            # So we must ensure gens are distinct and no inverse pairs.
            
    # Check girth
    if _check_girth(n, edges, 9):
        return n, edges
        
    # Try random generators if fixed set fails
    for _ in range(50):
        random_gens = random.sample(group, 5)
        # Ensure no inverses
        valid = True
        for i in range(5):
            for j in range(i+1, 5):
                if random_gens[i] == random_gens[j]:
                    valid = False
                inv = (random_gens[j][3], -random_gens[j][1] % p, -random_gens[j][2] % p, random_gens[j][0])
                if random_gens[i] == inv:
                    valid = False
        if not valid:
            continue
            
        edges = []
        adj = [[] for _ in range(n)]
        for i, g in enumerate(group):
            for gen in random_gens:
                a1,b1,c1,d1 = g
                a2,b2,c2,d2 = gen
                a3 = (a1*a2 + b1*c2) % p
                b3 = (a1*b2 + b1*d2) % p
                c3 = (c1*a2 + d1*c2) % p
                d3 = (c1*b2 + d1*d2) % p
                ng = (a3,b3,c3,d3)
                ni = idx_map[ng]
                if i < ni:
                    edges.append((i, ni))
                    adj[i].append(ni)
                    adj[ni].append(i)
                    
        if _check_girth(n, edges, 9):
            return n, edges
            
    return None

def _check_girth(n, edges, g):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = [start]
        for v in q:
            d = dist[v]
            if d >= g // 2:
                continue
            for u in adj[v]:
                if dist[u] == -1:
                    dist[u] = d + 1
                    parent[u] = v
                    q.append(u)
                else:
                    if u == parent[v]:
                        continue
                    cycle_len = d + 1 + dist[u]
                    if cycle_len < g:
                        return False
    return True

def _luw_construct(k, g):
    # Same as before
    q = k
    d = g - 5 if (g - 5) % 2 else g - 4
    if d < 3:
        d = 3

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

    rules = {}
    rules[2] = (1, 1)
    rules[3] = (2, 1)
    rules[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        base = c - r
        candidates = [(1, base - 2), (base - 1, 1), (base, 1), (1, base + 1)]
        rules[c] = candidates[r]

    start_key = (0, (0,) * (d + 1))
    label = {start_key: 0}
    queue = [start_key]
    edge_list = []

    for vkey in queue:
        side, coords = vkey
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
            nkey = (1 - side, tuple(ncoords))
            if nkey not in label:
                label[nkey] = len(queue)
                queue.append(nkey)
            if side == 0:
                edge_list.append((vidx, label[nkey]))

    return len(queue), edge_list
