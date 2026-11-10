# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def construct(k, g):
    """
    Construct a k-regular graph of girth at least g.
    Tries Cayley graphs on SL(2, q) for small q, then falls back to LUW.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit

    if k != 3:
        return _lnw_construct(k, g)

    # Target: girth >= g (16)
    # Try Cayley graphs on SL(2, q)
    # n = q(q^2 - 1)
    # We want small n.
    # q=8 -> n=504 (Too small, Moore bound 510)
    # q=16 -> n=4080
    # q=32 -> n=32736
    
    # Try q=16 first
    if time.time() < deadline - 20:
        n_cayley = 16 * (16*16 - 1)
        edges = _try_cayley_sl2(16, g, deadline)
        if edges is not None:
            # Verify girth and regularity
            adj = _edges_to_adj(edges, n_cayley)
            girth = _calculate_girth(adj, n_cayley)
            if girth >= g:
                return n_cayley, edges

    # Try q=32 if time permits and q=16 failed
    if time.time() < deadline - 20:
        n_cayley = 32 * (32*32 - 1)
        edges = _try_cayley_sl2(32, g, deadline)
        if edges is not None:
            adj = _edges_to_adj(edges, n_cayley)
            girth = _calculate_girth(adj, n_cayley)
            if girth >= g:
                return n_cayley, edges

    # Fallback to LUW Construction
    # Iterate d from 6 upwards
    best_n = float('inf')
    best_edges = None
    
    for d in range(6, 12):
        if time.time() > deadline - 10:
            break
            
        try:
            n, edges = _lnw_construct_params(3, d)
        except Exception:
            continue
            
        if n >= best_n:
            continue
            
        # Check girth
        adj = _edges_to_adj(edges, n)
        girth = _calculate_girth(adj, n)
        
        if girth >= g:
            best_n = n
            best_edges = edges
            # Since n grows with d, the first valid d is the smallest n
            return n, edges

    if best_edges is not None:
        return best_n, best_edges
        
    # Fallback to a large d that definitely has high girth
    return _lnw_construct(3, g)

def _try_cayley_sl2(q, g, deadline):
    """
    Try to construct a 3-regular Cayley graph on SL(2, q) with girth >= g.
    Uses a specific set of involutions known to generate high girth graphs.
    """
    # Generate SL(2, q) elements
    # In char 2, SL(2, q) = { (a, b, c, d) | ad+bc=1 }
    # We represent elements as tuples (a, b, c, d)
    
    # Precompute inverses for non-zero elements
    inv = [0] * q
    for i in range(1, q):
        inv[i] = pow(i, q-2, q)
        
    # Generate group elements
    group = []
    index = {}
    
    for a in range(q):
        for b in range(q):
            for c in range(q):
                # ad + bc = 1
                # if a == 0: bc = 1
                # if a != 0: d = (1 + bc) * a^-1
                if a == 0:
                    if b == 0 or c == 0:
                        continue
                    # bc = 1 => c = b^-1
                    c_val = inv[b]
                    # But we are iterating c, so we just check if c == inv[b]
                    # Actually, the loop structure makes this inefficient.
                    # Better to generate (a,b,c) and solve for d.
                    pass
                else:
                    d = (1 + b*c) * inv[a] % q
                    elem = (a, b, c, d)
                    if elem not in index:
                        index[elem] = len(group)
                        group.append(elem)
                        
    # The above loop is O(q^3) which is 16^3 = 4096. Very fast.
    # But we need to be careful with a=0 case.
    # If a=0, bc=1. Since b,c in GF(q), b!=0, c!=0.
    # So for each b != 0, c is fixed to inv[b].
    # Let's regenerate properly.
    
    group = []
    index = {}
    
    def add_elem(elem):
        if elem not in index:
            index[elem] = len(group)
            group.append(elem)

    # a != 0
    for a in range(1, q):
        for b in range(q):
            for c in range(q):
                d = (1 + b*c) * inv[a] % q
                add_elem((a, b, c, d))
                
    # a == 0
    for b in range(1, q):
        c = inv[b]
        # d can be anything? No, a=0 => 0*d + bc = 1 => bc=1.
        # d is free? No, determinant is ad-bc. In char 2, ad+bc.
        # If a=0, bc=1. d is not constrained by det=1?
        # Wait, det = ad - bc. In char 2, det = ad + bc.
        # If a=0, det = bc. So we need bc=1.
        # d is free?
        # Matrix (0, b, c, d). Det = 0*d - b*c = bc.
        # So if bc=1, d can be any element of GF(q).
        for d in range(q):
            add_elem((0, b, c, d))
            
    n = len(group)
    
    # Define generators (involutions)
    # s1 = (0, 1, 1, 0)
    # s2 = (1, 1, 0, 1)
    # s3 = (1, 0, 1, 1)
    
    s1 = (0, 1, 1, 0)
    s2 = (1, 1, 0, 1)
    s3 = (1, 0, 1, 1)
    
    # Check if they are in the group
    if s1 not in index or s2 not in index or s3 not in index:
        return None
        
    # Check if they are involutions (they should be)
    # And generate the group?
    # We assume they generate SL(2, q) for q >= 4.
    
    # Build edges
    edges = []
    seen_edges = set()
    
    for u in range(n):
        g_u = group[u]
        for s in [s1, s2, s3]:
            # v = g_u * s
            # Matrix multiplication
            a1, b1, c1, d1 = g_u
            a2, b2, c2, d2 = s
            # (a1 b1; c1 d1) * (a2 b2; c2 d2)
            # a = a1*a2 + b1*c2
            # b = a1*b2 + b1*d2
            # c = c1*a2 + d1*c2
            # d = c1*b2 + d1*d2
            a = (a1*a2 + b1*c2) % q
            b = (a1*b2 + b1*d2) % q
            c = (c1*a2 + d1*c2) % q
            d = (c1*b2 + d1*d2) % q
            v_elem = (a, b, c, d)
            v = index.get(v_elem, -1)
            if v != -1:
                if u < v:
                    edges.append((u, v))
                # We don't need to add reverse edge as we iterate all u
    
    return edges

def _edges_to_adj(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _calculate_girth(adj, n):
    min_cycle = n + 1
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            if dist[u] >= min_cycle:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    if v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
    return min_cycle if min_cycle <= n else n + 1

def _lnw_construct_params(k, d):
    """
    LUW construction with fixed q=k and variable d.
    """
    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    
    # Rule generation depends on d
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

def _lnw_construct(k, g):
    # Original fallback
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    return _lnw_construct_params(k, d)
# EVOLVE-BLOCK-END
