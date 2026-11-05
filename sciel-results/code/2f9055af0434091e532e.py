import time
import math
import numpy as np
from collections import deque

def construct(k, g):
    deadline = time.time() + 140.0

    # Fallback: LUW Construction with optimized d selection
    def luw_construct(k, g):
        q = k
        # Find minimal d such that girth >= g
        # For LUW, girth = d + 5 if d odd, d + 4 if d even
        # We want the smallest d with girth >= g
        best_d = None
        for d in range(3, 20):  # search up to d=19
            girth_est = d + 5 if d % 2 == 1 else d + 4
            if girth_est >= g:
                best_d = d
                break
        if best_d is None:
            best_d = 3  # fallback

        d = best_d

        if q == 4:
            mul_table = [
                [0, 0, 0, 0],
                [0, 1, 2, 3],
                [0, 2, 3, 1],
                [0, 3, 1, 2]
            ]
            def add(x, y): return x ^ y
            def sub(x, y): return x ^ y
            def mul(x, y): return mul_table[x][y]
        else:
            def add(x, y): return (x + y) % q
            def sub(x, y): return (x - y) % q
            def mul(x, y): return (x * y) % q

        rule_a = [0] * (d + 1)
        rule_b = [0] * (d + 1)
        rule_a[2], rule_b[2] = 1, 1
        rule_a[3], rule_b[3] = 2, 1
        rule_a[4], rule_b[4] = 1, 2
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - r
            if r == 0:
                a, b = 1, c0 - 2
            elif r == 1:
                a, b = c0 - 1, 1
            elif r == 2:
                a, b = c0, 1
            else:
                a, b = 1, c0 + 1
            rule_a[c] = a
            rule_b[c] = b

        def compute_neighbours(vertex):
            side, x = vertex
            result = []
            for first in range(q):
                y = [0] * (d + 1)
                y[1] = first
                if side == 0:
                    for c in range(2, d + 1):
                        a, b = rule_a[c], rule_b[c]
                        y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    for c in range(2, d + 1):
                        a, b = rule_a[c], rule_b[c]
                        y[c] = sub(x[c], mul(x[a], y[b]))
                result.append((1 - side, tuple(y)))
            return result

        start = (0, (0,) * (d + 1))
        index = {start: 0}
        order = [start]
        edges = []
        head = 0
        while head < len(order):
            if time.time() > deadline:
                break
            vertex = order[head]
            head += 1
            for other in compute_neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # Try Ramanujan Graph Gamma(3, 11)
    if k == 4 and g <= 10:
        try:
            n, edges = construct_ramanujan(3, 11)
            # Verify girth
            if verify_girth(n, edges, g):
                return n, edges
        except Exception:
            pass

    return luw_construct(k, g)

def construct_ramanujan(p, q):
    # Construct the Ramanujan graph Gamma(p, q) as a quotient of PGL(2, q)
    # For p=3, q=11: vertices are PGL(2,11)/{I, -I} giving 660 vertices
    # The graph is the Cayley graph with 4 generators
    
    # Represent elements of PGL(2, q) as (a, b, c, d) with ad-bc != 0
    # Normalize so that a != 0, and if a == 0, normalize b != 0
    # Two matrices are equivalent if they differ by a scalar
    
    def normalize(m):
        a, b, c, d = m
        # Find a non-zero entry to normalize
        if a != 0:
            inv_a = pow(a, -1, q)
            return (1, (b * inv_a) % q, (c * inv_a) % q, (d * inv_a) % q)
        elif b != 0:
            inv_b = pow(b, -1, q)
            return ((a * inv_b) % q, 1, (c * inv_b) % q, (d * inv_b) % q)
        elif c != 0:
            inv_c = pow(c, -1, q)
            return ((a * inv_c) % q, (b * inv_c) % q, 1, (d * inv_c) % q)
        else:
            inv_d = pow(d, -1, q)
            return ((a * inv_d) % q, (b * inv_d) % q, (c * inv_d) % q, 1)
    
    def mat_mul(m1, m2):
        a1, b1, c1, d1 = m1
        a2, b2, c2, d2 = m2
        return (
            (a1 * a2 + b1 * c2) % q,
            (a1 * b2 + b1 * d2) % q,
            (c1 * a2 + d1 * c2) % q,
            (c1 * b2 + d1 * d2) % q
        )
    
    # Generators for the 4-regular graph
    # Standard generators for PGL(2,11) Ramanujan graph
    gens = [
        (0, 1, 1, 0),  # [[0,1],[1,0]] - involution
        (1, 1, 0, 1),  # [[1,1],[0,1]]
        (1, 0, 1, 1),  # [[1,0],[1,1]]
        (0, 1, -1, 1)  # [[0,1],[-1,1]]
    ]
    # Normalize generators
    gens = [normalize(g) for g in gens]
    
    # Generate all elements of PGL(2, q)
    identity = (1, 0, 0, 1)
    elements = set()
    elements.add(identity)
    queue = deque([identity])
    while queue:
        m = queue.popleft()
        for g in gens:
            ng = normalize(mat_mul(m, g))
            if ng not in elements:
                elements.add(ng)
                queue.append(ng)
    
    # The Ramanujan graph is the quotient by {I, -I}
    # -I in PGL(2,q) is (q-1, 0, 0, q-1) which normalizes to (1, 0, 0, 1) = I
    # Wait, -I = (-1, 0, 0, -1). Normalizing: a=-1, inv_a = pow(-1, -1, 11) = pow(10, -1, 11) = 10 (since 10*10=100=1 mod 11)
    # So (-1, 0, 0, -1) -> (1, 0, 0, 1) = I
    # So -I is already identified with I in PGL!
    # This means the quotient doesn't reduce the size.
    # Actually, PGL(2, q) = GL(2, q) / {scalar matrices}
    # So -I is already in the same coset as I.
    # Therefore |PGL(2, 11)| = |GL(2, 11)| / 10 = (11^2 - 1)(11^2 - 11) / 10 = 120 * 110 / 10 = 1320
    # So we have 1320 vertices, but the Ramanujan graph should have 660.
    # Let me check: the (3,11) Ramanujan graph has 2*(11^3-11)/2 = 1320? No.
    # Actually, the standard Ramanujan graph from PSL(2,11) has 660 vertices.
    # PSL(2, q) = SL(2, q) / {±I} for odd q
    # |SL(2, 11)| = q(q^2-1) = 11*120 = 1320
    # |PSL(2, 11)| = 660
    # So we should use PSL, not PGL.
    
    # Let me restart with PSL(2, 11)
    # Elements of SL(2, q) have determinant 1
    # PSL(2, q) = SL(2, q) / {±I}
    
    # Generate SL(2, 11) elements
    sl_elements = set()
    sl_elements.add(identity)
    queue = deque([identity])
    # Use generators that are in SL(2, q)
    # g1 = [[0,-1],[1,0]] has det = 1
    # g2 = [[1,1],[0,1]] has det = 1
    # g3 = [[1,0],[-1,1]] has det = 1
    # g4 = [[0,1],[1,0]] has det = -1, not in SL
    # Let's use: A = [[0,-1],[1,0]], B = [[1,1],[0,1]], C = [[1,0],[-1,1]], D = [[1,1],[-1,0]] det=1
    sl_gens = [
        (0, 10, 1, 0),  # [[0,-1],[1,0]] in mod 11
        (1, 1, 0, 1),   # [[1,1],[0,1]]
        (1, 0, 10, 1),  # [[1,0],[-1,1]]
        (1, 1, 10, 10)  # [[1,1],[-1,-1]] det = -1-1 = -2 = 9? Let me check: 1*10 - 1*10 = 0. No.
    ]
    # Let me use standard generators for PSL(2,11)
    # From literature: PSL(2,11) is generated by S = [[0,-1],[1,0]] and T = [[1,1],[0,1]]
    # with relations S^2 = T^11 = (ST)^5 = 1
    # For a 4-regular graph, we need 4 generators (or 2 pairs of inverses)
    
    # Let me just use the LUW construction which is guaranteed to work
    raise Exception("Ramanujan construction too complex, using LUW")

def verify_girth(n, edges, g):
    # Exact girth verification using non-backtracking walks
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check that graph is k-regular and simple
    for i in range(n):
        if len(adj[i]) != k:
            return False
        if len(set(adj[i])) != len(adj[i]):
            return False
    
    # For each vertex, do BFS to find shortest cycle
    # Use non-backtracking: track (current, previous) pairs
    for start in range(n):
        # dist[prev][curr] = distance
        from collections import defaultdict
        dist = defaultdict(lambda: defaultdict(int))
        visited = set()
        queue = deque()
        
        for nb in adj[start]:
            dist[start][nb] = 1
            visited.add((start, nb))
            queue.append((start, nb))
        
        while queue:
            prev, curr = queue.popleft()
            for nb in adj[curr]:
                if nb == prev:
                    continue
                if (prev, curr, nb) in visited:
                    continue
                # Check if nb is start
                if nb == start:
                    if dist[prev][curr] + 1 < g:
                        return False
                elif (curr, nb) in visited:
                    # Found a cycle
                    cycle_len = dist[prev][curr] + 1 + dist[nb][curr]
                    if cycle_len < g:
                        return False
                else:
                    dist[curr][nb] = dist[prev][curr] + 1
                    visited.add((curr, nb))
                    queue.append((curr, nb))
    
    return True
