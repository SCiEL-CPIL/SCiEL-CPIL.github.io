import time
import random
from collections import deque
import numpy as np

def construct(k, g):
    deadline = time.time() + 110  # 10s safety margin

    # Fallback: LUW construction (kept from original)
    def luw_construct(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
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

    # New Strategy: Cayley Graph Search on SL(2, Z_p)
    def try_cayley_sl2(k, g):
        if k != 3:
            return None
            
        # Try primes p such that |SL(2, Z_p)| is reasonable
        # |SL(2, Z_p)| = p(p^2 - 1)
        # p=7: 336, p=11: 1320, p=13: 2184
        # We want n ~ 936. 
        # If we use a subgroup of index 2 in SL(2, Z_13), n=1092.
        # If we use SL(2, Z_7), n=336 (too small for girth 16? Moore bound for (3,16) is ~512).
        # So p=7 is too small. p=11 gives 1320. p=13 gives 2184.
        # Let's try to find a 3-regular graph on a subgroup or the whole group.
        
        primes = [11, 13, 17]
        
        for p in primes:
            if time.time() > deadline - 20:
                return None
            
            # Generate all elements of SL(2, Z_p)
            # Matrix is [[a, b], [c, d]] with ad - bc = 1
            elements = []
            elem_to_idx = {}
            for a in range(p):
                for b in range(p):
                    for c in range(p):
                        # d = (1 + bc) * a^-1 mod p
                        if a == 0:
                            # If a=0, then -bc = 1 => bc = -1.
                            # d is free? No, ad-bc=1 => -bc=1.
                            # If a=0, c must be non-zero.
                            # d can be anything? No, the equation is 0*d - bc = 1.
                            # So bc = -1. d is not constrained by the determinant eq?
                            # Wait, the determinant is ad - bc.
                            # If a=0, det = -bc. So -bc = 1 => bc = -1.
                            # d is free? No, the matrix is determined by a,b,c,d.
                            # We iterate a,b,c. If a!=0, d is fixed.
                            # If a==0, we need bc = -1.
                            # Then d can be any value?
                            # No, the group is SL(2), so det=1.
                            # The elements are just the matrices.
                            # So if a=0, we require bc = -1 mod p.
                            # And d can be any value in 0..p-1?
                            # Yes, because the condition is only on the determinant.
                            if c == 0:
                                continue
                            bc = (b * c) % p
                            if bc != (p - 1) % p: # -1 mod p
                                continue
                            for d in range(p):
                                mat = (a, b, c, d)
                                if mat not in elem_to_idx:
                                    elem_to_idx[mat] = len(elements)
                                    elements.append(mat)
                        else:
                            # a != 0
                            # ad - bc = 1 => ad = 1 + bc
                            # d = (1 + bc) * a^-1
                            a_inv = pow(a, p - 2, p)
                            d = ((1 + b * c) * a_inv) % p
                            mat = (a, b, c, d)
                            if mat not in elem_to_idx:
                                elem_to_idx[mat] = len(elements)
                                elements.append(mat)
            
            n = len(elements)
            
            # Define multiplication
            def mat_mul(m1, m2):
                a1, b1, c1, d1 = m1
                a2, b2, c2, d2 = m2
                a = (a1 * a2 + b1 * c2) % p
                b = (a1 * b2 + b1 * d2) % p
                c = (c1 * a2 + d1 * c2) % p
                d = (c1 * b2 + d1 * d2) % p
                return (a, b, c, d)
            
            # Define inversion
            def mat_inv(m):
                a, b, c, d = m
                # det = 1
                # inv = [[d, -b], [-c, a]]
                return (d, (-b) % p, (-c) % p, a)
            
            # Precompute multiplication table for speed? 
            # n is up to 2184. n^2 = ~4.7M. Might be slow in Python.
            # We will compute on the fly but cache if needed.
            
            # Candidate generators: "small" matrices
            # A = [[1, 1], [0, 1]]
            # B = [[1, 0], [1, 1]]
            # C = [[0, -1], [1, 0]]
            # D = [[-1, 0], [0, 1]]
            
            gens = [
                (1, 1, 0, 1), # A
                (1, 0, 1, 1), # B
                (0, p-1, 1, 0), # C (0, -1; 1, 0)
                (p-1, 0, 0, 1), # D (-1, 0; 0, 1)
                (1, 0, 0, 1), # I (Identity, exclude)
            ]
            
            # We need a set of 3 generators S = {s1, s2, s3}
            # Such that the Cayley graph is 3-regular and connected.
            # And girth >= g.
            
            # Randomly sample triples from a larger set of candidates
            # To ensure symmetry, if we pick s, we must also have s^-1?
            # No, the graph is undirected. So if s is in S, s^-1 is effectively in S for connectivity.
            # But the degree is |S|. So S must be symmetric: s in S => s^-1 in S.
            # If |S|=3, and S is symmetric, then one element must be its own inverse (order 2)
            # and the other two are inverses of each other.
            # Or, we can define the graph with edges (u, u*s) for s in S.
            # If S is not symmetric, the graph is directed.
            # For undirected 3-regular, we need S to be symmetric.
            # So S = {x, x^-1, y} where y = y^-1 (order 2).
            
            # Let's find an element of order 2 in SL(2, Z_p).
            # Trace 0? If M^2 = I, then Tr(M)^2 - 2 det = 0 => Tr^2 = 2.
            # Or simply search.
            
            order_2_elems = []
            for m in elements:
                if m == (1, 0, 0, 1):
                    continue
                m2 = mat_mul(m, m)
                if m2 == (1, 0, 0, 1):
                    order_2_elems.append(m)
            
            if not order_2_elems:
                continue
                
            # Candidates for the pair {x, x^-1}
            # We can just pick random elements.
            
            best_n = float('inf')
            best_edges = None
            
            for _ in range(200): # 200 attempts
                if time.time() > deadline - 10:
                    break
                
                # Pick a random element x
                x = random.choice(elements)
                if x == (1, 0, 0, 1):
                    continue
                x_inv = mat_inv(x)
                if x == x_inv:
                    continue # Must be distinct from x_inv if we want 3 distinct generators?
                             # If x = x_inv, then S = {x, y, z}? No, S must be symmetric.
                             # If x is order 2, then x^-1 = x.
                             # So S = {x, y, y^-1}.
                             # So we need one order-2 element and one pair of inverses.
                
                # Pick a random order-2 element y
                y = random.choice(order_2_elems)
                
                # Ensure x, x_inv, y are distinct
                if x == y or x_inv == y:
                    continue
                
                S = [x, x_inv, y]
                
                # Build Cayley graph
                edges = []
                adj = [set() for _ in range(n)]
                is_valid = True
                
                for u_idx, u in enumerate(elements):
                    for s in S:
                        v = mat_mul(u, s)
                        v_idx = elem_to_idx.get(v)
                        if v_idx is None:
                            is_valid = False
                            break
                        if v_idx == u_idx:
                            is_valid = False
                            break
                        if v_idx in adj[u_idx]:
                            # Multi-edge
                            is_valid = False
                            break
                        adj[u_idx].add(v_idx)
                        if len(adj[u_idx]) > 3:
                            is_valid = False
                            break
                    if not is_valid:
                        break
                
                if not is_valid:
                    continue
                
                # Check connectivity (BFS from 0)
                visited = [False] * n
                q = deque([0])
                visited[0] = True
                count = 1
                while q:
                    u = q.popleft()
                    for v in adj[u]:
                        if not visited[v]:
                            visited[v] = True
                            count += 1
                            q.append(v)
                
                if count != n:
                    continue
                
                # Check girth
                # BFS from each vertex
                girth_ok = True
                max_depth = g // 2
                for start in range(n):
                    dist = [-1] * n
                    parent = [-1] * n
                    dist[start] = 0
                    q = deque([start])
                    while q:
                        u = q.popleft()
                        if dist[u] >= max_depth:
                            continue
                        for v in adj[u]:
                            if dist[v] == -1:
                                dist[v] = dist[u] + 1
                                parent[v] = u
                                q.append(v)
                            elif v != parent[u]:
                                cycle_len = dist[u] + dist[v] + 1
                                if cycle_len < g:
                                    girth_ok = False
                                    break
                    if not girth_ok:
                        break
                
                if girth_ok:
                    # Build edge list
                    edge_list = []
                    for u in range(n):
                        for v in adj[u]:
                            if u < v:
                                edge_list.append((u, v))
                    if n < best_n:
                        best_n = n
                        best_edges = edge_list
            
            if best_edges is not None:
                return best_n, best_edges
        
        return None

    # Try the new strategy
    result = try_cayley_sl2(k, g)
    if result:
        return result

    # Fallback to original strategy (Random Lift + 2-switch)
    # ... (code from original program) ...
    
    if k == 3 and g == 16:
        random.seed(42)
        base_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
            (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
            (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
        ]
        n_base = 10
        m_base = 15

        def build_lift(n_lift):
            voltages = [random.randint(0, n_lift - 1) for _ in range(m_base)]
            edges = []
            for i, (u, v) in enumerate(base_edges):
                vol = voltages[i]
                for l in range(n_lift):
                    u_lift = u * n_lift + l
                    v_lift = v * n_lift + (l + vol) % n_lift
                    edges.append((u_lift, v_lift))
            return n_base * n_lift, edges

        def is_simple_and_regular(edges, n):
            if len(edges) != n * 3 // 2:
                return False
            adj = [set() for _ in range(n)]
            for u, v in edges:
                if u == v:
                    return False
                if v in adj[u]:
                    return False
                adj[u].add(v)
                adj[v].add(u)
            for i in range(n):
                if len(adj[i]) != 3:
                    return False
            return True

        def check_girth_fast(adj, n, required_girth):
            max_depth = required_girth // 2
            for start in range(n):
                dist = [-1] * n
                parent = [-1] * n
                dist[start] = 0
                q = deque([start])
                while q:
                    u = q.popleft()
                    if dist[u] >= max_depth:
                        continue
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            q.append(v)
                        elif v != parent[u]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < required_girth:
                                return False
            return True

        best_n = float('inf')
        best_edges = None

        for L in range(94, 120, 2):
            n_total = 10 * L
            if time.time() > deadline - 15:
                break
            found = False
            for attempt in range(50):
                if time.time() > deadline - 15:
                    break
                n, edges = build_lift(L)
                if not is_simple_and_regular(edges, n):
                    continue
                adj = [set() for _ in range(n)]
                for u, v in edges:
                    adj[u].add(v)
                    adj[v].add(u)
                if check_girth_fast(adj, n, g):
                    best_n = n
                    best_edges = edges
                    found = True
                    break
            if found:
                break

        if best_edges is not None:
            return best_n, best_edges

    return luw_construct(k, g)
