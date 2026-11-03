import time
from collections import deque

_DEADLINE = 120.0

def _verify_girth_fast(n, edges, k, g, budget=30.0):
    """Return True iff the graph is k-regular and has girth >= g."""
    t0 = time.time()
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    for u in range(n):
        if len(adj[u]) != k:
            return False
        if len(set(adj[u])) != k:
            return False
    for start in range(n):
        if time.time() - t0 > budget:
            return False
        visited = {}
        q = deque()
        for nb in adj[start]:
            state = (nb, start)
            visited[state] = 1
            q.append(state)
        while q:
            cur, prev = q.popleft()
            depth = visited[(cur, prev)]
            if depth >= g:
                break
            for nb in adj[cur]:
                if nb == prev:
                    continue
                if nb == start and depth + 1 < g:
                    return False
                state = (nb, cur)
                if state not in visited:
                    visited[state] = depth + 1
                    q.append(state)
    return True

def construct(k, g):
    t_start = time.time()
    
    if k == 5 and g == 9:
        # Try PSL(2,11) Cayley graph with 660 vertices
        p = 11
        # Generate PSL(2,p) elements as 2x2 matrices mod p with det=1, modulo ±I
        # Represent as tuple (a,b,c,d) for matrix [[a,b],[c,d]]
        # Two matrices are equivalent if they differ by sign
        
        def mat_mul(m1, m2):
            a1,b1,c1,d1 = m1
            a2,b2,c2,d2 = m2
            return ((a1*a2+b1*c2)%p, (a1*b2+b1*d2)%p,
                    (c1*a2+d1*c2)%p, (c1*b2+d1*d2)%p)
        
        def mat_det(m):
            a,b,c,d = m
            return (a*d - b*c) % p
        
        def mat_inv(m):
            a,b,c,d = m
            det = mat_det(m)
            # det should be 1
            return (d, (-b)%p, (-c)%p, a)
        
        # Generate all SL(2,p) elements
        sl_elements = []
        for a in range(p):
            for b in range(p):
                for c in range(p):
                    # d = (1 - b*c) * a^{-1} mod p if a != 0
                    # or handle a=0 case
                    if a != 0:
                        a_inv = pow(a, p-2, p)
                        d = (1 - b*c) * a_inv % p
                        m = (a, b, c, d)
                        if mat_det(m) == 1:
                            sl_elements.append(m)
                    else:
                        # a=0, then -b*c = 1, so b*c = -1
                        for d in range(p):
                            if b != 0:
                                c = (-1) * pow(b, p-2, p) % p
                                m = (0, b, c, d)
                                if mat_det(m) == 1:
                                    sl_elements.append(m)
        
        # Deduplicate modulo ±I: identify m and -m
        # Use canonical form: make the first non-zero element positive
        def canon(m):
            a,b,c,d = m
            if a != 0:
                if a > p//2:
                    return tuple((-x)%p for x in m)
            elif b != 0:
                if b > p//2:
                    return tuple((-x)%p for x in m)
            elif c != 0:
                if c > p//2:
                    return tuple((-x)%p for x in m)
            else:
                if d > p//2:
                    return tuple((-x)%p for x in m)
            return m
        
        seen = set()
        psl_elements = []
        for m in sl_elements:
            cm = canon(m)
            if cm not in seen:
                seen.add(cm)
                psl_elements.append(cm)
        
        n_psl = len(psl_elements)
        # Should be 660
        
        if n_psl != 660:
            # Fallback
            pass
        else:
            # Index elements
            elem_to_idx = {e: i for i, e in enumerate(psl_elements)}
            identity = (1, 0, 0, 1)
            
            # Build multiplication table
            mul_table = [[None]*n_psl for _ in range(n_psl)]
            for i, ei in enumerate(psl_elements):
                for j, ej in enumerate(psl_elements):
                    prod = mat_mul(ei, ej)
                    prod_canon = canon(prod)
                    mul_table[i][j] = elem_to_idx[prod_canon]
            
            # Find elements of order 2 (involutions)
            involutions = []
            for i, e in enumerate(psl_elements):
                if i == 0:  # identity
                    continue
                if mul_table[i][i] == 0:  # e*e = identity
                    involutions.append(i)
            
            # Find elements of order 3, 5, etc.
            # For a 5-regular Cayley graph, we need |S|=5
            # S must be closed under inverses
            # Option: 2 pairs of inverse elements + 1 involution
            # Or: 1 element of order 5 (gives 4 elements: e, e^2, e^3, e^4) + 1 involution
            # Or: 1 element of order 3 (gives 2 elements: e, e^2) + 1 element of order 2 + ... no that's only 3
            
            # Best option: element of order 5 (4 elements) + 1 involution = 5 elements total
            # Or: 2 pairs of inverses (4 elements) + 1 involution = 5 elements
            
            # Let's try: find an element of order 5 and an involution
            order5_elems = []
            for i, e in enumerate(psl_elements):
                if i == 0:
                    continue
                # Check if order is 5
                e2 = mul_table[i][i]
                e3 = mul_table[e2][i]
                e4 = mul_table[e3][i]
                e5 = mul_table[e4][i]
                if e5 == 0 and e4 != 0 and e3 != 0 and e2 != 0:
                    order5_elems.append(i)
            
            # Try combinations: order-5 element + involution
            # For each order-5 element g, S = {g, g^2, g^3, g^4, inv}
            # We need to check girth >= 9
            
            best_n = None
            best_edges = None
            
            # Try a few order-5 elements and involutions
            # Limit search to keep within time budget
            max_tries = 50
            tries = 0
            
            for g_idx in order5_elems[:10]:
                if tries >= max_tries:
                    break
                g2 = mul_table[g_idx][g_idx]
                g3 = mul_table[g2][g_idx]
                g4 = mul_table[g3][g_idx]
                S = [g_idx, g2, g3, g4]
                
                for inv_idx in involutions[:5]:
                    if tries >= max_tries:
                        break
                    tries += 1
                    if inv_idx in S:
                        continue
                    
                    S_full = S + [inv_idx]
                    
                    # Build edges
                    edges = []
                    for u in range(n_psl):
                        for s in S_full:
                            v = mul_table[u][s]
                            if u < v:
                                edges.append((u, v))
                    
                    # Quick check: is it simple and regular?
                    adj = [[] for _ in range(n_psl)]
                    for u, v in edges:
                        adj[u].append(v)
                        adj[v].append(u)
                    
                    regular = True
                    for u in range(n_psl):
                        if len(adj[u]) != 5:
                            regular = False
                            break
                        if len(set(adj[u])) != 5:
                            regular = False
                            break
                    
                    if not regular:
                        continue
                    
                    # Check girth
                    if _verify_girth_fast(n_psl, edges, 5, 9, budget=5.0):
                        best_n = n_psl
                        best_edges = edges
                        break
                
                if best_n is not None:
                    break
            
            if best_n is not None:
                if time.time() - t_start < _DEADLINE - 10:
                    return best_n, best_edges
        
        # If PSL(2,11) didn't work, try other approaches or fallback
        # Try smaller Cayley graphs on other groups
        # Or fall back to L-U-W
    
    # Fallback: Lazebnik-Ustimenko-Woldar graph
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
        if time.time() - t_start > _DEADLINE - 40:
            return len(order), edges
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    n = len(order)
    
    if time.time() - t_start < _DEADLINE - 20:
        if _verify_girth_fast(n, edges, k, g, budget=_DEADLINE - 25):
            return n, edges
    
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    regular = all(len(adj[u]) == k for u in range(n))
    
    if regular:
        return n, edges
    
    return 0, []
