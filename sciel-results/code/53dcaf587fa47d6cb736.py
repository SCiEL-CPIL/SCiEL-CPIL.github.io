"""Construct a 5-regular graph of girth 9 (cage problem n(5,9))."""
# EVOLVE-BLOCK-START
def construct(k, g):
    import time
    import random
    import sys

    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    # For k=5, g=9: use PSL(2,13) Cayley graph with n=1092
    # PSL(2,13) has order q(q^2-1)/gcd(2,q-1) = 13*168/2 = 1092
    q = 13
    n = q * (q * q - 1) // 2  # 1092

    def mat_mult(a, b):
        """Multiply two 2x2 matrices over GF(q)."""
        return (
            (a[0][0] * b[0][0] + a[0][1] * b[1][0]) % q,
            (a[0][0] * b[0][1] + a[0][1] * b[1][1]) % q,
            (a[1][0] * b[0][0] + a[1][1] * b[1][0]) % q,
            (a[1][0] * b[0][1] + a[1][1] * b[1][1]) % q,
        )

    def mat_inv(a):
        """Inverse of a 2x2 matrix with det 1 over GF(q)."""
        return (a[3], (-a[1]) % q, (-a[2]) % q, a[0])

    def mat_equal(a, b):
        return a[0] == b[0] and a[1] == b[1] and a[2] == b[2] and a[3] == b[3]

    def mat_key(m):
        return (m[0], m[1], m[2], m[3])

    # Generate all elements of PSL(2, q)
    # Each element is a 2x2 matrix with det 1, identified up to scalar (only ±1 for q odd)
    # We can enumerate all (a,b,c) with ad-bc=1, d determined, and quotient by ±I
    elements = []
    element_set = set()
    for a in range(q):
        for b in range(q):
            for c in range(q):
                # Need ad - bc = 1, so d = (1 + bc) * a^{-1} mod q
                if a == 0:
                    # Then -bc = 1, so bc = -1 = q-1
                    # d is free? No: ad - bc = -bc = 1 => bc = -1
                    if c == 0:
                        continue
                    # bc = q-1
                    if b == 0:
                        continue
                    # d can be anything? No, ad - bc = 0*d - bc = -bc = 1
                    # So d is not determined. Wait, for a=0: -bc = 1 => bc = -1
                    # d is free? No, the matrix is [[0,b],[c,d]] with -bc=1
                    # So d is not constrained by the determinant. But we need to pick specific matrices.
                    # Actually in PSL(2,q), we consider all SL(2,q) matrices.
                    # For a=0: -bc=1, so b and c are determined up to scaling, d is free.
                    # Let's just iterate d too.
                    continue
                else:
                    # a is invertible
                    a_inv = pow(a, q - 2, q)  # a^(q-2) = a^{-1} mod q
                    d = (1 + b * c) * a_inv % q
                    m = (a, b, c, d)
                    # Check det = 1
                    if (a * d - b * c) % q != 1:
                        continue
                    key = mat_key(m)
                    neg_key = ((-a) % q, (-b) % q, (-c) % q, (-d) % q)
                    # We want each PSL element once: pick the one where (a,d) or (a,c) has canonical sign
                    # Simpler: just add both and deduplicate, or pick canonical rep
                    # Use: if a > 0 or (a == 0 and d > 0) or (a == 0 and d == 0 and c > 0): canonical
                    # Actually for PSL(2,q) with q odd, identify M ~ -M
                    # Pick canonical: first nonzero entry positive
                    entries = [m[0], m[1], m[2], m[3]]
                    neg_entries = [(-e) % q for e in entries]
                    # Compare lexicographically
                    if entries <= neg_entries:
                        if key not in element_set:
                            element_set.add(key)
                            elements.append(m)
                    else:
                        if neg_key not in element_set:
                            element_set.add(neg_key)
                            elements.append(tuple(neg_entries))
    # Also handle a=0 case properly
    for b in range(1, q):
        for c in range(1, q):
            if (b * c) % q == (q - 1):  # bc = -1
                for d in range(q):
                    m = (0, b, c, d)
                    key = mat_key(m)
                    neg_key = (0, (-b) % q, (-c) % q, (-d) % q)
                    entries = [0, b, c, d]
                    neg_entries = [0, (-b) % q, (-c) % q, (-d) % q]
                    if entries <= neg_entries:
                        if key not in element_set:
                            element_set.add(key)
                            elements.append(m)
                    else:
                        if neg_key not in element_set:
                            element_set.add(neg_key)
                            elements.append(tuple(neg_entries))

    # This enumeration might miss some. Let's use a simpler approach:
    # Just enumerate all SL(2,q) matrices and quotient by ±I
    elements = []
    element_set = set()
    for a in range(q):
        for b in range(q):
            for c in range(q):
                for d in range(q):
                    if (a * d - b * c) % q != 1:
                        continue
                    key = (a, b, c, d)
                    neg_key = ((-a) % q, (-b) % q, (-c) % q, (-d) % q)
                    if key <= neg_key:
                        if key not in element_set:
                            element_set.add(key)
                            elements.append((a, b, c, d))
                    else:
                        if neg_key not in element_set:
                            element_set.add(neg_key)
                            elements.append(neg_key)

    n_actual = len(elements)
    # n_actual should be 1092
    if n_actual != 1092:
        # Fallback to LUW construction
        # LUW(4) over GF(5)
        q_luw = 5
        d_luw = 4
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        def add(x, y):
            return (x + y) % q_luw
        def sub(x, y):
            return (x - y) % q_luw
        def mul(x, y):
            return x * y % q_luw

        def neighbours_luw(vertex):
            side, x = vertex
            for first in range(q_luw):
                y = [None, first] + [0] * (d_luw - 1)
                for c in range(2, d_luw + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start_luw = (0, (None,) + (0,) * d_luw)
        index_luw, order_luw, edges_luw = {start_luw: 0}, [start_luw], []
        for vertex in order_luw:
            for other in neighbours_luw(vertex):
                if other not in index_luw:
                    index_luw[other] = len(order_luw)
                    order_luw.append(other)
                if vertex[0] == 0:
                    edges_luw.append((index_luw[vertex], index_luw[other]))
        return len(order_luw), edges_luw

    # Build index for elements
    elem_index = {elem: i for i, elem in enumerate(elements)}
    identity = (1, 0, 0, 1)
    id_idx = elem_index[identity]

    # Build multiplication table
    # For each element g, compute g * h for all h
    # This is O(n^2) = 1092^2 ≈ 1.2M multiplications, feasible
    mult_table = [None] * n_actual
    for i, g in enumerate(elements):
        row = [None] * n_actual
        for j, h in enumerate(elements):
            prod = mat_mult(g, h)
            # Find index of prod
            row[j] = elem_index[prod]
        mult_table[i] = row

    # Find involutions (elements of order 2, i.e., g^2 = identity, g != identity)
    involutions = []
    for i, g in enumerate(elements):
        if i == id_idx:
            continue
        g2 = mat_mult(g, g)
        if mat_equal(g2, identity):
            involutions.append(i)

    # Find elements of order 3, 4, etc. for candidate generators
    # We need: 2 inverse pairs {a, a^{-1}}, {b, b^{-1}} and 1 involution c
    # a != b, a != c, etc., and no two generators are inverses of each other
    
    # Random search for generators
    # Candidate generator pool: elements that are not identity
    non_identity = [i for i in range(n_actual) if i != id_idx]
    
    random.seed(42)
    
    best_n = None
    best_edges = None
    
    def check_girth(n_verts, edges, target_girth):
        """Check girth >= target_girth using non-backtracking BFS."""
        adj = [[] for _ in range(n_verts)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        for start in range(n_verts):
            visited = set()
            queue = []
            for s0 in adj[start]:
                visited.add((start, s0))
                queue.append((start, s0, 1))
            
            qi = 0
            while qi < len(queue):
                prev, cur, length = queue[qi]
                qi += 1
                if length >= target_girth:
                    continue
                for nxt in adj[cur]:
                    if nxt == prev:
                        continue
                    if nxt == start:
                        if length + 1 < target_girth:
                            return False
                    else:
                        state = (cur, nxt)
                        if state not in visited:
                            visited.add(state)
                            queue.append((cur, nxt, length + 1))
        return True

    def build_edges_from_generators(gen_indices):
        """Build edges for Cayley graph Cay(G, S) where S is the set of generators (symmetric)."""
        edges = []
        seen_edges = set()
        for i, v in enumerate(elements):
            for gen_idx in gen_indices:
                # v * gen
                w_idx = mult_table[i][gen_idx]
                u, w = min(i, w_idx), max(i, w_idx)
                edge = (u, w)
                if edge not in seen_edges:
                    seen_edges.add(edge)
                    edges.append(edge)
        return edges

    # Try random combinations of generators
    # We need: pick 2 pairs and 1 involution
    # Sample from elements with various orders
    # Prefer elements of small order for better mixing? Or large order?
    # For girth, we want generators to be "far apart"
    
    attempts = 0
    max_attempts = 50000
    
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        
        # Pick an involution
        c = random.choice(involutions)
        
        # Pick two other elements a, b (not identity, not c, not c^{-1}=c)
        a = random.choice(non_identity)
        while a == c or a == id_idx:
            a = random.choice(non_identity)
        
        b = random.choice(non_identity)
        while b == c or b == a or b == id_idx:
            b = random.choice(non_identity)
        
        # Check a != a^{-1} (a is not an involution) - actually we allow it, but then {a,a^{-1}} has size 1
        # For 5-regular: we need |S| = 5
        # S = {a, a^{-1}, b, b^{-1}, c}
        # Need: a != a^{-1}, b != b^{-1}, a != b, a != b^{-1}, a^{-1} != b, a^{-1} != b^{-1}, c != a, c != a^{-1}, c != b, c != b^{-1}
        
        a_inv = elem_index[mat_inv(elements[a])]
        b_inv = elem_index[mat_inv(elements[b])]
        
        # Check distinctness
        gens = {a, a_inv, b, b_inv, c}
        if len(gens) != 5:
            continue
        
        # Build edges and check girth
        edges = build_edges_from_generators([a, a_inv, b, b_inv, c])
        
        if len(edges) != n_actual * 5 // 2:
            # Should have exactly n*5/2 = 2730 edges
            continue
        
        if check_girth(n_actual, edges, g):
            return n_actual, edges
        
        # If we're running out of time, save best effort
        if best_edges is None:
            best_edges = edges
    
    if best_edges is not None:
        return n_actual, best_edges
    
    # Fallback: LUW construction
    q_luw = 5
    d_luw = 4
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    def add(x, y):
        return (x + y) % q_luw
    def sub(x, y):
        return (x - y) % q_luw
    def mul(x, y):
        return x * y % q_luw

    def neighbours_luw(vertex):
        side, x = vertex
        for first in range(q_luw):
            y = [None, first] + [0] * (d_luw - 1)
            for c in range(2, d_luw + 1):
                a2, b2 = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a2], x[b2]))
                else:
                    y[c] = sub(x[c], mul(x[a2], y[b2]))
            yield 1 - side, tuple(y)

    start_luw = (0, (None,) + (0,) * d_luw)
    index_luw, order_luw, edges_luw = {start_luw: 0}, [start_luw], []
    for vertex in order_luw:
        for other in neighbours_luw(vertex):
            if other not in index_luw:
                index_luw[other] = len(order_luw)
                order_luw.append(other)
            if vertex[0] == 0:
                edges_luw.append((index_luw[vertex], index_luw[other]))
    return len(order_luw), edges_luw
# EVOLVE-BLOCK-END
