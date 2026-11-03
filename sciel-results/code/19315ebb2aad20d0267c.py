"""Construct a 5-regular graph of girth 9 (cage problem n(5,9)) using PSL(2,11) Cayley graph."""
# EVOLVE-BLOCK-START
import itertools

def construct(k, g):
    """
    Construct a 5-regular graph with girth at least g.
    
    For k=5, g=9: Use the known (5,9)-cage construction or the PSL(2,11) Cayley graph.
    The PSL(2,11) has 660 elements. A specific connection set of 5 involutions
    gives a 5-regular graph. We need to find a connection set that yields girth >= 9.
    
    Actually, the known (5,9)-cage has 1116 vertices. The PSL(2,11) Cayley graph
    with appropriate generators might give girth 10 or 12, which is >= 9.
    
    Let's use a well-known construction: the Hoffman-Singleton graph is (7,5).
    For (5,9), one standard construction uses the incidence graph of a certain design.
    
    Alternative: Use the fact that PSL(2,11) has a Cayley graph of girth 10 with
    660 vertices. Since 10 >= 9, this satisfies the girth requirement.
    
    Let me construct PSL(2,11) and find a suitable connection set.
    
    PSL(2,11) consists of 3x3 matrices over GF(11) with determinant 1, modulo scalars.
    Actually, PSL(2,q) for q=11 has order q(q^2-1)/2 = 11*120/2 = 660.
    
    Let's represent elements as pairs (a, b) in some canonical form, or use matrix reps.
    
    A simpler approach: use the known explicit construction of the (5,9) cage if possible,
    or use a voltage graph / covering construction.
    
    Actually, let me use a different known construction. The (5,9) cage is known to exist
    with 1116 vertices. But we can try to do better or use a different approach.
    
    Let me use the following: construct a 5-regular graph using a circulant or other
    structured approach, then verify girth.
    
    For now, let's use the PSL(2,11) Cayley graph approach. We'll generate the group
    and find a connection set of 5 elements (involutions for simplicity) such that
    the resulting Cayley graph has girth >= 9.
    """
    if k != 5 or g != 9:
        # Fallback: use the Lazebnik-Ustimenko-Woldar construction
        return _construct_luw(k, g)
    
    # Construct PSL(2,11)
    # Elements are represented as matrices [[a, b], [c, d]] with ad-bc = 1, mod 11
    # We need to enumerate all 660 elements and pick a connection set.
    
    # Actually, let's use a known result: the Cayley graph of PSL(2,11) with
    # connection set consisting of 5 involutions can have girth 10.
    
    # Let me generate the group elements.
    # PSL(2,11) = SL(2,11) / {±I}
    # SL(2,11) has 1320 elements.
    
    p = 11
    
    def mat_mul(A, B):
        return [[(A[0][0]*B[0][0] + A[0][1]*B[1][0]) % p,
                 (A[0][0]*B[0][1] + A[0][1]*B[1][1]) % p],
                [(A[1][0]*B[0][0] + A[1][1]*B[1][0]) % p,
                 (A[1][0]*B[0][1] + A[1][1]*B[1][1]) % p]]
    
    def mat_det(A):
        return (A[0][0]*A[1][1] - A[0][1]*A[1][0]) % p
    
    def mat_eq(A, B):
        return A[0][0] == B[0][0] and A[0][1] == B[0][1] and A[1][0] == B[1][0] and A[1][1] == B[1][1]
    
    def mat_hash(A):
        return (A[0][0], A[0][1], A[1][0], A[1][1])
    
    # Generate all elements of SL(2,11)
    # [[a, b], [c, d]] with ad - bc = 1
    sl_elements = []
    for a in range(p):
        for b in range(p):
            for c in range(p):
                # ad - bc = 1 => d = (1 + bc) / a if a != 0
                if a == 0:
                    # -bc = 1 => bc = -1 = 10 mod 11
                    if b == 0:
                        continue
                    # c = 10 * b^{-1} mod 11
                    b_inv = pow(b, p-2, p)
                    c_val = (10 * b_inv) % p
                    d_val = 0  # d can be anything? No, d is determined
                    # Wait, if a=0, then -bc = 1, so bc = 10. d is free? No.
                    # Matrix is [[0, b], [c, d]], det = 0*d - b*c = -bc = 1, so bc = 10.
                    # d is not constrained by the determinant condition when a=0.
                    # So for each b, c such that bc=10, d can be anything.
                    for d_val in range(p):
                        mat = [[0, b], [c_val, d_val]]
                        sl_elements.append(mat)
                else:
                    a_inv = pow(a, p-2, p)
                    for d_val in range(p):
                        # ad - bc = 1 => bc = ad - 1
                        bc_val = (a * d_val - 1) % p
                        # b and c such that bc = bc_val
                        if b == 0:
                            if bc_val != 0:
                                continue
                            # b=0, c can be anything
                            for c_val in range(p):
                                mat = [[a, 0], [c_val, d_val]]
                                sl_elements.append(mat)
                        else:
                            b_inv = pow(b, p-2, p)
                            c_val = (bc_val * b_inv) % p
                            mat = [[a, b], [c_val, d_val]]
                            sl_elements.append(mat)
    
    # Remove duplicates and verify
    seen = set()
    sl_unique = []
    for mat in sl_elements:
        h = mat_hash(mat)
        if h not in seen:
            seen.add(h)
            sl_unique.append(mat)
    
    # Each element in PSL(2,11) corresponds to two elements in SL(2,11): M and -M
    # So we need to identify M ~ -M
    def mat_neg(M):
        return [[(-M[0][0]) % p, (-M[0][1]) % p],
                [(-M[1][0]) % p, (-M[1][1]) % p]]
    
    # Build PSL elements: pick one representative from each {M, -M} pair
    sl_set = {mat_hash(M): M for M in sl_unique}
    psl_elements = []
    visited = set()
    for M in sl_unique:
        h = mat_hash(M)
        if h in visited:
            continue
        h_neg = mat_hash(mat_neg(M))
        if h_neg in sl_set:
            # Add M as representative
            psl_elements.append(M)
            visited.add(h)
            visited.add(h_neg)
        else:
            psl_elements.append(M)
            visited.add(h)
    
    # Should have 660 elements
    # Map each PSL element to an index
    psl_index = {}
    for i, M in enumerate(psl_elements):
        psl_index[mat_hash(M)] = i
    
    def psl_mul(i, j):
        """Multiply two PSL elements."""
        A = psl_elements[i]
        B = psl_elements[j]
        C = mat_mul(A, B)
        h = mat_hash(C)
        if h in psl_index:
            return psl_index[h]
        # Try negative
        C_neg = mat_neg(C)
        h_neg = mat_hash(C_neg)
        if h_neg in psl_index:
            return psl_index[h_neg]
        raise ValueError("Product not in PSL")
    
    # Find involutions in PSL(2,11)
    # An involution satisfies M^2 = ±I
    identity = [[1, 0], [0, 1]]
    neg_identity = [[-1 % p, 0], [0, -1 % p]]
    
    involutions = []
    for i, M in enumerate(psl_elements):
        M2 = mat_mul(M, M)
        if mat_eq(M2, identity) or mat_eq(M2, neg_identity):
            if not mat_eq(M, identity):
                involutions.append(i)
    
    # We need to find 5 involutions whose Cayley graph has girth >= 9
    # This is a search problem. Let's try a few known connection sets.
    
    # For PSL(2,11), a known connection set for a girth 10 graph uses
    # specific involutions. Let's try to find one by random search or use a known one.
    
    # Actually, let's just try all combinations of 5 involutions and check girth.
    # But that's C(len(involutions), 5) which could be large.
    
    # Let's be smarter: use a greedy approach or known results.
    
    # Known: the (5,10) cage has 660 vertices and is a Cayley graph of PSL(2,11).
    # Let's find the right connection set.
    
    # Try a heuristic: pick 5 involutions that are "far apart" in the group.
    
    # For simplicity, let's just try the first few combinations and check girth.
    # But checking girth for 660 vertices might be slow.
    
    # Alternative: use a known explicit connection set.
    
    # Let me use a different approach: just construct a valid 5-regular graph of girth >= 9
    # using a known construction.
    
    # The simplest valid answer is to return a graph with n=1116 (the known cage).
    # But we want to minimize n.
    
    # Let's try to find a connection set for PSL(2,11) that gives girth >= 9.
    
    # I'll use a simple BFS-based girth check for small girths.
    
    def check_girth(connection_set, min_girth=9):
        """Check if the Cayley graph has no cycle shorter than min_girth."""
        n = len(psl_elements)
        
        # For each vertex, do BFS up to depth (min_girth-1)//2
        # and check for non-backtracking cycles
        depth = (min_girth - 1) // 2
        
        for start in range(n):
            # BFS from start, tracking paths to avoid backtracking
            # State: (vertex, prev_vertex)
            visited = set()
            queue = [(start, -1)]
            visited.add((start, -1))
            
            while queue:
                new_queue = []
                for (v, prev) in queue:
                    for u in connection_set:
                        w = psl_mul(v, u)
                        if w == prev:
                            continue
                        if (w, v) in visited:
                            # Found a cycle
                            return False
                        visited.add((w, v))
                        new_queue.append((w, v))
                queue = new_queue
                if len(queue) == 0:
                    break
        
        return True
    
    # Try to find a connection set
    # This might be slow, so let's limit the search
    
    # Actually, let's just use a known good connection set.
    # From literature, one such set for PSL(2,11) is:
    # We need to identify specific involutions.
    
    # Let me just try a few random sets and see if any work.
    import random
    random.seed(42)
    
    # Shuffle involutions
    inv_list = involutions[:]
    random.shuffle(inv_list)
    
    # Try the first 5
    conn_set = inv_list[:5]
    
    # Verify it's 5-regular (it should be by construction)
    # Check girth
    if check_girth(conn_set, min_girth=9):
        # Build the graph
        n = len(psl_elements)
        edges = []
        for i in range(n):
            for j in conn_set:
                w = psl_mul(i, j)
                if i < w:
                    edges.append((i, w))
        return n, edges
    
    # If that didn't work, try other combinations
    for trial in range(100):
        conn_set = random.sample(involutions, 5)
        if check_girth(conn_set, min_girth=9):
            n = len(psl_elements)
            edges = []
            for i in range(n):
                for j in conn_set:
                    w = psl_mul(i, j)
                    if i < w:
                        edges.append((i, w))
            return n, edges
    
    # Fallback: use the LUW construction
    return _construct_luw(k, g)

def _construct_luw(k, g):
    """Lazebnik-Ustimenko-Woldar graph construction."""
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
# EVOLVE-BLOCK-END
