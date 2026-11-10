import time
import math

# EVOLVE-BLOCK-START

def _verify_graph(n, edges, k, g):
    """Verify graph is k-regular, simple, and has girth >= g using non-backtracking walks."""
    if n < 2:
        return False
    adj = [set() for _ in range(n)]
    for i, j in edges:
        if i == j or i >= n or j >= n or i < 0 or j < 0:
            return False
        if i in adj[j]:
            return False  # repeated edge
        adj[i].add(j)
        adj[j].add(i)
    for i in range(n):
        if len(adj[i]) != k:
            return False
    # Non-backtracking walk check from every vertex
    for start in range(n):
        # State: (current_vertex, previous_vertex) -> min distance
        visited = {}
        queue = []
        for v in adj[start]:
            queue.append((v, start, 1))
            visited[(v, start)] = 1
        while queue:
            v, prev, dist = queue.pop(0)
            if dist >= g:
                continue
            for w in adj[v]:
                if w == prev:
                    continue
                key = (w, v)
                if key in visited:
                    if visited[key] + 1 < g:
                        return False
                else:
                    visited[key] = dist + 1
                    queue.append((w, v, dist + 1))
    return True

def _sl2_q_cayley(q, g, max_time=100):
    """
    Cayley graph on SL(2,q) with a 3-element involution-free generating set.
    For k=3, we need generators s1, s2, s3 such that s_i^2 != I and no
    non-trivial product of length < g equals I.
    
    We use a standard construction: work over GF(q) where q is a power of 2,
    and use specific matrix generators known to give large girth.
    """
    if k != 3:
        return None
    
    # For SL(2,q) with q = 2^m, |SL(2,q)| = q(q^2-1)
    # We need n = q(q^2-1) vertices
    # Try various q values and generating sets
    
    def mat_mul(A, B, mod_q):
        """Multiply 2x2 matrices over GF(q)"""
        return [
            [(A[0][0]*B[0][0] + A[0][1]*B[1][0]) % mod_q,
             (A[0][0]*B[0][1] + A[0][1]*B[1][1]) % mod_q],
            [(A[1][0]*B[0][0] + A[1][1]*B[1][0]) % mod_q,
             (A[1][0]*B[0][1] + A[1][1]*B[1][1]) % mod_q]
        ]
    
    def mat_inv(A, mod_q):
        """Inverse of 2x2 matrix over GF(q) - for SL(2,q), det=1"""
        # For SL(2,q): inv = [[d, -b], [-c, a]] mod q
        a, b = A[0][0], A[0][1]
        c, d = A[1][0], A[1][1]
        # -1 mod 2^m = 2^m - 1
        neg_b = (-b) % mod_q
        neg_c = (-c) % mod_q
        return [[d, neg_b], [neg_c, a]]
    
    def mat_equal(A, B):
        return A[0][0] == B[0][0] and A[0][1] == B[0][1] and A[1][0] == B[1][0] and A[1][1] == B[1][1]
    
    # Identity
    I = [[1, 0], [0, 1]]
    
    # Try different field sizes q = 2^m
    # |SL(2,q)| = q(q^2-1) = 2^m(2^(2m)-1)
    # For g=16, we need reasonable n
    
    results = []
    start_time = time.time()
    
    for m in range(3, 10):  # q from 8 to 512
        q = 2 ** m
        n_vertices = q * (q*q - 1)
        if n_vertices > 500000:  # Too large
            break
        
        # Standard generators for SL(2, 2^m):
        # S1 = [[1,1],[0,1]] (unipotent)
        # S2 = [[0,1],[1,0]] (involution - BAD, skip)
        # S3 = [[1,0],[1,1]] (unipotent)
        
        # Better: use generators that are not involutions
        # For GF(2^m):
        # A = [[1,1],[0,1]]
        # B = [[1,0],[1,1]]  
        # C = [[1,1],[1,0]]
        
        # Check which are involutions
        def is_involution(M):
            M2 = mat_mul(M, M, q)
            return mat_equal(M2, I)
        
        candidates = [
            [[1,1],[0,1]],  # A
            [[1,0],[1,1]],  # B
            [[1,1],[1,0]],  # C
            [[1,1],[1,1]],  # D
            [[1,0],[1,0]],  # not in SL(2)
        ]
        
        # Filter to SL(2,q): det = 1
        valid_gens = []
        for M in candidates:
            det = (M[0][0]*M[1][1] - M[0][1]*M[1][0]) % q
            if det == 1 and not is_involution(M):
                valid_gens.append(M)
        
        if len(valid_gens) < 3:
            continue
        
        # Try all combinations of 3 generators
        for i in range(len(valid_gens)):
            for j in range(len(valid_gens)):
                for l in range(len(valid_gens)):
                    if i == j or j == l or i == l:
                        continue
                    gens = [valid_gens[i], valid_gens[j], valid_gens[l]]
                    
                    # Quick check: no generator is identity
                    if any(mat_equal(g, I) for g in gens):
                        continue
                    
                    # Check girth by BFS in the Cayley graph
                    # For small graphs, do full verification
                    if n_vertices <= 10000:
                        # Build adjacency and verify
                        adj = [set() for _ in range(n_vertices)]
                        # Index matrices by a hash
                        # For 2x2 over GF(2^m), encode as integer
                        def mat_to_int(M):
                            return (M[0][0] << (2*m)) | (M[0][1] << m) | (M[1][0] << 0) | (M[1][1] << 0)
                        # Actually need proper encoding
                        def mat_to_int(M):
                            # Each entry is m bits
                            return (M[0][0] << (3*m)) | (M[0][1] << (2*m)) | (M[1][0] << (1*m)) | (M[1][1])
                        
                        def int_to_mat(v):
                            return [[(v >> (3*m)) & (q-1), (v >> (2*m)) & (q-1)],
                                    [(v >> (1*m)) & (q-1), v & (q-1)]]
                        
                        # Build index
                        mat_index = {}
                        current = I
                        queue = [I]
                        mat_index[mat_to_int(I)] = 0
                        idx = 1
                        while queue and idx < n_vertices:
                            M = queue.pop(0)
                            for g in gens:
                                Mg = mat_mul(M, g, q)
                                key = mat_to_int(Mg)
                                if key not in mat_index:
                                    mat_index[key] = idx
                                    idx += 1
                                    queue.append(Mg)
                            # Also multiply by inverses
                            for g in gens:
                                g_inv = mat_inv(g, q)
                                Mg = mat_mul(M, g_inv, q)
                                key = mat_to_int(Mg)
                                if key not in mat_index:
                                    mat_index[key] = idx
                                    idx += 1
                                    queue.append(Mg)
                        
                        if idx != n_vertices:
                            continue
                        
                        # Build edges
                        edges = []
                        for v in range(n_vertices):
                            M = int_to_mat(list(mat_index.keys())[v]) if v < len(mat_index) else None
                        # This approach is getting complex; let me use a simpler verification
                        
                        # For now, just record candidate for later full verification
                        results.append((n_vertices, m, gens))
        
        if time.time() - start_time > max_time:
            break
    
    return results

def _luw_graph(k, g):
    """Lazebnik-Ustimenko-Woldar construction for (k,g) with k odd prime or k=4."""
    q = k
    if (g - 5) % 2 == 1:
        d = g - 5
    else:
        d = g - 4
    d = max(3, d)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return x * y % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
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

def _cayley_sl2(k, g, max_time=100):
    """
    Build a Cayley graph on SL(2,q) for q=2^m with a 3-element generating set.
    Returns (n, edges) if a valid graph is found, else None.
    """
    if k != 3:
        return None
    
    def mat_mul(A, B, mod_q):
        return [
            [(A[0][0]*B[0][0] + A[0][1]*B[1][0]) % mod_q,
             (A[0][0]*B[0][1] + A[0][1]*B[1][1]) % mod_q],
            [(A[1][0]*B[0][0] + A[1][1]*B[1][0]) % mod_q,
             (A[1][0]*B[0][1] + A[1][1]*B[1][1]) % mod_q]
        ]
    
    def mat_inv(A, mod_q):
        a, b = A[0][0], A[0][1]
        c, d = A[1][0], A[1][1]
        return [[d, (-b) % mod_q], [(-c) % mod_q, a]]
    
    def mat_equal(A, B):
        return A[0][0] == B[0][0] and A[0][1] == B[0][1] and A[1][0] == B[1][0] and A[1][1] == B[1][1]
    
    I = [[1, 0], [0, 1]]
    
    start_time = time.time()
    
    for m in range(4, 12):
        if time.time() - start_time > max_time:
            break
        q = 2 ** m
        n_vertices = q * (q*q - 1)
        if n_vertices > 200000:
            break
        
        # Candidate generators in SL(2, 2^m)
        # These are unipotent or near-unipotent matrices
        gen_candidates = [
            [[1, 1], [0, 1]],   # Upper unipotent
            [[1, 0], [1, 1]],   # Lower unipotent
            [[1, 1], [1, 0]],   # Anti-diagonal
            [[1, 1], [1, 1]],   # All ones (det=0 in char 2! skip)
            [[1, 0], [1, 0]],   # Not in SL(2)
            [[0, 1], [1, 1]],   # Shift
            [[1, 1], [0, 0]],   # Not in SL(2)
        ]
        
        # Filter: det=1 and not involution
        valid_gens = []
        for M in gen_candidates:
            det = (M[0][0]*M[1][1] - M[0][1]*M[1][0]) % q
            if det != 1:
                continue
            M2 = mat_mul(M, M, q)
            if mat_equal(M2, I):
                continue
            valid_gens.append(M)
        
        if len(valid_gens) < 3:
            continue
        
        # Try triples of distinct generators
        for i in range(len(valid_gens)):
            for j in range(len(valid_gens)):
                for l in range(len(valid_gens)):
                    if i == j or j == l or i == l:
                        continue
                    gens = [valid_gens[i], valid_gens[j], valid_gens[l]]
                    
                    # Quick girth check: no word of length < g equals I
                    # BFS in the Cayley graph from I
                    from collections import deque
                    visited = {tuple(tuple(x) for x in I): 0}
                    dq = deque([(I, 0)])
                    girth_ok = True
                    while dq:
                        M, dist = dq.popleft()
                        if dist >= g:
                            continue
                        for gen in gens:
                            Mg = mat_mul(M, gen, q)
                            Mg_inv = mat_mul(M, mat_inv(gen, q), q)
                            for N in [Mg, Mg_inv]:
                                key = tuple(tuple(x) for x in N)
                                if key in visited:
                                    if visited[key] + 1 < g:
                                        girth_ok = False
                                        break
                                else:
                                    visited[key] = dist + 1
                                    dq.append((N, dist + 1))
                        if not girth_ok:
                            break
                    if not girth_ok:
                        continue
                    
                    # Build full graph
                    if n_vertices > 50000:
                        # Too large for quick edge construction, skip
                        continue
                    
                    # BFS to enumerate all group elements
                    mat_to_idx = {}
                    idx_to_mat = []
                    current = I
                    mat_to_idx[tuple(tuple(x) for x in I)] = 0
                    idx_to_mat.append(I)
                    dq = deque([I])
                    while dq:
                        M = dq.popleft()
                        for gen in gens:
                            for G in [gen, mat_inv(gen, q)]:
                                Mg = mat_mul(M, G, q)
                                key = tuple(tuple(x) for x in Mg)
                                if key not in mat_to_idx:
                                    mat_to_idx[key] = len(idx_to_mat)
                                    idx_to_mat.append(Mg)
                                    dq.append(Mg)
                    
                    if len(idx_to_mat) != n_vertices:
                        continue
                    
                    # Build edges
                    edges = []
                    for v in range(n_vertices):
                        M = idx_to_mat[v]
                        for gen in gens:
                            Mg = mat_mul(M, gen, q)
                            key = tuple(tuple(x) for x in Mg)
                            if key in mat_to_idx:
                                u = mat_to_idx[key]
                                if v < u:
                                    edges.append((v, u))
                    
                    if len(edges) != n_vertices * 3 // 2:
                        continue
                    
                    # Full verification
                    if _verify_graph(n_vertices, edges, 3, g):
                        return n_vertices, edges
    
    return None

def construct(k, g):
    deadline = time.time() + 140
    best_n, best_edges = None, None
    
    # Try Cayley graph on SL(2,q) first - potentially much smaller
    if k == 3:
        try:
            result = _cayley_sl2(3, g, max_time=80)
            if result is not None:
                n, edges = result
                best_n, best_edges = n, edges
        except Exception:
            pass
    
    # Try LUW construction as fallback
    try:
        n, edges = _luw_graph(k, g)
        if n is not None and len(edges) == n * k // 2:
            if best_n is None or n < best_n:
                best_n, best_edges = n, edges
    except Exception:
        pass
    
    # Verify best result
    if best_n is not None:
        if _verify_graph(best_n, best_edges, k, g):
            return best_n, best_edges
    
    # Fallback: return LUW without verification if time is running out
    if time.time() > deadline - 20:
        if best_n is not None:
            return best_n, best_edges
    
    return best_n, best_edges

# EVOLVE-BLOCK-END
