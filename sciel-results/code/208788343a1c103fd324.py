import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s kill

    def luw_fallback():
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

    if k == 5 and g <= 12:
        try:
            q = 13
            # Build PSL(2,13) elements: matrices [a b; c d] with det = 1, mod q
            # det = a*d - b*c ≡ 1 (mod q)
            # For each (a,b,c), d = (1 + b*c) * a^{-1} mod q if a ≠ 0
            # If a = 0, then -b*c ≡ 1, so c = -b^{-1}, and d is free
            psl_elements = []
            seen = set()
            for a in range(q):
                for b in range(q):
                    for c in range(q):
                        if a != 0:
                            d = (1 + b * c) * pow(a, -1, q) % q
                            mat = (a, b, c, d)
                            # In PSL, M ~ -M, so pick canonical representative
                            neg = (-a % q, -b % q, -c % q, -d % q)
                            canon = mat if mat < neg else neg
                            if canon not in seen:
                                seen.add(canon)
                                psl_elements.append(canon)
                        else:
                            # a = 0: need -b*c ≡ 1 mod q, so c = -b^{-1} mod q
                            if b == 0:
                                continue
                            c_val = (-1 * pow(b, -1, q)) % q
                            if c_val != c:
                                continue
                            # d is free
                            for d in range(q):
                                mat = (a, b, c, d)
                                neg = (-a % q, -b % q, -c % q, -d % q)
                                canon = mat if mat < neg else neg
                                if canon not in seen:
                                    seen.add(canon)
                                    psl_elements.append(canon)

            n_vertices = len(psl_elements)
            idx_map = {e: i for i, e in enumerate(psl_elements)}
            identity = (1, 0, 0, 1)

            # Precompute multiplication table for speed
            # mat_mult(u, s) = u * s
            def mat_mult(u, s):
                ua, ub, uc, ud = u
                sa, sb, sc, sd = s
                return ((ua * sa + ub * sc) % q,
                        (ua * sb + ub * sd) % q,
                        (uc * sa + ud * sc) % q,
                        (uc * sb + ud * sd) % q)

            # Precompute all pairwise products for the full group
            # This is 169*169 = 28561 entries, manageable
            # Actually n_vertices = 1092 for PSL(2,13)... wait let me check
            # |PSL(2,13)| = 13*(13^2-1)/2 = 13*168/2 = 1092
            # That's too many for a full table. Let's use a dict cache instead.
            
            # Find involutions: elements of order 2, i.e., M^2 = I
            # For M = [a b; c d], M^2 = I means trace = 0 (in PSL)
            # Actually in PSL(2,q), an element has order 2 iff trace ≡ 0 (mod q)
            # and it's not the identity
            involutions = []
            for e in psl_elements:
                a, b, c, d = e
                if (a + d) % q == 0 and (b, c) != (0, 0):
                    # Verify it's actually order 2: M^2 = I
                    m2 = mat_mult(e, e)
                    if m2 == identity:
                        involutions.append(e)

            # Precompute neighbors for each involution pair
            # For girth >= 9, we need no cycles of length 4, 6, 8
            # In a Cayley graph with generating set S (all involutions), 
            # a cycle of length 2l corresponds to s1 s2 ... sl s1 s2 ... sl = I
            # More precisely, for girth >= 9 (odd), we need no cycles of length 4, 6, 8
            # Cycle of length 4: s1 s2 s1 s2 = I for s1 ≠ s2, or s1 s2 s1 s2 = I
            # Actually for involutions, s1 s2 s1 s2 = (s1 s2)^2, so we need (s1 s2)^2 ≠ I
            # i.e., s1 s2 ≠ I and s1 s2 ≠ s1 s2 (trivial), so (s1 s2)^2 ≠ I means s1 s2 has order > 2
            # Cycle of length 6: s1 s2 s3 s1 s2 s3 = I, i.e., (s1 s2 s3)^2 = I, so s1 s2 s3 has order 2
            # Cycle of length 8: s1 s2 s3 s4 s1 s2 s3 s4 = I, i.e., (s1 s2 s3 s4)^2 = I

            # Better approach: just do BFS from each vertex up to depth 4
            # and check for conflicts

            best_n = n_vertices
            best_edges = None

            # Precompute: for each element and each involution, the product
            # This is n_vertices * len(involutions) products
            # n_vertices ~ 1092, len(involutions) ~ 78 (for PSL(2,13))
            # That's about 85,000 products, each O(1)
            
            # Let's precompute a lookup: prod_table[u][i] = idx_map[mat_mult(u, involutions[i])]
            # But we're choosing subsets of 5 from involutions, so precompute for all involutions
            n_inv = len(involutions)
            # prod_table[u][j] = index of mat_mult(psl_elements[u], involutions[j])
            prod_table = np.empty((n_vertices, n_inv), dtype=np.int32)
            for u_idx, u in enumerate(psl_elements):
                for j, s in enumerate(involutions):
                    prod_table[u_idx, j] = idx_map[mat_mult(u, s)]

            # For girth checking, we need to verify no cycle of length < 9
            # Use BFS from each vertex, tracking distance
            # A cycle exists if during BFS we find an edge (u,v) where dist[u] + dist[v] + 1 < g
            # Actually for girth >= g, we need dist[u] + dist[v] + 1 >= g for all edges
            # where u and v are adjacent and both are reached in the BFS tree
            # More carefully: if we do BFS from start, and for an edge (u,v):
            #   if dist[v] == -1: set dist[v] = dist[u] + 1
            #   else if dist[u] + 1 + dist[v] < g: cycle too short
            # But this only catches cycles where the "back edge" creates a cycle
            # We need to check: for every edge (u,v), the cycle length is dist[u] + dist[v] + 1
            # if u and v are in the BFS tree from the same root, the cycle length is 
            # dist[u] + dist[v] + 1 - 2*dist[lca(u,v)]... this is getting complex.
            # 
            # Simpler: just check that during BFS, when we encounter an already-visited node v
            # from u, if dist[u] + 1 <= dist[v] and dist[u] + 1 + dist[v] < g, fail.
            # Actually the standard girth check via BFS: for each edge (u,v) where 
            # dist[v] != -1 and dist[v] != dist[u] + 1 (i.e., it's a back/cross edge),
            # the cycle length is dist[u] + dist[v] + 1. We need this >= g.
            # But we also need dist[u] + 1 + dist[v] >= g for the case where 
            # v was already visited at a different level.
            # 
            # The cleanest check: for each edge (u,v), if both u and v are reached,
            # the shortest cycle through this edge has length dist[u] + dist[v] + 1
            # (where dist is from the BFS root). We need this >= g.
            # But this overcounts because the actual cycle might be shorter via LCA.
            # 
            # Correct approach for girth >= g: run BFS from each vertex up to depth (g-1)//2
            # and check that no vertex is reached twice at depth <= (g-2)//2 from the same root
            # via different paths. Actually, the standard method:
            # For each start vertex, do BFS. When we find an edge (u,v) where v is already
            # visited, the cycle length is dist[u] + dist[v] + 1. If this is < g, fail.
            # This works because any cycle of length L < g will be detected by some 
            # back edge in the BFS tree.

            # Let's use a faster check: for girth >= 9, we need no cycles of length 4, 6, 8.
            # For a Cayley graph with involutions, we can check:
            # - No 4-cycles: for all s1, s2 in S, s1 ≠ s2: (s1*s2)^2 ≠ I
            # - No 6-cycles: for all s1, s2, s3 in S: (s1*s2*s3)^2 ≠ I (with s1*s2*s3 ≠ I)
            # - No 8-cycles: for all s1,s2,s3,s4 in S: (s1*s2*s3*s4)^2 ≠ I
            # But this is necessary but not sufficient in general... 
            # Actually for Cayley graphs, these conditions ARE sufficient for girth >= 9
            # when the generators are involutions. Let me verify...
            # 
            # A cycle of length 2m in a Cayley graph corresponds to s1 s2 ... sm s1 s2 ... sm = I
            # which is (s1 s2 ... sm)^2 = I, so s1 s2 ... sm has order 2 (or 1).
            # For m=2: (s1 s2)^2 = I → s1 s2 has order 1 or 2. Order 1 means s1=s2 (not allowed since distinct). Order 2 means s1 s2 is an involution ≠ I.
            # For m=3: (s1 s2 s3)^2 = I → s1 s2 s3 has order 2.
            # For m=4: (s1 s2 s3 s4)^2 = I → s1 s2 s3 s4 has order 2.
            # 
            # So the conditions are:
            # m=2: s1*s2 is not an involution (for s1≠s2 in S)
            # m=3: s1*s2*s3 is not an involution (for distinct s1,s2,s3 in S)  
            # m=4: s1*s2*s3*s4 is not an involution (for distinct s1,s2,s3,s4 in S)
            # 
            # Wait, I need to be more careful. The product s1 s2 ... sm should have order exactly 2,
            # not 1. Order 1 means s1...sm = I, which would be a shorter cycle.
            # 
            # Let me just use BFS for correctness. It's O(n * k * g/2) per candidate set.
            # n=1092, k=5, g/2=4, so ~22k operations per check. With 500 trials, ~11M ops. Fine.

            # Precompute adjacency for a given S
            # adj[u] = list of neighbors of u in the Cayley graph with generators S

            # For speed, let's precompute the full multiplication result as indices
            # mult_idx[u_idx, j] = idx of (psl_elements[u_idx] * involutions[j])
            # We already have prod_table for this.

            # Girth check using BFS
            def check_girth(S_indices):
                """S_indices: list of 5 indices into involutions"""
                # Build adjacency
                # adj[u] = [prod_table[u, s] for s in S_indices]
                # Use the precomputed prod_table
                
                # BFS from each vertex
                for start in range(n_vertices):
                    if time.time() > deadline:
                        return False
                    dist = np.full(n_vertices, -1, dtype=np.int32)
                    dist[start] = 0
                    queue = [start]
                    head = 0
                    while head < len(queue):
                        u = queue[head]
                        head += 1
                        du = dist[u]
                        if du >= 4:  # We only need to check up to depth 4 for girth 9
                            continue
                        for si in S_indices:
                            v = prod_table[u, si]
                            if dist[v] == -1:
                                dist[v] = du + 1
                                queue.append(v)
                            else:
                                # Check if this creates a cycle shorter than g
                                # Cycle length = du + 1 + dist[v]
                                if du + 1 + dist[v] < g:
                                    return False
                return True

            # Search for good S
            # Random search with more iterations
            random.seed(42)
            for trial in range(2000):
                if time.time() > deadline - 5:
                    break
                S_indices = random.sample(range(n_inv), 5)
                
                # Quick check: no 4-cycles
                # For all pairs (i,j) in S_indices, i<j: (s_i * s_j)^2 ≠ I
                # i.e., s_i * s_j is not an involution
                ok = True
                for i in range(5):
                    for j in range(i+1, 5):
                        si = involutions[S_indices[i]]
                        sj = involutions[S_indices[j]]
                        prod = mat_mult(si, sj)
                        # Check if prod is an involution: prod^2 = I
                        prod2 = mat_mult(prod, prod)
                        if prod2 == identity:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue
                
                # Check girth
                if check_girth(S_indices):
                    # Found a valid set! Build edges
                    edges = []
                    for u in range(n_vertices):
                        for si in S_indices:
                            v = prod_table[u, si]
                            if u < v:  # avoid duplicate edges
                                edges.append((u, v))
                    best_n = n_vertices
                    best_edges = edges
                    break

            if best_edges is not None:
                return best_n, best_edges

        except Exception:
            pass

    return luw_fallback()
