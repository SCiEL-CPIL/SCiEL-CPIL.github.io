def construct(k, g):
    import time
    import random
    from collections import deque
    start_time = time.time()
    deadline = start_time + 108.0
    random.seed(42)

    # Fast paths for small known cages
    if k == 3:
        if g <= 3:
            return 3, [(0,1),(1,2),(0,2)]
        elif g <= 4:
            return 6, [(0,3),(0,4),(0,5),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5)]
        elif g <= 5:
            edges = []
            for i in range(5):
                edges.append((i, (i+1)%5))
                edges.append((i, 5+i))
                edges.append((5+i, 5+(i+2)%5))
            return 10, edges
        elif g <= 6:
            edges = []
            lines = [[0,1,2],[0,3,4],[0,5,6],[1,3,5],[1,4,6],[2,3,6],[2,4,5]]
            for i, line in enumerate(lines):
                for pt in line:
                    edges.append((i, 7+pt))
            return 14, edges
    elif k == 4:
        if g <= 3:
            return 4, [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
        elif g <= 4:
            return 8, [(i,j) for i in range(4) for j in range(4,8)]
    elif k == 5:
        if g <= 3:
            return 6, [(i,j) for i in range(6) for j in range(i+1,6)]
        elif g <= 4:
            return 10, [(i,j) for i in range(5) for j in range(5,10)]

    def find_primes(min_p, max_p):
        """Find primes in [min_p, max_p] that are suitable for GF(p) with k dividing p-1."""
        def is_prime(n):
            if n < 2: return False
            if n < 4: return True
            if n % 2 == 0 or n % 3 == 0: return False
            i = 5
            while i * i <= n:
                if n % i == 0 or n % (i+2) == 0:
                    return False
                i += 6
            return True
        primes = []
        for p in range(max(min_p, 2), min(max_p, 10000)+1):
            if is_prime(p) and (p - 1) % k == 0:
                primes.append(p)
        return primes

    def find_primitive_root(p):
        """Find a primitive root modulo prime p."""
        def is_prime(n):
            if n < 2: return False
            if n < 4: return True
            if n % 2 == 0 or n % 3 == 0: return False
            i = 5
            while i * i <= n:
                if n % i == 0 or n % (i+2) == 0:
                    return False
                i += 6
            return True
        # Factor p-1
        phi = p - 1
        factors = []
        tmp = phi
        d = 2
        while d * d <= tmp:
            if tmp % d == 0:
                factors.append(d)
                while tmp % d == 0:
                    tmp //= d
            d += 1
        if tmp > 1:
            factors.append(tmp)
        # Find primitive root
        for g in range(2, p):
            ok = True
            for f in factors:
                if pow(g, phi // f, p) == 1:
                    ok = False
                    break
            if ok:
                return g
        return 2  # fallback

    def build_tanner(p, d, seed_val):
        """
        Build a Tanner-like graph over GF(p) with dimension d.
        Vertices: (side, x) where side in {0,1}, x in GF(p)^d
        Edges: (0, x) -- (1, y) where y = A*x + b for random invertible A
        and y[0] = x[0] (fixing first coordinate to ensure k-regularity)
        
        Actually, let's use a different approach:
        Vertices: (side, x) where x in GF(p)^{d-1}
        For each x, connect (0,x) to (1, y) where y[0] = x[0] and y[1:] = A*x[1:] + b
        This gives degree p for each vertex if we have p choices for y[0].
        
        Wait, that's not right either. Let me think more carefully.
        
        Tanner graph approach:
        - Left vertices: (x) for x in GF(p)^d  -> p^d vertices
        - Right vertices: (y) for y in GF(p)^d  -> p^d vertices  
        - Edge (x, y) iff y = M*x + c where M is a fixed invertible matrix
        But this gives degree p^d on each side, not k.
        
        Better: Use the standard k-regular bipartite construction.
        Left: GF(p)^{d-1}, Right: GF(p)^{d-1}
        (x, y) edge iff y = A*x + b where A is (d-1)x(d-1) invertible over GF(p)
        This gives degree p^{d-1} on each side.
        
        For degree k, we need p^{d-1} = k, so p must be a power such that p^{d-1} = k.
        That's very restrictive.
        
        Alternative: Use voltage graphs / Cayley graphs.
        Let G = Z_p^d. Define a k-regular Cayley graph with connection set S of size k
        (symmetric, no identity). Girth depends on S.
        
        Let's use a random Cayley graph approach:
        - Group: Z_p^d (elementary abelian)
        - Connection set S: k random elements, closed under negation
        - This gives a k-regular graph on p^d vertices
        - Girth can be checked and optimized by choosing S
        """
        rng = random.Random(seed_val)
        
        # Choose d such that p^d is reasonable
        # For k-regular, we need |S| = k, so p^d >= k+1
        # Let's pick d = 2 or 3 for small p
        
        # Actually, let's use a different structural family:
        # Random regular bipartite graph via configuration model, then check girth.
        # But that's not algebraic.
        
        # Let me use the "random Cayley graph on Z_p^d" approach.
        # Group elements: tuples of length d with entries in Z_p
        # Connection set: k elements (with inverses)
        
        # For k-regular Cayley: S has k elements, S = S^{-1}, 0 not in S
        # If k is even: pick k/2 random elements, add their negatives
        # If k is odd: this doesn't work for abelian groups (need k even for symmetric S without 0)
        
        # For odd k, use non-abelian or directed... Let's stick with even k for Cayley.
        # For odd k, use a different approach.
        
        if k % 2 == 1:
            # For odd k, use a bipartite double cover or other construction
            # Use random k-regular bipartite graph on n/2 + n/2 vertices
            # n must be even
            pass  # handled below
        
        # For even k: Cayley graph on Z_p^d
        # Need p^d vertices, k-regular
        # Girth: shortest cycle in Cayley graph
        
        # Try different (p, d) pairs
        best_result = None
        
        # Candidate (p, d) pairs: p^d should be not too large
        candidates = []
        for p in [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
            if (p - 1) % k != 0 and p != 2:
                continue
            for d in range(1, 6):
                n = p ** d
                if n >= k + 1 and n <= 20000:
                    candidates.append((p, d, n))
        
        # Sort by n (prefer smaller)
        candidates.sort(key=lambda x: x[2])
        
        for (p, d, n) in candidates[:30]:  # try up to 30 candidates
            if time.time() > deadline:
                break
            
            # Try multiple random connection sets
            for attempt in range(50):
                if time.time() > deadline:
                    break
                
                # Generate random symmetric connection set of size k
                all_elements = [(a1, a2, a3, a4, a5, a6)[:d] 
                                for a1 in range(p) for a2 in range(p) for a3 in range(p)
                                for a4 in range(p) for a5 in range(p) for a6 in range(p)]
                # Filter out zero
                all_nonzero = [e for e in all_elements if e != tuple([0]*d)]
                
                if len(all_nonzero) < k:
                    continue
                
                # Pick k/2 random elements (for even k)
                half_k = k // 2
                S_half = rng.sample(all_nonzero, min(half_k, len(all_nonzero)))
                
                # Build S: S_half + inverses
                S = list(S_half)
                for s in S_half:
                    inv = tuple((-s[i]) % p for i in range(d))
                    if inv not in S:
                        S.append(inv)
                
                # If we have more than k, trim; if less, add more
                while len(S) > k:
                    S.pop()
                while len(S) < k and len(all_nonzero) > len(S):
                    remaining = [e for e in all_nonzero if e not in S and tuple((-e[i])%p for i in range(d)) not in S]
                    if not remaining:
                        break
                    s = rng.choice(remaining)
                    S.append(s)
                    inv = tuple((-s[i]) % p for i in range(d))
                    if inv not in S:
                        S.append(inv)
                
                if len(S) != k:
                    continue
                
                # Build adjacency and check girth using BFS
                # Map group elements to indices
                elem_to_idx = {}
                idx = 0
                for a in range(p):
                    for b in range(p):
                        if d == 1:
                            elem_to_idx[(a,)] = idx
                            idx += 1
                        else:
                            for c in range(p):
                                if d == 2:
                                    elem_to_idx[(a,b)] = idx
                                    idx += 1
                                else:
                                    for e in range(p):
                                        if d == 3:
                                            elem_to_idx[(a,b,c)] = idx
                                            idx += 1
                                        else:
                                            for f in range(p):
                                                if d == 4:
                                                    elem_to_idx[(a,b,c,e)] = idx
                                                    idx += 1
                                                else:
                                                    for h in range(p):
                                                        elem_to_idx[(a,b,c,e,f)] = idx
                                                        idx += 1
                                                        if d == 6:
                                                            for i in range(p):
                                                                elem_to_idx[(a,b,c,e,f,h)] = idx
                                                                idx += 1
                # This is getting complicated. Let's use a simpler encoding.
                # Encode element as integer: x0 + x1*p + x2*p^2 + ...
                def enc(elem):
                    val = 0
                    mult = 1
                    for i in range(d):
                        val += elem[i] * mult
                        mult *= p
                    return val
                def dec(val):
                    elem = []
                    for i in range(d):
                        elem.append(val % p)
                        val //= p
                    return tuple(elem)
                
                S_enc = [enc(s) for s in S]
                
                # Check girth via BFS from each vertex (expensive for large n)
                # For n > 1000, use a faster check
                girth_ok = True
                if n <= 500:
                    adj = [[] for _ in range(n)]
                    for u in range(n):
                        eu = dec(u)
                        for s in S_enc:
                            es = dec(s)
                            # Compute u + s in group
                            prod = tuple((eu[i] + es[i]) % p for i in range(d))
                            v = enc(prod)
                            adj[u].append(v)
                    
                    # Check girth
                    for start in range(n):
                        dist = [-1] * n
                        parent = [-1] * n
                        dist[start] = 0
                        queue = deque([start])
                        while queue:
                            u = queue.popleft()
                            for v in adj[u]:
                                if dist[v] == -1:
                                    dist[v] = dist[u] + 1
                                    parent[v] = u
                                    queue.append(v)
                                elif v != parent[u]:
                                    cycle_len = dist[u] + dist[v] + 1
                                    if cycle_len < g:
                                        girth_ok = False
                                        break
                        if not girth_ok:
                            break
                    
                    if girth_ok:
                        # Build edge list
                        edges = []
                        edge_set = set()
                        for u in range(n):
                            eu = dec(u)
                            for s in S_enc:
                                es = dec(s)
                                prod = tuple((eu[i] + es[i]) % p for i in range(d))
                                v = enc(prod)
                                e = (min(u,v), max(u,v))
                                if e not in edge_set:
                                    edge_set.add(e)
                                    edges.append(e)
                        if best_result is None or n < best_result[0]:
                            best_result = (n, edges)
                        break  # found valid for this (p,d)
                else:
                    # For larger n, do a quick check with limited BFS depth
                    # Only check if girth >= g/2 approximately
                    pass
        
        return best_result

    def build_random_regular_bipartite(n_half, k, seed_val):
        """
        Build a random k-regular bipartite graph with n_half vertices on each side.
        Total vertices: 2 * n_half.
        Uses configuration model / random permutation approach.
        """
        rng = random.Random(seed_val)
        n = 2 * n_half
        
        # Left vertices: 0..n_half-1, Right vertices: n_half..2*n_half-1
        # Each left vertex has k stubs, each right vertex has k stubs
        # Randomly match stubs
        
        left_stubs = [(i, j) for i in range(n_half) for j in range(k)]
        right_stubs = [(i, j) for i in range(n_half) for j in range(k)]
        
        rng.shuffle(right_stubs)
        
        edges = []
        edge_set = set()
        for (l, _), (r, _) in zip(left_stubs, right_stubs):
            u = l
            v = n_half + r
            e = (min(u,v), max(u,v))
            if e not in edge_set:
                edge_set.add(e)
                edges.append(e)
        
        # Check for multi-edges (shouldn't happen with this construction since each pair is unique)
        # Actually, this can create multi-edges if the same (l,r) pair appears multiple times
        # But since left_stubs has unique l values per j and right_stubs has unique r values per j,
        # and we're doing a bijection, we can still get multi-edges if (l1,r1) = (l2,r2) for different stubs
        # Wait no: left_stubs has (l, j) for j in range(k), so each l appears k times
        # right_stubs has (r, j) for j in range(k), so each r appears k times
        # The zip pairs them up, so we could have (l, r) appearing multiple times
        
        # Filter out duplicates
        unique_edges = []
        seen = set()
        for e in edges:
            if e not in seen:
                seen.add(e)
                unique_edges.append(e)
        
        # Check degree
        deg = [0] * n
        for u, v in unique_edges:
            deg[u] += 1
            deg[v] += 1
        
        if any(d != k for d in deg):
            return None
        
        # Check girth
        adj = [[] for _ in range(n)]
        for u, v in unique_edges:
            adj[u].append(v)
            adj[v].append(u)
        
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return None
        
        return (n, unique_edges)

    # Strategy 1: Try Cayley graphs for even k
    best_n, best_edges = None, None
    
    if k % 2 == 0:
        result = build_tanner(0, 0, 42)
        if result is not None:
            best_n, best_edges = result
    
    # Strategy 2: Try random regular bipartite graphs
    # n_half should be large enough for girth g
    # Moore bound for bipartite: n >= 2 * (1 + k + k(k-1) + ... + k(k-1)^{(g-3)/2}) for odd g
    # For even g: n >= 2 * (1 + k + k(k-1) + ... + k(k-1)^{(g-4)/2}) * (k-1) + 2 * (1 + k + ... )
    # Simplify: try n_half from Moore bound estimate up
    
    if best_n is None or best_n > 200:
        # Estimate minimum n_half
        if g % 2 == 1:
            # Odd girth g: tree depth (g-1)/2 from center
            # n_half >= 1 + k + k(k-1) + ... + k(k-1)^{(g-3)/2}
            depth = (g - 3) // 2
        else:
            depth = (g - 4) // 2
        
        min_n_half = 1
        for i in range(1, depth + 1):
            if i == 1:
                min_n_half += k
            else:
                min_n_half += k * (k-1) ** (i-1)
        
        # Try a few values of n_half
        for n_half in [min_n_half, min_n_half + k, min_n_half + 2*k]:
            if time.time() > deadline:
                break
            for seed_val in range(100):
                if time.time() > deadline:
                    break
                result = build_random_regular_bipartite(n_half, k, seed_val + 1000)
                if result is not None:
                    n, edges = result
                    if best_n is None or n < best_n:
                        best_n, best_edges = n, edges
                    break
    
    # Fallback: L-U-W construction
    if best_n is None:
        def build_luw(d, q):
            if q == 4:
                table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
                add_table = [[x ^ y for y in range(4)] for x in range(4)]
                sub_table = [[x ^ y for y in range(4)] for x in range(4)]
                mul_table = table
            else:
                add_table = [[(x + y) % q for y in range(q)] for x in range(q)]
                sub_table = [[(x - y) % q for y in range(q)] for x in range(q)]
                mul_table = [[x * y % q for y in range(q)] for x in range(q)]

            rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
            for c in range(5, d + 1):
                r, c0 = (c - 5) % 4, c - (c - 5) % 4
                rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

            rule_list = [None] * (d + 1)
            for c in range(2, d + 1):
                rule_list[c] = rule[c]

            def neighbours(vertex):
                side, x = vertex
                if side == 0:
                    for first in range(q):
                        y = [None, first] + [0] * (d - 1)
                        for c in range(2, d + 1):
                            a, b = rule_list[c]
                            y[c] = add_table[x[c]][mul_table[y[a]][x[b]]]
                        yield 1, tuple(y)
                else:
                    for first in range(q):
                        y = [None, first] + [0] * (d - 1)
                        for c in range(2, d + 1):
                            a, b = rule_list[c]
                            y[c] = sub_table[x[c]][mul_table[x[a]][y[b]]]
                        yield 0, tuple(y)

            start = (0, (None,) + (0,) * d)
            index, order, edges = {start: 0}, [start], []
            edge_set = set()

            for vertex in order:
                for other in neighbours(vertex):
                    if other not in index:
                        index[other] = len(order)
                        order.append(other)
                    if vertex[0] == 0:
                        e = (index[vertex], index[other])
                        if e not in edge_set:
                            edge_set.add(e)
                            edges.append(e)
                if time.time() > deadline:
                    break

            return len(order), edges

        q = k
        if g % 2 == 1:
            start_d = g - 5
            if start_d < 3:
                start_d = 3
            if start_d % 2 == 0:
                start_d += 1
        else:
            start_d = g - 5
            if start_d < 2:
                start_d = 2
            if start_d % 2 == 1:
                start_d += 1

        for d in range(start_d, g + 2):
            if time.time() > deadline:
                break
            if g % 2 == 1 and d % 2 == 0:
                continue
            if g % 2 == 0 and d % 2 == 1:
                continue
            n, edges = build_luw(d, q)
            best_n, best_edges = n, edges
            break

    return best_n, best_edges
