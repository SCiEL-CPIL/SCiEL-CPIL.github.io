"""Combined program: greedy C4-free construction with local search using bitmasks."""
import time
import random

def _prime_power(p, max_n=50):
    """Find a prime power >= max_n. Return (q, is_prime) where q is the prime power."""
    # Check primes first
    for p in range(max_n, max_n + 200):
        if p < 2:
            continue
        is_p = True
        for d in range(2, int(p**0.5) + 1):
            if p % d == 0:
                is_p = False
                break
        if is_p:
            return p, True
    # Check prime powers
    for base in range(2, max_n + 20):
        for power in range(2, 8):
            q = base ** power
            if q >= max_n:
                return q, False
            if q > 1000:
                break
    return max_n, True

def _polarity_graph(q):
    """
    Construct a polarity graph on q+1 vertices (for prime q).
    Vertices are 0..q. Edge (i,j) exists if i*j ≡ 1 (mod q) for i,j != 0,
    or specific rules for vertex 0 (infinity).
    
    Actually, for a polarity graph on q+1 vertices:
    - Vertices: 0, 1, ..., q-1 (elements of GF(q)) plus vertex q (infinity)
    - Edge between i and j (i != j, both < q) if i*j ≡ 1 (mod q)
    - Edge between i (< q) and q if i == 0
    - No self-loops
    
    Wait, let me use the standard construction:
    For prime p, polarity graph has p+1 vertices.
    Vertices: {0, 1, ..., p-1} ∪ {∞}
    Edge (i, j) for i,j in {0,...,p-1}: i*j ≡ 1 (mod p)
    Edge (i, ∞): i = 0
    Edge (∞, ∞): none (simple graph)
    
    This gives a C4-free graph. Let me verify the degree.
    """
    n = q + 1
    adj = [0] * n
    for i in range(q):
        for j in range(q):
            if i == j:
                continue
            if (i * j) % q == 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    # Vertex q is infinity
    # Edge (0, q)
    adj[0] |= (1 << q)
    adj[q] |= (1 << 0)
    return adj, n

def construct():
    start_time = time.time()
    deadline = 145  # safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    def check_c4_free(adj, n):
        """Check C4-freeness: no two vertices share 2+ common neighbors."""
        for i in range(n):
            for j in range(i + 1, n):
                common = adj[i] & adj[j]
                if common.bit_count() >= 2:
                    return False
        return True

    def min_degree(adj, n):
        return min(adj[i].bit_count() for i in range(n))

    def greedy_construct(n, target):
        """
        Randomized greedy construction: build graph vertex by vertex,
        adding edges that don't create a C4, prioritizing low-degree vertices.
        """
        adj = [0] * n
        degs = [0] * n
        mask = (1 << n) - 1

        # Process vertices in random order
        order = list(range(n))
        random.shuffle(order)

        for u in order:
            # Find all non-neighbors of u
            non_neighbors = (~adj[u]) & mask
            non_neighbors &= ~(1 << u)
            
            nn_list = []
            tmp = non_neighbors
            while tmp:
                lsb = tmp & -tmp
                idx = lsb.bit_length() - 1
                nn_list.append(idx)
                tmp ^= lsb
            
            random.shuffle(nn_list)
            
            # Add edges to u that don't create C4
            for v in nn_list:
                if degs[u] >= target:
                    break
                # Check if adding (u,v) creates a C4
                common = adj[u] & adj[v]
                if common.bit_count() < 2:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    degs[u] += 1
                    degs[v] += 1

        return adj, degs

    def local_search(adj, n, target, max_iters=100):
        """
        Local search: try to increase minimum degree by adding edges
        that don't create C4s.
        """
        degs = [adj[i].bit_count() for i in range(n)]
        mask = (1 << n) - 1

        for _ in range(max_iters):
            if elapsed() > deadline - 2:
                break
            
            min_deg = min(degs)
            if min_deg >= target:
                break
            
            # Find vertices with lowest degree
            low_verts = [i for i in range(n) if degs[i] <= min_deg + 1]
            random.shuffle(low_verts)
            
            improved = False
            for u in low_verts:
                if degs[u] >= target:
                    continue
                
                non_neighbors = (~adj[u]) & mask
                non_neighbors &= ~(1 << u)
                
                nn_list = []
                tmp = non_neighbors
                while tmp:
                    lsb = tmp & -tmp
                    idx = lsb.bit_length() - 1
                    nn_list.append(idx)
                    tmp ^= lsb
                
                random.shuffle(nn_list)
                
                for v in nn_list:
                    common = adj[u] & adj[v]
                    if common.bit_count() < 2:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        degs[u] += 1
                        degs[v] += 1
                        improved = True
                        if degs[u] >= target:
                            break
                
                if not improved:
                    continue
            
            if not improved:
                break

        return adj, degs

    def try_construct(n, attempts=30):
        target = n - 39
        if target < 1:
            target = 1
        
        best_adj = None
        best_md = -1

        # Try polarity graph seed first
        q, is_prime = _prime_power(n - 1)
        if q >= n - 1 and q <= n + 10:
            pol_adj, pol_n = _polarity_graph(q)
            if pol_n == n:
                if check_c4_free(pol_adj, n):
                    md = min_degree(pol_adj, n)
                    if md > best_md:
                        best_md = md
                        best_adj = pol_adj[:]
                    if md < target:
                        pol_adj2, _ = local_search(pol_adj[:], n, target, max_iters=50)
                        if check_c4_free(pol_adj2, n):
                            md2 = min_degree(pol_adj2, n)
                            if md2 > best_md:
                                best_md = md2
                                best_adj = pol_adj2[:]
                            if md2 >= target:
                                edges = []
                                for i in range(n):
                                    for j in range(i + 1, n):
                                        if best_adj[i] & (1 << j):
                                            edges.append([i, j])
                                return n, edges

        # Randomized greedy construction attempts
        for attempt in range(attempts):
            if elapsed() > deadline - 2:
                break

            # 1. Greedy construction
            adj, degs = greedy_construct(n, target)

            # 2. Verify C4-free
            if not check_c4_free(adj, n):
                continue

            md = min(degs)
            if md > best_md:
                best_md = md
                best_adj = adj[:]

            # 3. If min degree is too low, try local search
            if md < target:
                adj2, degs2 = local_search(adj, n, target, max_iters=50)
                if check_c4_free(adj2, n):
                    md2 = min(degs2)
                    if md2 > best_md:
                        best_md = md2
                        best_adj = adj2[:]
                    if md2 >= target:
                        edges = []
                        for i in range(n):
                            for j in range(i + 1, n):
                                if best_adj[i] & (1 << j):
                                    edges.append([i, j])
                        return n, edges

            if md >= target:
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                return n, edges

        if best_adj is not None and best_md >= target:
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if best_adj[i] & (1 << j):
                        edges.append([i, j])
            return n, edges
            
        return None

    # Try from largest n down
    for n in [47, 46, 45, 44, 43, 42, 41, 40]:
        result = try_construct(n)
        if result is not None:
            return {'n': result[0], 'edges': result[1]}

    # Fallback: trivial graph
    return {'n': 39, 'edges': []}
