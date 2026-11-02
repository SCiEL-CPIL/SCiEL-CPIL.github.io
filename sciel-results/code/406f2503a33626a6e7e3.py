import numpy as np
import random
import time
from collections import deque

def construct(k, g):
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k: an explicit q-regular bipartite graph
    # whose girth is at least d + 5 for odd d. Points and lines are vectors over GF(q) (q prime,
    # or q = 4); the connected component of the zero point is returned.
    def seed_construct():
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

    # Proper girth verification using BFS from each vertex
    def get_girth(adj, n, min_girth=10):
        """Returns the girth of the graph. If girth >= min_girth, returns min_girth."""
        for i in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[i] = 0
            queue = deque([i])
            
            while queue:
                u = queue.popleft()
                if dist[u] >= min_girth // 2:
                    continue
                for v in adj[u]:
                    if v == parent[u]:
                        continue
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif dist[v] <= dist[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            return cycle_len
        return min_girth

    def verify_graph(n, edges):
        """Verify the graph is valid: simple, k-regular, girth >= g"""
        if n < 1:
            return False
        adj = [set() for _ in range(n)]
        edge_set = set()
        for i, j in edges:
            if i < 0 or i >= n or j < 0 or j >= n:
                return False
            if i == j:
                return False
            key = (min(i, j), max(i, j))
            if key in edge_set:
                return False
            edge_set.add(key)
            adj[i].add(j)
            adj[j].add(i)
        
        for i in range(n):
            if len(adj[i]) != k:
                return False
        
        girth = get_girth(adj, n, g)
        if girth < g:
            return False
        
        return True

    def random_4reg(n):
        """Generate random 4-regular graph using configuration model with retries."""
        for _ in range(100):
            adj = [set() for _ in range(n)]
            # Use circulant as base
            for i in range(n):
                for j in [1, 2, n-1, n-2]:
                    v = (i + j) % n
                    if i != v and v not in adj[i]:
                        adj[i].add(v)
                        adj[v].add(i)
            
            valid = True
            for i in range(n):
                if len(adj[i]) != 4:
                    valid = False
                    break
            
            if valid:
                return adj
        return None

    def swap_edges(adj, n):
        """Perform a 2-switch to maintain 4-regularity."""
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        
        if len(edges) < 2:
            return False
            
        for _ in range(10):
            e1 = random.choice(edges)
            e2 = random.choice(edges)
            if e1 == e2:
                continue
                
            u, v = e1
            x, y = e2
            
            if len({u, v, x, y}) < 4:
                continue
            
            if x not in adj[u] and y not in adj[v]:
                adj[u].discard(v)
                adj[v].discard(u)
                adj[x].discard(y)
                adj[y].discard(x)
                adj[u].add(x)
                adj[x].add(u)
                adj[v].add(y)
                adj[y].add(v)
                return True
            
            if y not in adj[u] and x not in adj[v]:
                adj[u].discard(v)
                adj[v].discard(u)
                adj[x].discard(y)
                adj[y].discard(x)
                adj[u].add(y)
                adj[y].add(u)
                adj[v].add(x)
                adj[x].add(v)
                return True
        
        return False

    def count_short_cycles(adj, n, max_len=8):
        """Count cycles of length < 10 using BFS."""
        count = 0
        for i in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[i] = 0
            queue = deque([i])
            while queue:
                u = queue.popleft()
                if dist[u] >= max_len // 2:
                    continue
                for v in adj[u]:
                    if v == parent[u]:
                        continue
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif dist[v] <= dist[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < 10:
                            count += 1
        return count

    # SL(2, Z_p) Cayley graph construction
    def cayley_sl2(p):
        """Construct a 4-regular Cayley graph on SL(2, Z_p) with 4 generators."""
        # SL(2, Z_p) has p^3 - p elements
        # We need to enumerate all matrices in SL(2, Z_p)
        
        # Generate all 2x2 matrices with det=1 mod p
        matrices = []
        for a in range(p):
            for b in range(p):
                for c in range(p):
                    # d = (1 - b*c) * inv(a) mod p if a != 0, else need a=0 case
                    if a == 0:
                        # Then -b*c = 1 mod p, so b != 0 and c != 0
                        if b == 0 or c == 0:
                            continue
                        bc = (b * c) % p
                        if bc != (p - 1) % p:  # -1 mod p
                            continue
                        d = 0
                    else:
                        inv_a = pow(a, p-2, p)  # Fermat's little theorem
                        d = (1 - b * c) % p * inv_a % p
                    matrices.append(((a, b), (c, d)))
        
        # Map matrices to indices
        mat_to_idx = {}
        for i, mat in enumerate(matrices):
            mat_to_idx[mat] = i
        
        n = len(matrices)
        
        # Define 4 generators (must be involutions or come in inverse pairs for undirected graph)
        # Use standard generators for SL(2, Z_p)
        # S = [[0, -1], [1, 0]] (order 4)
        # T = [[1, 1], [0, 1]] (order p)
        
        # For a 4-regular graph, we can use 2 generators and their inverses
        # Or 4 involutions
        # Let's use S and T with their inverses
        
        def mat_mul(m1, m2):
            a1, b1 = m1[0]
            c1, d1 = m1[1]
            a2, b2 = m2[0]
            c2, d2 = m2[1]
            return (((a1*a2 + b1*c2) % p, (a1*b2 + b1*d2) % p),
                    ((c1*a2 + d1*c2) % p, (c1*b2 + d1*d2) % p))
        
        def mat_inv(m):
            a, b = m[0]
            c, d = m[1]
            # For SL(2), inverse is [[d, -b], [-c, a]]
            return ((d % p, (-b) % p), ((-c) % p, a % p))
        
        S = ((0, (p-1) % p), (1, 0))  # [[0, -1], [1, 0]]
        T = ((1, 1), (0, 1))  # [[1, 1], [0, 1]]
        S_inv = mat_inv(S)
        T_inv = mat_inv(T)
        
        # Check if S and S_inv are the same (order 2) or different
        # S^2 = -I, S^4 = I, so S has order 4
        # T has order p
        
        # For undirected 4-regular: use S, S_inv, T, T_inv as generators
        # But if S = S_inv (order 2), we'd only have 3 distinct generators
        # Let's check: S = [[0,-1],[1,0]], S_inv = [[0,1],[-1,0]] = -S
        # So S != S_inv unless p=2
        
        # Build adjacency
        adj = [set() for _ in range(n)]
        
        for i, mat in enumerate(matrices):
            for gen in [S, S_inv, T, T_inv]:
                prod = mat_mul(mat, gen)
                if prod in mat_to_idx:
                    j = mat_to_idx[prod]
                    if j != i:  # No loops
                        adj[i].add(j)
                        adj[j].add(i)
        
        # Verify 4-regular
        for i in range(n):
            if len(adj[i]) != 4:
                # This might happen if some generators give same neighbor
                # For now, return None if not exactly 4-regular
                return None, None
        
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        
        return n, edges

    def cayley_sl2_search(p_list):
        """Try several primes for SL(2, Z_p) Cayley graphs."""
        best_n, best_edges = None, None
        deadline = time.time() + 110
        
        for p in p_list:
            if time.time() > deadline:
                break
            
            n, edges = cayley_sl2(p)
            if n is None:
                continue
            
            # Check girth
            adj = [set() for _ in range(n)]
            for i, j in edges:
                adj[i].add(j)
                adj[j].add(i)
            
            girth = get_girth(adj, n, g)
            if girth >= g:
                if best_n is None or n < best_n:
                    best_n, best_edges = n, edges
        
        return best_n, best_edges

    # Start with the seed construction
    best_n, best_edges = seed_construct()
    
    # Verify the seed construction
    if not verify_graph(best_n, best_edges):
        # Seed failed, we need a valid fallback
        # Try SL(2, Z_p) constructions
        p_candidates = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        cay_n, cay_edges = cayley_sl2_search(p_candidates)
        if cay_n is not None:
            best_n, best_edges = cay_n, cay_edges
        else:
            # Last resort: return seed even if invalid (shouldn't happen)
            pass
    
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Try to improve by searching for smaller graphs using Cayley graphs
    # SL(2, Z_p) has p^3 - p vertices
    # For p=5: 120, p=7: 336, p=11: 1320
    # We want n < best_n (which is 320 from seed)
    # So p=5 gives 120 vertices - too small likely for girth 10
    # p=7 gives 336 - close to 320
    
    # Try Cayley graphs with different generator sets
    def cayley_sl2_with_generators(p, gen_set):
        """Construct Cayley graph with specific generator set."""
        matrices = []
        for a in range(p):
            for b in range(p):
                for c in range(p):
                    if a == 0:
                        if b == 0 or c == 0:
                            continue
                        bc = (b * c) % p
                        if bc != (p - 1) % p:
                            continue
                        d = 0
                    else:
                        inv_a = pow(a, p-2, p)
                        d = (1 - b * c) % p * inv_a % p
                    matrices.append(((a, b), (c, d)))
        
        mat_to_idx = {}
        for i, mat in enumerate(matrices):
            mat_to_idx[mat] = i
        
        n = len(matrices)
        
        def mat_mul(m1, m2):
            a1, b1 = m1[0]
            c1, d1 = m1[1]
            a2, b2 = m2[0]
            c2, d2 = m2[1]
            return (((a1*a2 + b1*c2) % p, (a1*b2 + b1*d2) % p),
                    ((c1*a2 + d1*c2) % p, (c1*b2 + d1*d2) % p))
        
        def mat_inv(m):
            a, b = m[0]
            c, d = m[1]
            return ((d % p, (-b) % p), ((-c) % p, a % p))
        
        adj = [set() for _ in range(n)]
        
        for i, mat in enumerate(matrices):
            for gen in gen_set:
                prod = mat_mul(mat, gen)
                if prod in mat_to_idx:
                    j = mat_to_idx[prod]
                    if j != i:
                        adj[i].add(j)
        
        # Check 4-regular
        for i in range(n):
            if len(adj[i]) != 4:
                return None, None
        
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        
        return n, edges

    # Try various primes and generator sets
    p_candidates = [5, 7, 11, 13]
    
    for p in p_candidates:
        if time.time() > deadline:
            break
        
        n_group = p**3 - p
        if n_group >= best_n:
            continue
        
        # Try different generator sets
        gen_sets = []
        
        # Standard: S, S_inv, T, T_inv
        S = ((0, (p-1) % p), (1, 0))
        T = ((1, 1), (0, 1))
        
        def mat_inv(m):
            a, b = m[0]
            c, d = m[1]
            return ((d % p, (-b) % p), ((-c) % p, a % p))
        
        S_inv = mat_inv(S)
        T_inv = mat_inv(T)
        
        gen_sets.append([S, S_inv, T, T_inv])
        
        # Try T^2 instead of T
        def mat_mul(m1, m2):
            a1, b1 = m1[0]
            c1, d1 = m1[1]
            a2, b2 = m2[0]
            c2, d2 = m2[1]
            return (((a1*a2 + b1*c2) % p, (a1*b2 + b1*d2) % p),
                    ((c1*a2 + d1*c2) % p, (c1*b2 + d1*d2) % p))
        
        T2 = mat_mul(T, T)
        T2_inv = mat_inv(T2)
        gen_sets.append([S, S_inv, T2, T2_inv])
        
        # Try T^3
        T3 = mat_mul(T2, T)
        T3_inv = mat_inv(T3)
        gen_sets.append([S, S_inv, T3, T3_inv])
        
        for gens in gen_sets:
            if time.time() > deadline:
                break
            
            n, edges = cayley_sl2_with_generators(p, gens)
            if n is None:
                continue
            
            adj = [set() for _ in range(n)]
            for i, j in edges:
                adj[i].add(j)
                adj[j].add(i)
            
            girth = get_girth(adj, n, g)
            if girth >= g:
                if n < best_n:
                    best_n, best_edges = n, edges
    
    # Also try local search on the seed to see if we can reduce size
    # This is unlikely to help much, but try briefly
    if best_n > 320:
        for n_target in range(320, best_n):
            if time.time() > deadline:
                break
            
            adj = random_4reg(n_target)
            if adj is None:
                continue
            
            for iteration in range(100):
                if time.time() > deadline:
                    break
                    
                short_count = count_short_cycles(adj, n_target)
                if short_count == 0:
                    edges = []
                    for i in range(n_target):
                        for j in adj[i]:
                            if i < j:
                                edges.append((i, j))
                    if verify_graph(n_target, edges):
                        best_n, best_edges = n_target, edges
                        break
                else:
                    if not swap_edges(adj, n_target):
                        continue
    
    return best_n, best_edges
