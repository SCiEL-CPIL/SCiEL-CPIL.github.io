import time
import random
import sys
import math
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # 110s limit, leaving 40s margin for safety

    # Fallback: LUW Construction
    def luw_construct(k, g):
        q = k
        d = g - 5
        if d % 2 == 0:
            d += 1
        d = max(3, d)
        
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q

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
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        
        for i, vertex in enumerate(order):
            if time.time() > deadline:
                break
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
                
        return len(order), edges

    # Helper to check girth using BFS from a single node
    # Returns the girth found (or inf if no cycle found within depth limit)
    def check_girth_single(adj, n, start_node, max_depth):
        dist = [-1] * n
        parent = [-1] * n
        queue = deque()
        queue.append(start_node)
        dist[start_node] = 0
        girth = float('inf')
        
        while queue:
            u = queue.popleft()
            if dist[u] >= max_depth:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    if parent[u] != v and parent[v] != u:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < girth:
                            girth = cycle_len
                            if girth < g:
                                return girth
        return girth

    # Attempt 1: Tinhofer Construction
    # Search for a Cayley graph on Z_2 x Z_p^d with generators {0, +/-v_i} and (1, 0)
    def try_tinhofer(k, g, time_limit):
        if k % 2 != 1:
            return None
            
        m = k // 2
        # Moore bound for k=5, g=9 is ~106. 
        # n = 2 * p^d. 
        # If p=5, d=3 -> 250. 
        # If p=7, d=3 -> 686.
        # If p=3, d=5 -> 486.
        
        # We want small n. 
        # Let's try small primes and dimensions.
        # n must be even.
        
        # Target n is roughly Moore bound.
        # We'll iterate over possible (p, d) pairs that give n in a reasonable range.
        
        # Precompute powers of primes
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
        
        # We'll collect candidate (p, d) sorted by n = 2*p^d
        candidates = []
        for p in primes:
            power = p
            d = 1
            while 2 * power < 1000: # Limit search space to < 1000 vertices for speed
                candidates.append((p, d, 2 * power))
                power *= p
                d += 1
            if power <= 500: # Allow slightly larger if p is small
                 candidates.append((p, d, 2 * power))
        
        # Sort by n
        candidates.sort(key=lambda x: x[2])
        
        # Try candidates
        for p, d, n in candidates:
            if time.time() > time_limit:
                return None
            
            # Skip if n is too small to have girth g?
            # Moore bound M(k, g).
            # If n < M(k, g), impossible.
            # M(5, 9) ~ 106.
            # So we can skip n < 106.
            if n < 106:
                continue
            
            # Generate random sets of vectors
            # We need m vectors v_1...v_m in Z_p^d
            # Such that the set {0, +/-v_1, ..., +/-v_m} generates no short relations.
            # Specifically, no product of < g/2 generators equals 0 (in the Cayley graph sense, considering the involution).
            # Actually, in Tinhofer, the graph is on Z_2 x Z_p^d.
            # Generators: (0, v), (0, -v) for v in V, and (1, 0).
            # A cycle of length L corresponds to a word in generators of length L that evaluates to identity.
            
            # Heuristic: Random vectors often work for large p.
            # We try a few random instances for each (p, d).
            
            max_attempts = 10
            for attempt in range(max_attempts):
                if time.time() > time_limit:
                    return None
                
                rng = random.Random(12345 + p * 100 + d + attempt)
                
                # Generate m random non-zero vectors
                vectors = []
                valid_vectors = True
                while len(vectors) < m:
                    v = tuple(rng.randint(0, p - 1) for _ in range(d))
                    if all(x == 0 for x in v):
                        continue
                    # Check if v is already in vectors or -v
                    conflict = False
                    for u in vectors:
                        # v == u or v == -u (mod p)
                        if all((v[i] == u[i] or (v[i] + u[i]) % p == 0) for i in range(d)):
                            conflict = True
                            break
                    if not conflict:
                        vectors.append(v)
                
                if len(vectors) < m:
                    continue
                
                # Build adjacency list
                adj = [[] for _ in range(n)]
                # State (bit, coord) -> index: bit * p^d + sum(coord[i] * p^i)
                # Actually, index = bit * p^d + coord[0]*p^(d-1) + ... ?
                # Let's use: index = bit * p^d + val, where val is integer representation of coord.
                
                # Precompute powers of p
                p_powers = [1] * d
                for i in range(1, d):
                    p_powers[i] = p_powers[i-1] * p
                
                def coord_to_val(c):
                    val = 0
                    for i in range(d):
                        val += c[i] * p_powers[d-1-i] # Big-endian or little-endian doesn't matter as long as consistent
                    return val
                
                def val_to_coord(val):
                    c = [0] * d
                    for i in range(d-1, -1, -1):
                        c[i] = val % p
                        val //= p
                    return tuple(c)
                
                # Generate all states? No, just build adj on the fly or full adj.
                # Full adj is O(n * k). n is small.
                
                # List of generators as (bit_change, vector)
                gens = []
                for v in vectors:
                    neg_v = tuple((-x) % p for x in v)
                    gens.append((0, v))
                    gens.append((0, neg_v))
                gens.append((1, tuple(0 for _ in range(d))))
                
                # Build adj
                for u in range(n):
                    bit = u // (p ** d)
                    val = u % (p ** d)
                    coord = val_to_coord(val)
                    
                    for b, v in gens:
                        nb = (bit + b) % 2
                        ncoord = tuple((coord[i] + v[i]) % p for i in range(d))
                        nval = coord_to_val(ncoord)
                        nv = nb * (p ** d) + nval
                        adj[u].append(nv)
                
                # Check girth
                # We need girth >= g.
                # BFS from node 0.
                # If we find a cycle of length < g, fail.
                
                # Optimization: Only need to check up to depth (g-1)//2
                max_depth = (g - 1) // 2
                girth = check_girth_single(adj, n, 0, max_depth)
                
                if girth >= g:
                    # Verify full girth? 
                    # If BFS from 0 finds no cycle < g, does that mean girth >= g?
                    # No, a cycle might not pass through 0.
                    # However, for regular graphs, if girth < g, there is a short cycle.
                    # It might not contain 0.
                    # We should check BFS from a few more nodes, or all nodes.
                    # Given time constraints, checking from 0 is a heuristic.
                    # To be safe, let's check from a few random nodes.
                    
                    valid = True
                    # Check from 1, 2, ... up to min(5, n)
                    for start_node in range(1, min(5, n)):
                        if time.time() > time_limit:
                            valid = False
                            break
                        girth2 = check_girth_single(adj, n, start_node, max_depth)
                        if girth2 < g:
                            valid = False
                            break
                    
                    if valid:
                        # Construct edges
                        edges = []
                        for u in range(n):
                            for v in adj[u]:
                                if u < v:
                                    edges.append((u, v))
                        return n, edges
            
        return None

    # 1. Try Tinhofer
    # Limit time to 60s
    tino_result = try_tinhofer(k, g, start_time + 60.0)
    if tino_result is not None:
        return tino_result

    # 2. Fallback to LUW
    return luw_construct(k, g)
