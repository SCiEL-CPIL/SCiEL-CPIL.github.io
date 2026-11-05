import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = 120.0
    
    # Fallback: Optimized LUW
    def luw_fallback():
        q = k
        # For g=9, d=4 gives girth 9 and n=2*q^4=1250
        # For general g, d = (g-1)//2
        d = (g - 1) // 2
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        
        # Rule for LUW
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

    # Try Cayley Graph on PSL(2, 11)
    def try_psl211():
        # Implement PSL(2, 11)
        mod = 11
        # Generate all elements
        elements = []
        for a in range(mod):
            for b in range(mod):
                for c in range(mod):
                    # ad - bc = 1 => d = (1 + bc) * a^-1 if a != 0
                    if a == 0:
                        # -bc = 1 => bc = -1 = 10
                        # b, c in GF(11)
                        # d can be anything? No, det=1.
                        # If a=0, -bc=1. d is free? No, matrix is [[0,b],[c,d]].
                        # det = -bc. So -bc=1.
                        # d is not determined by det=1 alone?
                        # In PSL, we identify M ~ -M.
                        # Actually, we just need a set of representatives.
                        # Let's just iterate all 11^4 = 14641 matrices and filter det=1.
                        pass
        # Better: iterate all 4-tuples
        all_mats = []
        for a in range(mod):
            for b in range(mod):
                for c in range(mod):
                    for d in range(mod):
                        if (a*d - b*c) % mod == 1:
                            all_mats.append((a, b, c, d))
        
        # Map to index
        mat_to_idx = {m: i for i, m in enumerate(all_mats)}
        n_group = len(all_mats) # Should be 660
        
        # Precompute multiplication table? 660x660x4 is ~1.7M ops. Fast.
        # But we only need to check girth for specific generators.
        
        def mul(m1, m2):
            a1, b1, c1, d1 = m1
            a2, b2, c2, d2 = m2
            return ((a1*a2 + b1*c2) % mod,
                    (a1*b2 + b1*d2) % mod,
                    (c1*a2 + d1*c2) % mod,
                    (c1*b2 + d1*d2) % mod)
        
        def inv(m):
            a, b, c, d = m
            return (d, -b % mod, -c % mod, a)
        
        identity = (1, 0, 0, 1)
        
        # Find involutions (s^2 = 1)
        involutions = [m for m in all_mats if mul(m, m) == identity]
        
        # Find non-involutions
        non_involutions = [m for m in all_mats if mul(m, m) != identity and m != identity]
        
        def check_girth(generators):
            # generators is a list of 5 elements
            # Build adjacency list
            adj = [[] for _ in range(n_group)]
            for i in range(n_group):
                for g in generators:
                    neighbor = mat_to_idx[mul(all_mats[i], g)]
                    adj[i].append(neighbor)
            
            # BFS from 0 (identity)
            dist = [-1] * n_group
            dist[0] = 0
            queue = [0]
            girth = float('inf')
            
            for u in queue:
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
                    else:
                        # Cycle length
                        # In undirected graph, if v is visited, 
                        # cycle length is dist[u] + 1 + dist[v]
                        # But we need to be careful with the tree edges.
                        # Standard BFS girth finding:
                        # If v is not parent of u, cycle length is dist[u] + dist[v] + 1
                        # We don't track parent here easily in this loop.
                        # Let's just do a full BFS and check edges.
                        pass
            
            # More robust girth check:
            # For each edge (u, v), if dist[u] + dist[v] + 1 is a cycle?
            # No, that's only if the path from root to u and root to v are disjoint except at root.
            # Simplest: BFS from each node? Too slow.
            # BFS from root, and for each edge (u, v) where u != parent(v) and v != parent(u),
            # cycle length = dist[u] + dist[v] + 1.
            
            parent = [-1] * n_group
            dist = [-1] * n_group
            dist[0] = 0
            queue = [0]
            girth = float('inf')
            
            for u in queue:
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u] and u != parent[v]:
                        # Found a cycle
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < girth:
                            girth = cycle_len
            return girth

        # Search
        best_girth = 0
        best_gen = None
        
        # Time limit for search
        search_deadline = time.time() + 110.0
        
        while time.time() < search_deadline:
            # Pick 2 pairs and 1 involution
            s1 = random.choice(non_involutions)
            s2 = random.choice(non_involutions)
            s3 = random.choice(involutions)
            
            # Ensure distinct
            if s1 == s2 or s1 == inv(s2) or s1 == s3 or s2 == s3:
                continue
            
            gens = [s1, inv(s1), s2, inv(s2), s3]
            
            # Check if they generate the group
            # BFS from identity
            visited = {0}
            queue = [0]
            while queue:
                u = queue.pop(0)
                for g in gens:
                    v = mat_to_idx[mul(all_mats[u], g)]
                    if v not in visited:
                        visited.add(v)
                        queue.append(v)
            
            if len(visited) < n_group:
                continue
            
            # Check girth
            girth = check_girth(gens)
            if girth >= g:
                # Construct edges
                edges = []
                for i in range(n_group):
                    for g in gens:
                        j = mat_to_idx[mul(all_mats[i], g)]
                        if i < j:
                            edges.append((i, j))
                return n_group, edges
            
            if girth > best_girth:
                best_girth = girth
                best_gen = gens

        return None

    result = try_psl211()
    if result:
        return result
    
    # Fallback to LUW
    return luw_fallback()
