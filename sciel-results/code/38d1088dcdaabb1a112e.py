def construct():
    from random import Random
    import sys
    sys.setrecursionlimit(10000)

    def instance(splits, seed, palette=5):
        rng = Random(seed)
        faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
        edges = {(min(a, b), max(a, b)) for f in faces for a, b in zip(f, f[1:] + f[:1])}
        n = 8
        for _ in range(splits):
            face = faces.pop(rng.randrange(len(faces)))
            inner = list(range(n, n + 4))
            n += 4
            for i in range(4):
                a, b = face[i], face[(i + 1) % 4]
                edges.add((min(a, inner[i]), max(a, inner[i])))
                edges.add((min(inner[i], inner[(i + 1) % 4]), max(inner[i], inner[(i + 1) % 4])))
                faces.append([a, b, inner[(i + 1) % 4], inner[i]])
            faces.append(inner)
        lists = [sorted(rng.sample(range(palette), 3)) for _ in range(n)]
        return n, sorted(edges), lists

    def solve(n, edges, lists):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # domains as bitmasks
        # Bit i set means colour i is available
        domains = [0] * n
        for i, l in enumerate(lists):
            mask = 0
            for c in l:
                mask |= (1 << c)
            domains[i] = mask
            
        color = [-1] * n
        uncoloured = set(range(n))
        
        # Cache for MRV? No, just calculate.
        
        def backtrack():
            if not uncoloured:
                return True
            
            # Find MRV vertex
            min_count = 4
            best_v = -1
            # Optimization: iterate list, not set, for speed? Set is fine.
            for v in uncoloured:
                count = bin(domains[v]).count('1')
                if count < min_count:
                    min_count = count
                    best_v = v
                    if count == 1:
                        break
                elif count == min_count:
                    # Tie break: Degree
                    if len(adj[v]) > len(adj[best_v]):
                        best_v = v
                        
            v = best_v
            uncoloured.remove(v)
            
            # Try colours in domain
            # Extract bits
            mask = domains[v]
            while mask:
                c_bit = mask & -mask
                c = c_bit.bit_length() - 1
                mask -= c_bit
                
                # Check if c is valid (not used by coloured neighbours)
                valid = True
                for u in adj[v]:
                    if color[u] == c:
                        valid = False
                        break
                        
                if valid:
                    color[v] = c
                    if backtrack():
                        return True
                    color[v] = -1
                    
            uncoloured.add(v)
            return False

        if backtrack():
            return color
        return None

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        sol = solve(n, edges, lists)
        if sol is None:
            sol = [0] * n
        out.append([int(c) for c in sol])
    return out
