def construct():
    from random import Random

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

    def solve_instance(n, edges, lists):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Bitmask for colors
        # Color c corresponds to bit (1 << c)
        avail = [0] * n
        for v in range(n):
            mask = 0
            for c in lists[v]:
                mask |= (1 << c)
            avail[v] = mask
            
        colour = [-1] * n
        
        # MRV Heuristic: Pick uncoloured vertex with fewest available colours
        # Tie-breaker: Highest degree
        
        def get_mrv_vertex():
            min_count = 100
            best_v = -1
            for v in range(n):
                if colour[v] == -1:
                    count = bin(avail[v]).count('1')
                    if count < min_count:
                        min_count = count
                        best_v = v
                    elif count == min_count:
                        if len(adj[v]) > len(adj[best_v]):
                            best_v = v
            return best_v

        def backtrack():
            v = get_mrv_vertex()
            if v == -1:
                return True
            
            # Try each color in avail[v]
            mask = avail[v]
            while mask:
                # Get lowest set bit
                c_bit = mask & -mask
                c = c_bit.bit_length() - 1
                mask ^= c_bit
                
                # Check conflicts
                conflict = False
                for u in adj[v]:
                    if colour[u] == c:
                        conflict = True
                        break
                
                if conflict:
                    continue
                
                # Assign
                colour[v] = c
                # Propagate: Remove c from neighbours
                saved = []
                valid = True
                for u in adj[v]:
                    if colour[u] == -1:
                        if avail[u] & c_bit:
                            saved.append(u)
                            avail[u] &= ~c_bit
                            if avail[u] == 0:
                                valid = False
                                break
                
                if valid and backtrack():
                    return True
                
                # Undo
                colour[v] = -1
                for u in saved:
                    avail[u] |= c_bit
            
            return False

        if backtrack():
            return [int(c) for c in colour]
        else:
            # Fallback (should not happen)
            return [lists[v][0] for v in range(n)]

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        out.append(solve_instance(n, edges, lists))
    return out
