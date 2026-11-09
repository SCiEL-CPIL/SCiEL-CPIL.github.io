"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    import heapq

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
        
        # Initial domains
        domains = [set(lst) for lst in lists]
        assigned = [-1] * n
        deg = [len(adj[i]) for i in range(n)]
        
        trail = []

        def backtrack():
            # Find MRV
            best_v = -1
            best_count = n + 1
            best_deg = -1
            for v in range(n):
                if assigned[v] == -1:
                    cnt = len(domains[v])
                    if cnt == 0:
                        return False
                    if cnt < best_count or (cnt == best_count and deg[v] > best_deg):
                        best_v = v
                        best_count = cnt
                        best_deg = deg[v]
            
            if best_v == -1:
                return True
            
            for c in sorted(domains[best_v]):
                conflict = False
                for nb in adj[best_v]:
                    if assigned[nb] == c:
                        conflict = True
                        break
                if conflict:
                    continue
                
                assigned[best_v] = c
                trail_mark = len(trail)
                
                valid = True
                for nb in adj[best_v]:
                    if assigned[nb] == -1 and c in domains[nb]:
                        domains[nb].discard(c)
                        trail.append((nb, c))
                        if not domains[nb]:
                            valid = False
                            break
                
                if valid and backtrack():
                    return True
                
                # Undo
                assigned[best_v] = -1
                while len(trail) > trail_mark:
                    nb, c_add = trail.pop()
                    domains[nb].add(c_add)
            
            return False

        if backtrack():
            return [assigned[v] for v in range(n)]
        else:
            return [lists[v][0] for v in range(n)]

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        coloring = solve(n, edges, lists)
        # Ensure types are Python int
        coloring = [int(c) for c in coloring]
        out.append(coloring)
    return out
# EVOLVE-BLOCK-END
