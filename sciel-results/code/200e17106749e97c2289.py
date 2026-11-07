"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
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
        # Order vertices by degree descending (most constrained first)
        order = sorted(range(n), key=lambda u: -len(adj[u]))
        # For each vertex in order, precompute which later vertices it constrains
        pos = {v: i for i, v in enumerate(order)}
        later = [[] for _ in range(n)]
        for v in range(n):
            pv = pos[v]
            for u in adj[v]:
                if pos[u] > pv:
                    later[v].append(u)

        colour = [None] * n
        domains = [set(lst) for lst in lists]

        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            # Find available colours: in domain and not used by any coloured neighbour
            used = set()
            for u in adj[v]:
                cu = colour[u]
                if cu is not None:
                    used.add(cu)
            # Try colours in domain minus used
            for c in sorted(domains[v] - used):
                colour[v] = c
                # Forward checking: ensure all uncoloured neighbours still have options
                ok = True
                for u in later[v]:
                    if u in used:
                        continue
                    # Check if u's domain minus {c} minus already-used-neighbours is non-empty
                    used_u = set()
                    for w in adj[u]:
                        cw = colour[w]
                        if cw is not None and cw != c:
                            used_u.add(cw)
                        elif cw == c:
                            # w is v, colour c
                            used_u.add(c)
                    if not (domains[u] - used_u):
                        ok = False
                        break
                if ok:
                    if backtrack(idx + 1):
                        return True
                colour[v] = None
            return False

        if not backtrack(0):
            # Fallback: should not happen per Alon-Tarsi, but return something valid-length
            return [lists[v][0] for v in range(n)]
        return colour

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        out.append(solve_instance(n, edges, lists))
    return out
# EVOLVE-BLOCK-END
