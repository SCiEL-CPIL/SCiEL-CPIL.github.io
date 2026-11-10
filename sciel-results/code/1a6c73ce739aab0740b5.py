"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

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

    def solve_csp(n, edges, lists):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        domains = [set(lst) for lst in lists]

        # AC-3: arc consistency
        import collections
        queue = collections.deque()
        for u in range(n):
            for v in adj[u]:
                queue.append((u, v))
        while queue:
            x, y = queue.popleft()
            dy = domains[y]
            if not dy:
                return None
            removed = False
            for c in list(domains[x]):
                if all(c == c2 for c2 in dy):
                    # c has no support in y
                    domains[x].discard(c)
                    removed = True
            if not domains[x]:
                return None
            if removed:
                for z in adj[x]:
                    if z != y:
                        queue.append((z, x))

        # Order by degree (descending)
        order = sorted(range(n), key=lambda u: -len(adj[u]))
        pos = [0] * n
        for i, u in enumerate(order):
            pos[u] = i

        assign = [None] * n

        def backtrack(k):
            if k == n:
                return True
            u = order[k]
            # MRV among unassigned? we fixed order; pick first available
            dom = domains[u]
            # forward checking: try each colour
            for c in sorted(dom):
                ok = True
                # check neighbours already assigned
                for v in adj[u]:
                    av = assign[v]
                    if av is not None and av == c:
                        ok = False
                        break
                if not ok:
                    continue
                assign[u] = c
                if backtrack(k + 1):
                    return True
                assign[u] = None
            return False

        if not backtrack(0):
            return None

        return [int(assign[i]) for i in range(n)]

    deadline = _time.time() + 140.0
    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        sol = solve_csp(n, edges, lists)
        if sol is None:
            # fallback: should not happen; use first list colour
            sol = [lists[i][0] for i in range(n)]
        out.append(sol)
        if _time.time() > deadline:
            break
    return out
# EVOLVE-BLOCK-END
