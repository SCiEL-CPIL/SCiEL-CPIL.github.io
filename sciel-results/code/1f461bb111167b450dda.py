"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import sys

    sys.setrecursionlimit(10000)

    def instance(splits, seed, palette=5):
        from random import Random
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
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Precompute degrees for tiebreaking
        deg = [len(adj[v]) for v in range(n)]

        # Initialize domains as sets of available colors
        domains = [set(lst) for lst in lists]
        color = [-1] * n

        # Count unassigned for early termination
        unassigned_count = [n]

        def propagate(v, c, changes):
            """Forward checking: remove c from neighbors' domains. Returns False if any domain becomes empty."""
            for u in adj[v]:
                if color[u] == -1:
                    if c in domains[u]:
                        domains[u].remove(c)
                        changes.append((u, c))
                        if not domains[u]:
                            return False
            return True

        def undo(changes):
            """Restore removed colors from domains."""
            for u, c in changes:
                domains[u].add(c)

        def backtrack():
            if unassigned_count[0] == 0:
                return True

            # Find unassigned variable with Minimum Remaining Values (MRV)
            # Tiebreak: highest degree (more constrained)
            min_len = 10
            best_v = -1
            best_deg = -1
            for v in range(n):
                if color[v] == -1:
                    d_len = len(domains[v])
                    if d_len == 0:
                        return False
                    if d_len < min_len or (d_len == min_len and deg[v] > best_deg):
                        min_len = d_len
                        best_v = v
                        best_deg = deg[v]
                        if min_len == 1:
                            break

            if best_v == -1:
                return True

            v = best_v
            saved_domain = set(domains[v])

            # LCV: order colors by how few options they remove from unassigned neighbors
            # For each candidate color c, count total domain reductions in unassigned neighbors
            def lcv_key(c):
                total_removed = 0
                for u in adj[v]:
                    if color[u] == -1 and c in domains[u]:
                        total_removed += 1
                return total_removed

            colors_to_try = sorted(saved_domain, key=lcv_key)

            for c in colors_to_try:
                color[v] = c
                unassigned_count[0] -= 1

                changes = []
                valid = propagate(v, c, changes)

                if valid and backtrack():
                    return True

                # Undo
                undo(changes)
                unassigned_count[0] += 1

            color[v] = -1
            domains[v] = saved_domain
            return False

        if backtrack():
            # Ensure types are Python int
            return [int(c) for c in color]
        else:
            # Fallback, though problem guarantees solvability
            return [0] * n

    out = []
    start_time = time.time()
    deadline = start_time + 110.0

    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline:
            n, edges, lists = instance(splits, seed)
            out.append([0] * n)
            continue

        n, edges, lists = instance(splits, seed)
        res = solve(n, edges, lists)
        out.append(res)

    return out
# EVOLVE-BLOCK-END
