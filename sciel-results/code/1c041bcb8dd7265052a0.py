"""MODEL-AND-OPTIMIZE: CSP solver with Bitmask Domains, MRV, and Forward Checking."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    sys.setrecursionlimit(20000)

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

    def solve_csp(n, edges, lists, deadline):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Precompute adjacency masks
        adj_masks = [0] * n
        for v in range(n):
            m = 0
            for u in adj[v]:
                m |= (1 << u)
            adj_masks[v] = m

        # Convert lists to bitmasks
        domains = [0] * n
        for i in range(n):
            mask = 0
            for c in lists[i]:
                mask |= (1 << c)
            domains[i] = mask

        for i in range(n):
            if domains[i] == 0:
                return None

        assignment = [-1] * n
        timed_out = [False]

        def backtrack(assigned_count):
            if time.time() > deadline:
                timed_out[0] = True
                return False

            if assigned_count == n:
                return True

            # MRV: Find unassigned vertex with smallest valid domain size
            min_var = -1
            min_size = 10
            min_valid_mask = 0

            for v in range(n):
                if assignment[v] == -1:
                    valid_mask = domains[v]
                    for u in adj[v]:
                        if assignment[u] != -1:
                            valid_mask &= ~(1 << assignment[u])

                    # Count bits
                    size = 0
                    tmp = valid_mask
                    while tmp:
                        tmp &= tmp - 1
                        size += 1

                    if size < min_size:
                        min_size = size
                        min_var = v
                        min_valid_mask = valid_mask
                        if min_size == 0:
                            return False
                        if min_size == 1:
                            break

            if min_var == -1:
                return False

            # Try colors in min_valid_mask
            mask = min_valid_mask
            while mask:
                lsb = mask & -mask
                c = lsb.bit_length() - 1
                mask ^= lsb

                assignment[min_var] = c

                if backtrack(assigned_count + 1):
                    return True

                assignment[min_var] = -1
                if timed_out[0]:
                    return False

            return False

        if backtrack(0):
            return assignment
        else:
            return None

    def build_result():
        params = ((20, 701), (40, 702), (60, 703), (100, 704))
        results = []
        for splits, seed in params:
            n, edges, lists = instance(splits, seed)
            coloring = solve_csp(n, edges, lists, deadline)
            if coloring is None:
                coloring = [lists[v][0] for v in range(n)]
            results.append([int(c) for c in coloring])
        return results

    start_time = time.time()
    deadline = start_time + 120.0

    result = build_result()

    return result

# EVOLVE-BLOCK-END
