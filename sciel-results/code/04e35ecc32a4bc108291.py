"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    _start = time.monotonic()
    _deadline = _start + 140.0  # safety margin before 150s kill

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

    def solve(n, edges, lists):
        """Complete backtracking solver with MRV, forward checking, and degree ordering."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Domain for each vertex: list of allowed colours
        domains = [list(l) for l in lists]

        # Order vertices by decreasing degree for initial ordering, but use MRV dynamically
        degree = [len(adj[v]) for v in range(n)]

        colour = [None] * n
        # We'll use a stack-based backtracking with MRV selection

        def backtrack():
            # Find uncoloured vertex with minimum remaining values (MRV)
            best_v = -1
            best_domain_size = n + 1
            for v in range(n):
                if colour[v] is None:
                    # Compute remaining domain
                    taken = set()
                    for u in adj[v]:
                        if colour[u] is not None:
                            taken.add(colour[u])
                    rem = [c for c in domains[v] if c not in taken]
                    if len(rem) < best_domain_size:
                        best_domain_size = len(rem)
                        best_v = v
                        if best_domain_size == 0:
                            return False
                        if best_domain_size == 1:
                            break

            if best_v == -1:
                return True  # All coloured

            # Try each colour in the domain
            taken = set()
            for u in adj[best_v]:
                if colour[u] is not None:
                    taken.add(colour[u])
            for c in domains[best_v]:
                if c in taken:
                    continue
                # Check if assigning c to best_v causes any neighbour to have empty domain
                valid = True
                for u in adj[best_v]:
                    if colour[u] is None:
                        # Check if u would lose all its options
                        taken_u = set()
                        for w in adj[u]:
                            if w == best_v:
                                taken_u.add(c)
                            elif colour[w] is not None:
                                taken_u.add(colour[w])
                        if not any(dom in domains[u] for dom in domains[u] if dom not in taken_u):
                            valid = False
                            break
                if valid:
                    colour[best_v] = c
                    if backtrack():
                        return True
                    colour[best_v] = None

            return False

        if backtrack():
            return colour
        else:
            return None

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.monotonic() > _deadline:
            # Emergency: return whatever we have, padded with None (will be invalid, but avoids timeout)
            break
        n, edges, lists = instance(splits, seed)
        result = solve(n, edges, lists)
        if result is None:
            # Should not happen by Alon-Tarsi, but fallback
            result = [lists[v][0] for v in range(n)]
        # Ensure all values are Python int
        result = [int(c) for c in result]
        out.append(result)

    return out
# EVOLVE-BLOCK-END
