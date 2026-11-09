"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

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

    def solve_csp(n, edges, lists, deadline):
        # Build adjacency as lists
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Precompute domain masks: bit i set if color i is in lists[v]
        # Colors are 0..4, so 5 bits
        domain = [0] * n
        for v in range(n):
            mask = 0
            for c in lists[v]:
                mask |= (1 << c)
            domain[v] = mask

        # Order vertices by degree descending (most constrained first)
        order = sorted(range(n), key=lambda u: -len(adj[u]))
        # Map original vertex -> position in order
        pos = [0] * n
        for i, v in enumerate(order):
            pos[v] = i

        # Build adjacency in order-index space
        adj_order = [[] for _ in range(n)]
        for i, u in enumerate(order):
            for v in adj[u]:
                adj_order[i].append(pos[v])

        # Backtracking with forward checking
        # color[i] = -1 if uncolored, else 0..4
        color = [-1] * n
        # domains[i] = current available colors (bitmask) for vertex at order-index i
        cur_domains = list(domain)

        # Precompute for each order-index i, which neighbors have lower order-index (already colored when we get to i)
        # Actually, we process in order, so when coloring order[i], all neighbors with order < i are already colored.
        # Neighbors with order > i are not yet colored.
        # For forward checking, when we color order[i], we need to update domains of neighbors with order > i.

        # For efficiency, precompute neighbor order-indices
        # nbrs_in_order[i] = list of order-indices of neighbors of order[i]
        nbrs_in_order = adj_order  # already in order-index space

        solution = [None] * n

        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == n:
                return True

            v = order[idx]
            d = cur_domains[idx]
            if d == 0:
                return False

            # Try each available color
            # Iterate colors in some order (low to high)
            c = 0
            while c < 5:
                if d & (1 << c):
                    # Try color c
                    color[idx] = c
                    solution[v] = c

                    # Forward check: update domains of neighbors with order > idx
                    saved = []
                    valid = True
                    for nb_idx in nbrs_in_order[idx]:
                        if nb_idx > idx:
                            old_d = cur_domains[nb_idx]
                            new_d = old_d & ~(1 << c)
                            if new_d == 0:
                                valid = False
                                break
                            saved.append((nb_idx, old_d))
                            cur_domains[nb_idx] = new_d

                    if valid:
                        if backtrack(idx + 1):
                            return True

                    # Undo
                    for nb_idx, old_d in saved:
                        cur_domains[nb_idx] = old_d

                    color[idx] = -1
                    solution[v] = None

                c += 1

            return False

        if backtrack(0):
            return solution

        return None

    deadline = time.time() + 110.0  # Safety margin before 120s limit
    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        sol = solve_csp(n, edges, lists, deadline)
        if sol is not None:
            # Convert to proper Python ints and ensure list type
            out.append([int(c) for c in sol])
        else:
            # Fallback: should not happen by Alon-Tarsi, but provide something
            # Use a simple greedy as fallback
            adj = [set() for _ in range(n)]
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            colour = [0] * n
            for v in sorted(range(n), key=lambda u: -len(adj[u])):
                taken = {colour[u] for u in adj[v]}
                free = [c for c in lists[v] if c not in taken]
                colour[v] = free[0] if free else lists[v][0]
            out.append([int(c) for c in colour])

    return out
# EVOLVE-BLOCK-END
