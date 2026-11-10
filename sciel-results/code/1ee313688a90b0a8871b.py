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

    def solve_list_coloring(n, edges, lists):
        # Build adjacency
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Bipartition: BFS from vertex 0
        color_side = [-1] * n
        color_side[0] = 0
        queue = [0]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adj[u]:
                if color_side[v] == -1:
                    color_side[v] = 1 - color_side[u]
                    queue.append(v)

        # If any vertex remains uncolored (disconnected component), assign arbitrarily
        for i in range(n):
            if color_side[i] == -1:
                color_side[i] = 0

        # Separate into left and right sets
        left = [i for i in range(n) if color_side[i] == 0]
        right = [i for i in range(n) if color_side[i] == 1]

        # For each vertex, its allowed colors
        # We'll use a backtracking approach with constraint propagation
        # Since lists have size 3 and max degree is bounded, this should be fast

        # Order vertices by degree (descending) for better pruning
        order = sorted(range(n), key=lambda v: -len(adj[v]))

        assignment = [None] * n
        # For efficient checking, maintain for each color a set of unassigned vertices that can take it
        # But simpler: just try colors in order

        def backtrack(idx):
            if idx == len(order):
                return True

            v = order[idx]
            # Try each color in the list
            for c in lists[v]:
                # Check if any neighbor already has color c
                conflict = False
                for u in adj[v]:
                    if assignment[u] == c:
                        conflict = True
                        break
                if conflict:
                    continue

                assignment[v] = c
                if backtrack(idx + 1):
                    return True
                assignment[v] = None

            return False

        if backtrack(0):
            # Convert to proper Python ints
            return [int(c) for c in assignment]
        else:
            # Fallback: should not happen by Alon-Tarsi, but return something
            return [lists[v][0] for v in range(n)]

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        coloring = solve_list_coloring(n, edges, lists)
        # Ensure all values are Python ints
        out.append([int(c) for c in coloring])
    return out
# EVOLVE-BLOCK-END
