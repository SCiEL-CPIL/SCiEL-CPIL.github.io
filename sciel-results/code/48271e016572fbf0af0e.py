"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from random import Random

    DEADLINE = 150.0
    SAFETY = 5.0

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

    def solve_backtrack(n, edges, lists, order):
        """Backtracking solver with forward checking. Returns colour list or None."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Build available colours per vertex (from list)
        avail = [set(l) for l in lists]

        # Order of vertices to assign
        remaining = set(range(n))
        assignment = [None] * n

        # We'll use iterative deepening or just a simple backtracking with good heuristics
        # For these small graphs (max ~408 vertices), backtracking should be fast enough

        def backtrack(idx):
            if idx == len(order):
                return True

            v = order[idx]
            # Check which colours are still available (not used by already-assigned neighbours)
            for c in avail[v]:
                conflict = False
                for u in adj[v]:
                    if assignment[u] is not None and assignment[u] == c:
                        conflict = True
                        break
                if not conflict:
                    assignment[v] = c
                    if backtrack(idx + 1):
                        return True
                    assignment[v] = None
            return False

        if backtrack(0):
            return assignment
        return None

    def get_orderings(n, edges, lists):
        """Generate multiple vertex orderings for backtracking."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        orderings = []

        # Ordering 1: descending degree
        order1 = sorted(range(n), key=lambda u: -len(adj[u]))
        orderings.append(order1)

        # Ordering 2: ascending list size, then descending degree
        order2 = sorted(range(n), key=lambda u: (len(lists[u]), -len(adj[u])))
        orderings.append(order2)

        # Ordering 3: minimum remaining values heuristic (list size / degree)
        order3 = sorted(range(n), key=lambda u: (len(lists[u]) / max(1, len(adj[u])), -len(adj[u])))
        orderings.append(order3)

        # Ordering 4: reverse of degree ordering
        order4 = sorted(range(n), key=lambda u: len(adj[u]))
        orderings.append(order4)

        return orderings

    out = []
    start_time = time.time()

    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - SAFETY:
            # Not enough time; break with what we have
            break

        n, edges, lists = instance(splits, seed)
        orderings = get_orderings(n, edges, lists)

        colour = None
        for order in orderings:
            elapsed = time.time() - start_time
            if elapsed > DEADLINE - SAFETY:
                break
            colour = solve_backtrack(n, edges, lists, order)
            if colour is not None:
                break

        # Fallback: greedy if backtracking failed
        if colour is None:
            adj = [set() for _ in range(n)]
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            colour = [None] * n
            for v in sorted(range(n), key=lambda u: -len(adj[u])):
                taken = {colour[u] for u in adj[v] if colour[u] is not None}
                free = [c for c in lists[v] if c not in taken]
                colour[v] = free[0] if free else lists[v][0]

        # Ensure all values are Python int
        colour = [int(c) for c in colour]
        out.append(colour)

    # If we didn't get all 4, fill remaining with greedy
    while len(out) < 4:
        splits, seed = ((20, 701), (40, 702), (60, 703), (100, 704))[len(out)]
        n, edges, lists = instance(splits, seed)
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        colour = [None] * n
        for v in sorted(range(n), key=lambda u: -len(adj[u])):
            taken = {colour[u] for u in adj[v] if colour[u] is not None}
            free = [c for c in lists[v] if c not in taken]
            colour[v] = free[0] if free else lists[v][0]
        colour = [int(c) for c in colour]
        out.append(colour)

    return out
# EVOLVE-BLOCK-END
