"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
def construct():
    import time
    _deadline = time.time() + 140  # safety margin before 150s kill
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

    def is_valid(colour, n, adj, lists):
        for v in range(n):
            c = colour[v]
            if c is None or c not in lists[v]:
                return False
            for u in adj[v]:
                if colour[u] == c:
                    return False
        return True

    def backtrack(n, adj, lists, order, colour, idx):
        if idx == n:
            return True
        v = order[idx]
        taken = set()
        for u in adj[v]:
            if colour[u] is not None:
                taken.add(colour[u])
        for c in lists[v]:
            if c not in taken:
                colour[v] = c
                if backtrack(n, adj, lists, order, colour, idx + 1):
                    return True
                colour[v] = None
        return False

    def greedy(n, edges, palette):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        colour = [None] * n
        for v in sorted(range(n), key=lambda u: -len(adj[u])):
            taken = {colour[u] for u in adj[v]}
            free = [c for c in palette(v) if c not in taken]
            colour[v] = free[0] if free else palette(v)[0]
        return colour

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > _deadline - 5:
            # Not enough time; fill remaining with None to keep structure valid
            # But we need valid ints. Use a fallback greedy that always produces something.
            n, edges, lists = instance(splits, seed)
            adj = [set() for _ in range(n)]
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            colour = [0] * n
            for v in range(n):
                taken = {colour[u] for u in adj[v]}
                free = [c for c in lists[v] if c not in taken]
                colour[v] = free[0] if free else lists[v][0]
            out.append(colour)
            continue

        n, edges, lists = instance(splits, seed)
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Try greedy first (fast)
        colour = [None] * n
        order_greedy = sorted(range(n), key=lambda u: -len(adj[u]))
        for v in order_greedy:
            taken = {colour[u] for u in adj[v]}
            free = [c for c in lists[v] if c not in taken]
            colour[v] = free[0] if free else lists[v][0]
        
        if is_valid(colour, n, adj, lists):
            out.append(colour)
            continue

        # Greedy failed, try backtracking with multiple orderings
        solved = False
        # Ordering 1: degree descending
        order1 = sorted(range(n), key=lambda u: -len(adj[u]))
        colour_bt = [None] * n
        if backtrack(n, adj, lists, order1, colour_bt, 0):
            out.append(colour_bt)
            solved = True
        elif time.time() < _deadline - 5:
            # Ordering 2: degree ascending (sometimes helps with constraints)
            order2 = sorted(range(n), key=lambda u: len(adj[u]))
            colour_bt2 = [None] * n
            if backtrack(n, adj, lists, order2, colour_bt2, 0):
                out.append(colour_bt2)
                solved = True
            elif time.time() < _deadline - 5:
                # Ordering 3: random-ish (use vertex index as tiebreaker but shuffle slightly)
                # Actually, let's try a "most constrained first" heuristic:
                # vertices with smallest list size first, then degree
                order3 = sorted(range(n), key=lambda u: (len(lists[u]), -len(adj[u])))
                colour_bt3 = [None] * n
                if backtrack(n, adj, lists, order3, colour_bt3, 0):
                    out.append(colour_bt3)
                    solved = True

        if not solved:
            # Fallback: use the greedy result even if invalid (scores 0 for this instance but keeps structure valid)
            out.append(colour)

    # Ensure all are Python ints
    for i in range(len(out)):
        out[i] = [int(c) for c in out[i]]
    return out
# EVOLVE-BLOCK-END
