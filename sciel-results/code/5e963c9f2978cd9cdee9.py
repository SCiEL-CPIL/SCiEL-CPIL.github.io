"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    # Set recursion limit high enough for the largest instance (n=408)
    # Default is usually 1000, which is fine, but let's be safe.
    sys.setrecursionlimit(5000)
    
    _deadline = time.time() + 140  # Safety margin before 150s kill

    from random import Random

    def instance(splits, seed, palette=5):
        rng = Random(seed)
        # Cube faces
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
        # Use a small set for taken colors
        taken = set()
        for u in adj[v]:
            cu = colour[u]
            if cu is not None:
                taken.add(cu)
        
        # Try colors from the list
        for c in lists[v]:
            if c not in taken:
                colour[v] = c
                if backtrack(n, adj, lists, order, colour, idx + 1):
                    return True
                colour[v] = None
        return False

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        # Check time
        if time.time() > _deadline - 2:
            # Fallback: generate instance and return a dummy valid-looking structure
            # This shouldn't happen if the solver is fast, but prevents crash
            n, edges, lists = instance(splits, seed)
            # Just return a list of 0s (invalid coloring, but valid type/length)
            # The score for this instance will be 0, but the program won't crash.
            out.append([0] * n)
            continue

        n, edges, lists = instance(splits, seed)
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # 1. Try Greedy (Degree Descending)
        colour = [None] * n
        order_greedy = sorted(range(n), key=lambda u: -len(adj[u]))
        greedy_success = True
        for v in order_greedy:
            taken = {colour[u] for u in adj[v] if colour[u] is not None}
            free = [c for c in lists[v] if c not in taken]
            if free:
                colour[v] = free[0]
            else:
                greedy_success = False
                break
        
        if greedy_success and is_valid(colour, n, adj, lists):
            out.append(colour)
            continue

        # 2. Backtracking with "Most Constrained First" heuristic
        # Sort by list size (ascending), then degree (descending)
        # This is often better for list coloring than just degree
        order_bt = sorted(range(n), key=lambda u: (len(lists[u]), -len(adj[u])))
        colour_bt = [None] * n
        
        if backtrack(n, adj, lists, order_bt, colour_bt, 0):
            out.append(colour_bt)
            continue

        # 3. If backtracking failed (should be rare/impossible for these params within time),
        #    try a different ordering just in case, with a time check
        if time.time() < _deadline - 2:
            order_bt2 = sorted(range(n), key=lambda u: -len(adj[u]))
            colour_bt2 = [None] * n
            if backtrack(n, adj, lists, order_bt2, colour_bt2, 0):
                out.append(colour_bt2)
                continue

        # 4. Fallback: Return the greedy attempt (might be invalid, scores 0 for this instance)
        #    Or a simple 0-fill if greedy didn't complete
        if not greedy_success:
            # Complete the greedy coloring arbitrarily for structure
            for v in order_greedy:
                if colour[v] is None:
                    colour[v] = lists[v][0]
        out.append(colour)

    # Ensure all are Python ints
    for i in range(len(out)):
        out[i] = [int(c) for c in out[i]]
    return out
# EVOLVE-BLOCK-END
