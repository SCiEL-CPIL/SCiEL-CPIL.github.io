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

    def solve_instance(n, edges, lists):
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Convert to lists for faster iteration, and precompute degrees
        adj_list = [list(s) for s in adj]
        degree = [len(a) for a in adj_list]

        # Exact backtracking with MRV (Minimum Remaining Values) heuristic
        # Since each vertex has only 3 colours from a list of 5, and the graph
        # is planar bipartite, this should be very fast.

        colour = [-1] * n
        # We'll use a stack-based or recursive approach. Recursive is simpler.
        # To avoid recursion depth issues (n can be up to 8 + 4*100 = 408),
        # we set recursion limit or use iterative. 408 is fine for default limit (1000).

        import sys
        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(10000)

        def backtrack(idx, remaining):
            """
            idx: not really needed if we track remaining count
            remaining: number of uncoloured vertices
            Returns True if a valid colouring is found.
            """
            if remaining == 0:
                return True

            # Find the uncoloured vertex with the fewest available colours (MRV)
            best_v = -1
            best_avail = None
            best_count = float('inf')

            # Iterate over uncoloured vertices
            for v in range(n):
                if colour[v] != -1:
                    continue
                # Compute available colours for v
                taken = set()
                for u in adj_list[v]:
                    if colour[u] != -1:
                        taken.add(colour[u])
                avail = [c for c in lists[v] if c not in taken]
                cnt = len(avail)
                if cnt == 0:
                    return False  # Dead end
                if cnt < best_count:
                    best_count = cnt
                    best_v = v
                    best_avail = avail
                    if cnt == 1:
                        break  # Can't do better than 1

            # Assign colour to best_v
            for c in best_avail:
                colour[best_v] = c
                if backtrack(idx + 1, remaining - 1):
                    return True
            colour[best_v] = -1
            return False

        # Start backtracking
        success = backtrack(0, n)

        sys.setrecursionlimit(old_limit)

        if success:
            # Ensure all are Python ints
            return [int(c) for c in colour]
        else:
            # Fallback: should not happen by Alon-Tarsi, but return something
            # to avoid crashing. Return list of 0s (will likely fail validation
            # for some edges but at least type-correct).
            return [0] * n

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        colouring = solve_instance(n, edges, lists)
        out.append(colouring)

    return out
# EVOLVE-BLOCK-END
