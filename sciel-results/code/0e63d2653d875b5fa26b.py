"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys

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

    def solve_one(n, edges, lists):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Convert to sets for fast membership tests
        adj_sets = [set(neigh) for neigh in adj]

        # Order vertices by degree (descending) for greedy
        order = sorted(range(n), key=lambda u: -len(adj[u]))

        # Backtracking with forward checking
        # State: colour[v] is int or None
        colour = [-1] * n
        # For each vertex, track available colours (from its list)
        available = [set(lst) for lst in lists]

        # We'll do a recursive backtracking search
        # To keep it efficient, we process vertices in the greedy order
        # But we need to handle the constraint that adjacent vertices differ

        # Let's use a more sophisticated approach:
        # Since the graph is bipartite and quadrangulation, it's planar
        # With lists of size 3, by Alon-Tarsi it's always colourable
        # But we need an efficient algorithm

        # Use backtracking with MRV (Minimum Remaining Values) heuristic
        # This is often very effective for list colouring

        # First, let's try a simple greedy to see if it works
        def greedy_color():
            colour_g = [-1] * n
            for v in order:
                taken = set()
                for u in adj[v]:
                    if colour_g[u] != -1:
                        taken.add(colour_g[u])
                free = [c for c in lists[v] if c not in taken]
                if free:
                    colour_g[v] = free[0]
                else:
                    return None
            return colour_g

        result = greedy_color()
        if result is not None:
            return result

        # If greedy fails, use backtracking with MRV
        # This should be fast for these small graphs

        # State: colour[v] = -1 if uncoloured, else colour
        # We'll use recursion with backtracking

        # To avoid deep recursion issues, let's use an iterative approach
        # But for n up to ~400, recursion should be fine

        sys.setrecursionlimit(10000)

        def backtrack(idx):
            if idx == n:
                return True

            v = order[idx]

            # Try each colour in the list
            for c in lists[v]:
                # Check if colour c is valid (no neighbour has colour c)
                valid = True
                for u in adj[v]:
                    if colour[u] == c:
                        valid = False
                        break

                if not valid:
                    continue

                colour[v] = c
                if backtrack(idx + 1):
                    return True
                colour[v] = -1

            return False

        if backtrack(0):
            return colour

        # This should never happen by Alon-Tarsi, but just in case
        # Return the greedy result even if invalid
        return greedy_color()

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        colouring = solve_one(n, edges, lists)
        # Ensure all values are Python ints
        out.append([int(c) for c in colouring])
    return out
# EVOLVE-BLOCK-END
