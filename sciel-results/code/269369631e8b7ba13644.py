"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def construct():
    import random
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

    def solve_instance(n, edges, lists, deadline):
        """Solve a single list colouring instance using backtracking with constraint propagation."""
        if time.time() > deadline:
            return None

        # Build adjacency as lists
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Use iterative deepening with MRV heuristic
        # First try a simple greedy with forward checking
        def greedy_with_fc():
            """Greedy assignment with forward checking, trying multiple orderings."""
            # Try multiple orderings
            orderings = [
                sorted(range(n), key=lambda v: len(adj[v])),  # min degree first
                sorted(range(n), key=lambda v: -len(adj[v])),  # max degree first
                list(range(n)),  # natural order
            ]

            for order in orderings:
                if time.time() > deadline:
                    return None

                color = [-1] * n
                domains = [list(lst) for lst in lists]

                # Forward checking backtracking
                def backtrack(idx):
                    if time.time() > deadline:
                        return False
                    if idx == n:
                        return True

                    v = order[idx]
                    # Try each colour in the domain
                    for c in list(domains[v]):
                        # Check if colour is valid for all uncoloured neighbours
                        valid = True
                        for u in adj[v]:
                            if color[u] == c:
                                valid = False
                                break
                        if not valid:
                            continue

                        # Assign
                        color[v] = c
                        # Remove c from domains of uncoloured neighbours
                        saved = {}
                        conflict = False
                        for u in adj[v]:
                            if color[u] == -1:
                                if c in domains[u]:
                                    saved[u] = domains[u].copy()
                                    domains[u].remove(c)
                                    if not domains[u]:
                                        conflict = True
                                        break
                        else:
                            # Check coloured neighbours
                            for u in adj[v]:
                                if color[u] != -1 and color[u] == c:
                                    conflict = True
                                    break

                        if conflict:
                            for uu, dd in saved.items():
                                domains[uu] = dd
                            color[v] = -1
                            continue

                        if backtrack(idx + 1):
                            return True

                        # Restore
                        for uu, dd in saved.items():
                            domains[uu] = dd
                        color[v] = -1

                    return False

                # Reset state
                color = [-1] * n
                domains = [list(lst) for lst in lists]

                if backtrack(0):
                    return color

            return None

        result = greedy_with_fc()
        if result is not None:
            return result

        # Fallback: more aggressive backtracking with MRV
        def backtrack_mrv():
            """Backtracking with MRV (Minimum Remaining Values) heuristic."""
            if time.time() > deadline:
                return None

            color = [-1] * n
            domains = [list(lst) for lst in lists]

            def find_mrv_vertex():
                """Find uncoloured vertex with smallest domain."""
                min_size = float('inf')
                best_v = -1
                for v in range(n):
                    if color[v] == -1:
                        if len(domains[v]) < min_size:
                            min_size = len(domains[v])
                            best_v = v
                            if min_size <= 1:
                                break
                return best_v

            def backtrack():
                if time.time() > deadline:
                    return False

                v = find_mrv_vertex()
                if v == -1:
                    return True  # All vertices coloured

                # Try each colour in the domain
                for c in list(domains[v]):
                    # Check if colour is valid for all coloured neighbours
                    valid = True
                    for u in adj[v]:
                        if color[u] == c:
                            valid = False
                            break
                    if not valid:
                        continue

                    # Assign
                    color[v] = c
                    # Remove c from domains of uncoloured neighbours
                    saved = {}
                    conflict = False
                    for u in adj[v]:
                        if color[u] == -1:
                            if c in domains[u]:
                                saved[u] = domains[u].copy()
                                domains[u].remove(c)
                                if not domains[u]:
                                    conflict = True
                                    break

                    if conflict:
                        for uu, dd in saved.items():
                            domains[uu] = dd
                        color[v] = -1
                        continue

                    if backtrack():
                        return True

                    # Restore
                    for uu, dd in saved.items():
                        domains[uu] = dd
                    color[v] = -1

                return False

            if backtrack():
                return color
            return None

        result = backtrack_mrv()
        if result is not None:
            return result

        # Final fallback: greedy without forward checking
        adj_set = defaultdict(set)
        for u, v in edges:
            adj_set[u].add(v)
            adj_set[v].add(u)
        colour = [None] * n
        for v in sorted(range(n), key=lambda u: -len(adj_set[u])):
            taken = {colour[u] for u in adj_set[v] if colour[u] is not None}
            free = [c for c in lists[v] if c not in taken]
            colour[v] = free[0] if free else lists[v][0]
        return colour

    deadline = time.time() + 110  # 110 second safety margin (150s limit)

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        result = solve_instance(n, edges, lists, deadline)
        if result is not None:
            # Ensure all values are Python int
            out.append([int(c) for c in result])
        else:
            # Should not happen, but add a dummy valid coloring
            adj = defaultdict(set)
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
