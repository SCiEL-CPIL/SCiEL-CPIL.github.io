"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

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
        """
        Solve list coloring using AC-3 arc consistency + backtracking with forward checking.
        Returns a list of colors (int 0..4) per vertex, or None if failed.
        """
        # Build adjacency
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Domain for each vertex
        domains = [list(l) for l in lists]

        # AC-3 arc consistency
        def ac3(domains):
            # Queue of arcs (i, j) meaning "check if domain[i] is consistent with domain[j]"
            queue = []
            for i in range(n):
                for j in adj[i]:
                    queue.append((i, j))

            while queue:
                i, j = queue.pop()
                if revise(domains, i, j):
                    if not domains[i]:
                        return False
                    for k in adj[i]:
                        if k != j:
                            queue.append((k, i))
            return True

        def revise(domains, i, j):
            """Remove values from domains[i] that have no support in domains[j]."""
            revised = False
            to_remove = set()
            for val_i in domains[i]:
                if not any(val_i != val_j for val_j in domains[j]):
                    to_remove.add(val_i)
            if to_remove:
                domains[i] = [v for v in domains[i] if v not in to_remove]
                revised = True
            return revised

        if not ac3(domains):
            return None

        # Backtracking with forward checking
        # Order vertices by most-constrained (smallest domain first), then by degree
        def get_order():
            # Use dynamic ordering: pick variable with smallest current domain, break ties by degree
            # But for simplicity in recursion, we'll use static ordering based on initial domain sizes
            # Actually, let's use a static ordering: sort by (len(domain), -degree)
            order = sorted(range(n), key=lambda v: (len(domains[v]), -len(adj[v])))
            return order

        order = get_order()
        color = [None] * n

        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == len(order):
                return True

            v = order[idx]
            # Try each value in domain[v]
            for val in domains[v]:
                # Check consistency with already-colored neighbors
                consistent = True
                for u in adj[v]:
                    if color[u] is not None and color[u] == val:
                        consistent = False
                        break
                if not consistent:
                    continue

                # Forward checking: check if all uncolored neighbors still have support
                fc_ok = True
                saved_domains = {}
                for u in adj[v]:
                    if color[u] is None:
                        saved_domains[u] = list(domains[u])
                        new_dom = [x for x in domains[u] if x != val]
                        if not new_dom:
                            fc_ok = False
                            break
                        domains[u] = new_dom

                if not fc_ok:
                    # Restore
                    for u, dom in saved_domains.items():
                        domains[u] = dom
                    continue

                color[v] = val
                if backtrack(idx + 1):
                    return True
                color[v] = None

                # Restore domains
                for u, dom in saved_domains.items():
                    domains[u] = dom

            return False

        # Make a copy of domains for backtracking (since we modify them)
        domains_backup = [list(d) for d in domains]
        # Reset domains for backtracking
        domains = [list(d) for d in domains_backup]

        if backtrack(0):
            return color
        return None

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline:
            # Not enough time; return what we have so far with None for remaining
            # But we need valid output, so break
            break
        n, edges, lists = instance(splits, seed)
        result = solve_list_coloring(n, edges, lists)
        if result is None:
            # Should not happen by Alon-Tarsi, but fallback to greedy
            result = greedy_fallback(n, edges, lists)
        out.append(result)

    # If we didn't solve all 4, fill in with greedy fallback for missing ones
    while len(out) < 4:
        idx = len(out)
        splits, seed = ((20, 701), (40, 702), (60, 703), (100, 704))[idx]
        n, edges, lists = instance(splits, seed)
        result = greedy_fallback(n, edges, lists)
        out.append(result)

    # Ensure all values are Python ints
    final_out = []
    for coloring in out:
        final_out.append([int(c) for c in coloring])

    return final_out


def greedy_fallback(n, edges, lists):
    """Simple greedy fallback."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    colour = [None] * n
    for v in sorted(range(n), key=lambda u: -len(adj[u])):
        taken = {colour[u] for u in adj[v] if colour[u] is not None}
        free = [c for c in lists[v] if c not in taken]
        colour[v] = free[0] if free else lists[v][0]
    return colour
# EVOLVE-BLOCK-END
