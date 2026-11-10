"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

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

    def solve_csp(n, edges, lists):
        # Build adjacency lists
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # AC-3 arc consistency pruning
        def ac3(domains, adj):
            queue = [(i, j) for i in range(n) for j in adj[i]]
            while queue:
                i, j = queue.pop()
                if _revise(domains, i, j):
                    if not domains[i]:
                        return False
                    for k in adj[i]:
                        if k != j:
                            queue.append((k, i))
            return True

        def _revise(domains, i, j):
            removed = False
            dj = domains[j]
            for val in list(domains[i]):
                if all(val == d for d in dj):
                    domains[i].remove(val)
                    removed = True
            return removed

        # Backtracking with forward checking
        def backtrack(domains, adj, colour):
            # Find unassigned vertex with minimum remaining values (MRV)
            unassigned = [v for v in range(n) if colour[v] is None]
            if not unassigned:
                return True

            # MRV heuristic: pick vertex with fewest available colours
            v = min(unassigned, key=lambda u: len(domains[u]))
            
            # LCV heuristic: try colours that conflict least with neighbours
            for c in domains[v]:
                colour[v] = c
                # Forward checking: check if any neighbour becomes empty
                valid = True
                saved_domains = {}
                for u in adj[v]:
                    if colour[u] is None:
                        saved_domains[u] = list(domains[u])
                        if c in domains[u]:
                            domains[u].remove(c)
                        if not domains[u]:
                            valid = False
                            break
                
                if valid:
                    if backtrack(domains, adj, colour):
                        return True
                
                # Undo
                for u, saved in saved_domains.items():
                    domains[u] = saved
                colour[v] = None

            return False

        # Initialize domains
        domains = [list(l) for l in lists]
        
        # Apply AC-3 first
        if not ac3(domains, adj):
            return None

        # Check if any domain is empty after AC-3
        for d in domains:
            if not d:
                return None

        # Backtracking search
        colour = [None] * n
        if backtrack(domains, adj, colour):
            return colour
        return None

    def greedy_fallback(n, edges, lists):
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

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline:
            # If we're out of time, return what we have padded with greedy
            break
        n, edges, lists = instance(splits, seed)
        result = solve_csp(n, edges, lists)
        if result is None:
            result = greedy_fallback(n, edges, lists)
        # Ensure all values are Python ints
        result = [int(c) for c in result]
        out.append(result)

    # If we ran out of time, fill remaining with greedy
    while len(out) < 4:
        idx = len(out)
        splits, seed = ((20, 701), (40, 702), (60, 703), (100, 704))[idx]
        n, edges, lists = instance(splits, seed)
        result = greedy_fallback(n, edges, lists)
        result = [int(c) for c in result]
        out.append(result)

    return out
# EVOLVE-BLOCK-END
