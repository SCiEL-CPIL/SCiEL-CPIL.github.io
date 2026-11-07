"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

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

    def solve_coloring(n, edges, lists):
        """Solve a 3-list coloring of a bipartite graph using backtracking with
        forward checking and MRV heuristic. Returns a list of colours or None."""
        # Build adjacency
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Try greedy first for speed
        def try_greedy(order):
            colour = [None] * n
            for v in order:
                taken = set()
                for u in adj[v]:
                    if colour[u] is not None:
                        taken.add(colour[u])
                free = [c for c in lists[v] if c not in taken]
                if not free:
                    return None
                colour[v] = free[0]
            return colour

        # Degree-based ordering (descending)
        deg_order = sorted(range(n), key=lambda u: -len(adj[u]))
        result = try_greedy(deg_order)
        if result is not None:
            return result

        # Try multiple greedy orderings
        import random
        rng = Random(42)
        for _ in range(50):
            shuffled = list(range(n))
            # Sort by degree with some randomness
            shuffled.sort(key=lambda u: (-len(adj[u]), rng.random()))
            result = try_greedy(shuffled)
            if result is not None:
                return result

        # Backtracking with MRV and forward checking
        def backtrack():
            # Find uncolored vertex with fewest available colours (MRV)
            best_v = -1
            best_opts = None
            for v in range(n):
                if colour[v] is None:
                    taken = set()
                    for u in adj[v]:
                        if colour[u] is not None:
                            taken.add(colour[u])
                    opts = [c for c in lists[v] if c not in taken]
                    if len(opts) < len(best_opts) if best_opts is not None else True:
                        if best_opts is None or len(opts) < len(best_opts):
                            best_v = v
                            best_opts = opts
                            if len(opts) <= 1:
                                break
                    if len(opts) == 0:
                        return False
            if best_v == -1:
                return True  # All coloured
            
            for c in best_opts:
                colour[best_v] = c
                # Forward check: ensure all uncolored neighbors still have options
                ok = True
                for u in adj[best_v]:
                    if colour[u] is None:
                        taken = set()
                        for w in adj[u]:
                            if colour[w] is not None:
                                taken.add(colour[w])
                        if not any(col in lists[u] for col in lists[u] if col not in taken):
                            ok = False
                            break
                if ok:
                    if backtrack():
                        return True
                colour[best_v] = None
            
            return False

        colour = [None] * n
        # Initialize with a good ordering: colour high-degree vertices first
        for v in deg_order:
            taken = set()
            for u in adj[v]:
                if colour[u] is not None:
                    taken.add(colour[u])
            free = [c for c in lists[v] if c not in taken]
            if free:
                colour[v] = free[0]
        
        if backtrack():
            return colour
        return None

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > DEADLINE:
            # Not enough time; use greedy as fallback
            n, edges, lists = instance(splits, seed)
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            colour = [None] * n
            for v in sorted(range(n), key=lambda u: -len(adj[u])):
                taken = set()
                for u in adj[v]:
                    if colour[u] is not None:
                        taken.add(colour[u])
                free = [c for c in lists[v] if c not in taken]
                colour[v] = free[0] if free else lists[v][0]
            out.append(colour)
        else:
            n, edges, lists = instance(splits, seed)
            colour = solve_coloring(n, edges, lists)
            if colour is None:
                # Fallback greedy
                adj = [[] for _ in range(n)]
                for u, v in edges:
                    adj[u].append(v)
                    adj[v].append(u)
                colour = [None] * n
                for v in sorted(range(n), key=lambda u: -len(adj[u])):
                    taken = set()
                    for u in adj[v]:
                        if colour[u] is not None:
                            taken.add(colour[u])
                    free = [c for c in lists[v] if c not in taken]
                    colour[v] = free[0] if free else lists[v][0]
            out.append([int(c) for c in colour])
    
    return out
# EVOLVE-BLOCK-END
