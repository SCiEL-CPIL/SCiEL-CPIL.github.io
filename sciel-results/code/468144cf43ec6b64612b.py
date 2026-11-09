# EVOLVE-BLOCK-START
import time

def construct():
    from random import Random
    import sys
    sys.setrecursionlimit(100000)

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
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Colour assignment: -1 means uncoloured
        colour = [-1] * n
        # Domains: list of available colours per vertex (from its list)
        # Using lists for mutable domains for forward checking
        domains = [list(lists[v]) for v in range(n)]

        # Uncoloured vertices
        uncoloured = set(range(n))
        
        # Deadline guard
        start_time = time.time()
        deadline = start_time + 110.0  # 10s margin from 120s limit

        def backtrack():
            nonlocal uncoloured, colour, domains, deadline
            if time.time() > deadline:
                return False

            if not uncoloured:
                return True

            # MRV: pick vertex with fewest available colours
            best_v = None
            best_count = 10
            # We need to consider the current domain size, but also filter by neighbors' colors
            # Actually, with forward checking, domains are already pruned.
            # But we must also ensure the color is not used by a colored neighbor.
            # If we do forward checking properly, domains of uncolored vertices are already
            # pruned of colors used by colored neighbors.
            
            # Let's verify: when we assign v, we prune neighbors.
            # So domains[u] for uncolored u should not contain colors of colored neighbors.
            # However, if we just maintain domains, we don't need to re-filter by `taken` in MRV.
            # We just pick the smallest domain.
            
            for v in uncoloured:
                cnt = len(domains[v])
                if cnt == 0:
                    return False
                if cnt < best_count:
                    best_count = cnt
                    best_v = v
                    if cnt == 1:
                        break

            if best_v is None:
                return False

            v = best_v
            # Save state for backtracking
            # We need to restore domains of neighbors
            
            # Try each available colour
            for c in domains[v]:
                # Check if c is valid (not in colored neighbors)
                # With forward checking, this should be true if domains are maintained correctly.
                # But let's be safe.
                valid = True
                for u in adj[v]:
                    if colour[u] == c:
                        valid = False
                        break
                if not valid:
                    continue

                colour[v] = c
                uncoloured.remove(v)
                
                # Forward checking: prune neighbors
                saved_domains = {}
                consistent = True
                for u in adj[v]:
                    if u in uncoloured:
                        # Save current domain
                        saved_domains[u] = domains[u][:]
                        # Prune c from u's domain
                        if c in domains[u]:
                            domains[u].remove(c)
                            if not domains[u]:
                                consistent = False
                                break
                
                if consistent and backtrack():
                    return True
                
                # Backtrack: restore
                uncoloured.add(v)
                colour[v] = -1
                for u, dom in saved_domains.items():
                    domains[u] = dom

            return False

        if backtrack():
            return [colour[v] for v in range(n)]
        else:
            # Should not happen per Alon-Tarsi, but return something
            # If timeout or failure, return a dummy valid-looking structure to avoid crash,
            # though it will score 0 for those instances.
            # The task says "An instance whose colouring breaks a list or an edge scores 0... but a wrong length or type makes the whole result invalid."
            # So we must return a list of ints.
            return [0] * n

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        out.append(solve_csp(n, edges, lists))

    # Ensure types are correct: Python int, list of lists
    result = []
    for colouring in out:
        result.append([int(c) for c in colouring])
    return result
# EVOLVE-BLOCK-END
