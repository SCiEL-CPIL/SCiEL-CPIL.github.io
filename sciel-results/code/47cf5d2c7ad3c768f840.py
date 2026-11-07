"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import sys

    sys.setrecursionlimit(10000)

    def instance(splits, seed, palette=5):
        from random import Random
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

    def solve(n, edges, lists):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Use bitmask representation for domains (colors 0-4 fit in 5 bits)
        # Domain as integer bitmask: bit i set means color i is available
        domains = [0] * n
        for i, lst in enumerate(lists):
            mask = 0
            for c in lst:
                mask |= (1 << c)
            domains[i] = mask

        # Precompute neighbor masks for faster forward checking
        # For each vertex, store list of (neighbor, bit_mask_of_neighbor)
        # Actually, we just need adj and domains as bitmasks

        color = [-1] * n

        # Arc Consistency (AC-3) preprocessing
        def enforce_arc_consistency():
            """Enforce arc consistency. Returns False if a domain becomes empty."""
            queue = []
            # Initialize queue with all arcs (u, v) where u and v are adjacent
            for u in range(n):
                for v in adj[u]:
                    if u < v:  # Avoid duplicates
                        queue.append((u, v))
                        queue.append((v, u))

            while queue:
                u, v = queue.pop()
                # Remove values from domains[u] that have no support in domains[v]
                changed = False
                du = domains[u]
                dv = domains[v]
                # For each color in du, check if it has support in dv
                # Support means: there exists a color in dv different from this color
                # Since all colors in dv are different from du's color (they're different vertices),
                # support exists if dv has at least one color different from c
                # Actually: for color c in du, it has support in v if (dv & ~(1 << c)) != 0
                # i.e., dv has some color other than c
                # But since du and dv are disjoint sets of allowed colors for adjacent vertices,
                # we need: for c in du, there must exist c' in dv such that c' != c
                # This is equivalent to: dv != 0 and (if dv has only one color, that color must differ from c)
                
                # Simpler: for each color c in du, check if dv has any color != c
                # If dv == (1 << c) only, then c has no support
                # So: remove c from du if (dv & ~(1 << c)) == 0, i.e., dv is subset of {c}
                # Which means dv == 0 or dv == (1 << c)
                
                new_du = du
                c = 0
                while (1 << c) <= du:
                    if du & (1 << c):
                        # Check if c has support in v: dv must have some color != c
                        if (dv & ~(1 << c)) == 0:
                            # No support, remove c from du
                            new_du &= ~(1 << c)
                            changed = True
                    c += 1
                
                if changed:
                    domains[u] = new_du
                    if new_du == 0:
                        return False
                    # Add arcs (w, u) for all neighbors w of u except v
                    for w in adj[u]:
                        if w != v:
                            queue.append((w, u))

            return True

        # Run initial arc consistency
        if not enforce_arc_consistency():
            return [0] * n  # Should not happen per problem guarantee

        # MRV + Forward Checking Backtracking with bitmask domains
        def backtrack():
            # Find unassigned variable with Minimum Remaining Values
            min_len = 1000
            best_v = -1
            for v in range(n):
                if color[v] == -1:
                    # Count bits in domains[v]
                    d = domains[v]
                    d_len = bin(d).count('1')
                    if d_len == 0:
                        return False
                    if d_len < min_len:
                        min_len = d_len
                        best_v = v
                        if min_len == 1:
                            break
            
            if best_v == -1:
                # All assigned
                return True

            v = best_v
            saved_domain = domains[v]
            
            # Try each color in the domain
            d = saved_domain
            c = 0
            while (1 << c) <= saved_domain:
                if saved_domain & (1 << c):
                    # Assign color c to v
                    color[v] = c
                    domains[v] = (1 << c)
                    
                    # Forward checking: remove c from unassigned neighbors
                    changes = []
                    valid = True
                    
                    for u in adj[v]:
                        if color[u] == -1:
                            if domains[u] & (1 << c):
                                new_du = domains[u] & ~(1 << c)
                                changes.append((u, domains[u]))
                                domains[u] = new_du
                                if new_du == 0:
                                    valid = False
                                    break
                    
                    if valid:
                        if backtrack():
                            return True
                    
                    # Undo
                    for u, old_d in changes:
                        domains[u] = old_d
                
                c += 1
            
            domains[v] = saved_domain
            color[v] = -1
            return False

        if backtrack():
            # Ensure types are Python int
            return [int(c) for c in color]
        else:
            # Fallback, though problem guarantees solvability
            return [0] * n

    out = []
    start_time = time.time()
    deadline = start_time + 110.0

    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline:
            # Fallback to avoid crash, though unlikely
            n, edges, lists = instance(splits, seed)
            out.append([0] * n)
            continue
            
        n, edges, lists = instance(splits, seed)
        res = solve(n, edges, lists)
        out.append(res)
        
    return out
# EVOLVE-BLOCK-END
