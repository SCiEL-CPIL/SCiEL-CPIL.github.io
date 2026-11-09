"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    from random import Random
    import sys
    import time

    # Increase recursion limit for backtracking
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(10000)

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

    def solve_csp(n, edges, lists, deadline=None):
        """Solve list colouring via backtracking with MRV + forward checking."""
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Convert lists to bitmask (5 colours, bits 0-4)
        domain = [0] * n
        for v in range(n):
            mask = 0
            for c in lists[v]:
                mask |= (1 << c)
            domain[v] = mask

        # Colour assignment: -1 means unassigned
        colour = [-1] * n
        assigned = [False] * n
        unassigned_count = n

        # Precompute degree for tie-breaking
        degree = [len(adj[v]) for v in range(n)]

        # Precompute popcount for fast domain size
        popcount = [0] * 32  # 2^5 = 32 possible masks
        for i in range(32):
            popcount[i] = bin(i).count('1')

        def pick_next():
            """MRV: pick unassigned vertex with fewest available colours; tie-break by degree."""
            best_v = -1
            best_count = 999
            best_deg = -1
            for v in range(n):
                if not assigned[v]:
                    cnt = popcount[domain[v]]
                    if cnt == 0:
                        return -1  # dead end
                    if cnt < best_count or (cnt == best_count and degree[v] > best_deg):
                        best_count = cnt
                        best_deg = degree[v]
                        best_v = v
            return best_v

        # Recursive backtracking with forward checking
        def backtrack():
            nonlocal unassigned_count
            if deadline is not None and time.time() > deadline:
                return False
            
            if unassigned_count == 0:
                return True
            
            v = pick_next()
            if v == -1:
                return False
            
            mask = domain[v]
            while mask:
                bit = mask & (-mask)
                c = bit.bit_length() - 1
                mask -= bit
                
                # Try colour c
                colour[v] = c
                assigned[v] = True
                unassigned_count -= 1
                
                # Forward checking
                saved = []
                ok = True
                for nb in adj[v]:
                    if not assigned[nb]:
                        old_dom = domain[nb]
                        saved.append((nb, old_dom))
                        if old_dom & bit:
                            new_domain = old_dom & ~bit
                            domain[nb] = new_domain
                            if new_domain == 0:
                                ok = False
                                break
                
                if ok and backtrack():
                    return True
                
                # Undo
                for nb, old_dom in saved:
                    domain[nb] = old_dom
                assigned[v] = False
                unassigned_count += 1
                colour[v] = -1
            
            return False

        if backtrack():
            return [int(c) for c in colour]
        return None

    out = []
    deadline = time.time() + 145.0  # safety margin before 150s kill
    instances = ((20, 701), (40, 702), (60, 703), (100, 704))

    for splits, seed in instances:
        if time.time() > deadline:
            # Not enough time, add empty solutions for remaining
            while len(out) < 4:
                out.append([])
            break
        
        n, edges, lists = instance(splits, seed)
        sol = solve_csp(n, edges, lists, deadline)
        if sol is None:
            # Fallback: should not happen per Alon-Tarsi, but guard
            # Return a list of correct length with default values
            sol = [0] * n
        out.append(sol)

    sys.setrecursionlimit(old_limit)

    # Ensure types are correct: list of lists of Python ints
    # If we didn't complete all 4 instances, we need to handle them
    # But we already appended fallbacks, so out should have 4 entries
    # Each entry should have the correct length for its instance
    
    # Recompute n for each instance to ensure correct length
    final_out = []
    for i, (splits, seed) in enumerate(instances):
        if i < len(out):
            n, edges, lists = instance(splits, seed)
            coloring = out[i]
            if len(coloring) != n:
                # Pad or truncate to correct length
                if len(coloring) < n:
                    coloring = coloring + [0] * (n - len(coloring))
                else:
                    coloring = coloring[:n]
            final_out.append([int(c) for c in coloring])
        else:
            # Should not happen, but guard
            n, edges, lists = instance(splits, seed)
            final_out.append([0] * n)
    return final_out
# EVOLVE-BLOCK-END
