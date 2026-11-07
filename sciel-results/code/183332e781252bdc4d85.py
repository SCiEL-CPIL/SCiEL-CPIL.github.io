"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
sys.setrecursionlimit(100000)

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

    def build_adj(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def solve_csp(n, adj, lists, deadline):
        # Bitmask domains: bit i set means color i is available
        # Colors are 0..4
        domains = [0] * n
        for i, l in enumerate(lists):
            mask = 0
            for c in l:
                mask |= (1 << c)
            domains[i] = mask
        
        uncoloured = list(range(n))
        uncoloured_set = set(uncoloured)
        colour = [None] * n
        original_domains = domains[:]

        def mrv_vertex():
            best = None
            best_size = 100
            for v in uncoloured:
                # Count bits
                s = bin(domains[v]).count('1')
                if s < best_size:
                    best_size = s
                    best = v
                    if s <= 1:
                        break
            return best

        def propagate(v, c):
            # assign v=c, remove c from neighbours' domains
            removed = []
            bit = 1 << c
            for u in adj[v]:
                if u in uncoloured_set:
                    if domains[u] & bit:
                        domains[u] ^= bit
                        removed.append(u)
                        if domains[u] == 0:
                            return None  # failure
            return removed

        def backtrack():
            if time.time() > deadline:
                return False
            if not uncoloured:
                return True
            
            # Find MRV vertex
            v = None
            min_size = 100
            for u in uncoloured:
                s = bin(domains[u]).count('1')
                if s < min_size:
                    min_size = s
                    v = u
                    if s == 0:
                        return False
                    if s == 1:
                        break
            
            if v is None:
                return True
            
            # Get available colors
            d = domains[v]
            # Iterate through set bits
            # Optimization: LCV is hard with bitmasks without extra work, so we just iterate
            # We can try colors in order of least constraining value by checking neighbor domains
            # But for speed, simple iteration is often sufficient for small domains
            colors = []
            tmp = d
            while tmp:
                lsb = tmp & -tmp
                c = lsb.bit_length() - 1
                colors.append(c)
                tmp ^= lsb
            
            # Sort by least constraining value (heuristic)
            # This is expensive, so we only do it if the domain size > 1?
            # For 3 colors, it's cheap enough.
            if len(colors) > 1:
                colors.sort(key=lambda c: sum(1 for u in adj[v] if u in uncoloured_set and (domains[u] & (1 << c))))

            for c in colors:
                bit = 1 << c
                if not (domains[v] & bit):
                    continue
                
                # Assign
                uncoloured_set.discard(v)
                uncoloured.remove(v)
                colour[v] = c
                domains[v] = bit # Only c is available now (effectively)
                
                removed = propagate(v, c)
                if removed is None:
                    # Undo
                    domains[v] = original_domains[v] # Wait, we need to restore the state before this choice
                    # Actually, we should save the state of v's domain before modifying?
                    # No, domains[v] is modified. We need to restore it.
                    # But we didn't save it. Let's save it before the loop.
                    pass # This logic is flawed. Let's use a cleaner state management.
                
                # Let's restart the backtrack implementation with proper state saving.
                # This edit is getting complex. Let's stick to the simpler Set-based one but optimize the MRV loop.
                # The Set-based one is likely fast enough for N~400.
                return False # Placeholder to prevent execution of this flawed draft

        # Re-implementing solve_csp with Set-based but optimized MRV
        domains = [set(l) for l in lists]
        uncoloured = set(range(n))
        colour = [None] * n
        orig_domains = [set(l) for l in lists]

        def mrv_vertex():
            best = None
            best_size = 100
            for v in uncoloured:
                s = len(domains[v])
                if s < best_size:
                    best_size = s
                    best = v
                    if s <= 1:
                        break
            return best

        def propagate(v, c):
            removed = []
            for u in adj[v]:
                if u in uncoloured and c in domains[u]:
                    domains[u].discard(c)
                    removed.append((u, c))
                    if not domains[u]:
                        return None
            return removed

        def backtrack():
            if time.time() > deadline:
                return False
            if not uncoloured:
                return True
            
            v = mrv_vertex()
            if v is None:
                return True
            
            # LCV Heuristic
            colors = list(domains[v])
            if len(colors) > 1:
                colors.sort(key=lambda c: sum(1 for u in adj[v] if u in uncoloured and c in domains[u]))
            
            for c in colors:
                uncoloured.discard(v)
                colour[v] = c
                removed = propagate(v, c)
                if removed is None:
                    # Undo
                    for u, rc in removed:
                        domains[u].add(rc)
                    uncoloured.add(v)
                    colour[v] = None
                    continue
                if backtrack():
                    return True
                # Undo
                for u, rc in removed:
                    domains[u].add(rc)
                uncoloured.add(v)
                colour[v] = None
            return False

        if backtrack():
            return colour
        return None

    def greedy(n, adj, lists):
        colour = [None] * n
        # Degree ordering
        order = sorted(range(n), key=lambda u: -len(adj[u]))
        for v in order:
            taken = {colour[u] for u in adj[v] if colour[u] is not None}
            free = [c for c in lists[v] if c not in taken]
            colour[v] = free[0] if free else lists[v][0]
        return colour

    start = time.time()
    deadline = start + 140.0

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        adj = build_adj(n, edges)

        if time.time() > deadline:
            out.append(greedy(n, adj, lists))
            continue

        result = solve_csp(n, adj, lists, deadline)
        if result is not None:
            valid = True
            for v in range(n):
                if result[v] not in lists[v]:
                    valid = False
                    break
                for u in adj[v]:
                    if u > v and result[u] == result[v]:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                out.append(result)
                continue

        out.append(greedy(n, adj, lists))

    return [[int(c) for c in lst] for lst in out]

# EVOLVE-BLOCK-END
