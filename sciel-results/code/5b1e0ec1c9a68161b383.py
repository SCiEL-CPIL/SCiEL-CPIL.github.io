"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from random import Random

sys.setrecursionlimit(10000)

def construct():
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
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        # Convert to tuple for faster iteration if needed, but list is fine
        return adj

    def solve_csp(n, adj, lists, deadline, rng=None):
        """
        CSP Solver with Bitmask Domains, MRV, and Forward Checking.
        """
        start_time = time.time()
        
        # Initialize domains as bitmasks
        # Bit i is set if color i is in the domain
        domains = [0] * n
        for v in range(n):
            mask = 0
            for c in lists[v]:
                mask |= (1 << c)
            domains[v] = mask
        
        uncoloured = set(range(n))
        colour = [None] * n
        
        # Precompute neighbor lists for fast access
        # adj is already a list of lists
        
        def get_mrv_vertex():
            best_v = -1
            best_count = float('inf')
            candidates = []
            for v in uncoloured:
                # Count bits in domains[v]
                cnt = bin(domains[v]).count('1')
                if cnt < best_count:
                    best_count = cnt
                    best_v = v
                    candidates = [v]
                elif cnt == best_count:
                    candidates.append(v)
            
            if not candidates:
                return -1
            
            if len(candidates) == 1:
                return candidates[0]
            
            if rng:
                return rng.choice(candidates)
            else:
                return min(candidates)

        def propagate(v, c):
            """
            Assign v=c, remove c from neighbours' domains.
            Returns list of (neighbor, old_domain) for undo, or None if failure.
            """
            removed = []
            bit_c = 1 << c
            for u in adj[v]:
                if u in uncoloured:
                    if domains[u] & bit_c:
                        old_domain = domains[u]
                        domains[u] &= ~bit_c
                        removed.append((u, old_domain))
                        if domains[u] == 0:
                            # Failure: undo changes
                            for uu, od in removed:
                                domains[uu] = od
                            return None
            return removed

        def backtrack():
            if time.time() > deadline:
                return None  # Timeout
            
            if not uncoloured:
                return True
            
            v = get_mrv_vertex()
            if v == -1:
                return True
            
            if domains[v] == 0:
                return False
            
            # Get list of colors in domain
            # We want to try colors that constrain neighbors the least (LCV)
            # But for speed, we can just iterate through bits.
            # LCV is expensive to compute for every step in Python.
            # Given the small domain size (<=3), MRV is usually sufficient.
            # Let's stick to MRV + simple iteration for speed.
            
            # Extract colors
            colors = []
            mask = domains[v]
            c = 0
            while mask:
                if mask & 1:
                    colors.append(c)
                mask >>= 1
                c += 1
            
            # Optional: LCV ordering
            # Sort colors by how many neighbors they eliminate
            def lcv_score(c):
                bit_c = 1 << c
                count = 0
                for u in adj[v]:
                    if u in uncoloured and (domains[u] & bit_c):
                        count += 1
                return count
            
            colors.sort(key=lambda c: lcv_score(c))
            
            for c in colors:
                if time.time() > deadline:
                    return None
                
                uncoloured.discard(v)
                colour[v] = c
                removed = propagate(v, c)
                
                if removed is None:
                    # Propagation failed
                    uncoloured.add(v)
                    colour[v] = None
                    continue
                
                result = backtrack()
                if result is True:
                    return True
                elif result is None:
                    # Timeout
                    for u, od in removed:
                        domains[u] = od
                    uncoloured.add(v)
                    colour[v] = None
                    return None
                
                # False, undo
                for u, od in removed:
                    domains[u] = od
                uncoloured.add(v)
                colour[v] = None
            
            return False

        result = backtrack()
        if result is True:
            return colour
        else:
            return None

    def greedy_fallback(n, adj, lists):
        """
        Simple greedy coloring to ensure we return a list of correct length.
        """
        colour = [0] * n
        order = sorted(range(n), key=lambda u: -len(adj[u]))
        for v in order:
            taken = 0
            for u in adj[v]:
                taken |= (1 << colour[u])
            
            # Find a color in lists[v] not in taken
            free_color = None
            for c in lists[v]:
                if not (taken & (1 << c)):
                    free_color = c
                    break
            
            if free_color is None:
                free_color = lists[v][0]
            
            colour[v] = free_color
        return colour

    def is_valid(n, adj, lists, colour):
        if len(colour) != n:
            return False
        for v in range(n):
            if colour[v] not in lists[v]:
                return False
            for u in adj[v]:
                if u > v and colour[u] == colour[v]:
                    return False
        return True

    out = []
    total_deadline = time.time() + 140.0
    
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        current_time = time.time()
        if current_time > total_deadline - 5:
            n, edges, lists = instance(splits, seed)
            adj = build_adj(n, edges)
            out.append(greedy_fallback(n, adj, lists))
            continue
        
        n, edges, lists = instance(splits, seed)
        adj = build_adj(n, edges)
        
        inst_deadline = min(total_deadline - 5, current_time + 30)
        
        solution = None
        
        # Try 1: Deterministic CSP
        solution = solve_csp(n, adj, lists, inst_deadline, rng=None)
        
        if solution is None:
            # Try 2: Randomized restarts
            rng = Random(seed + 12345)
            restart_deadline = min(total_deadline - 5, time.time() + 10)
            solution = solve_csp(n, adj, lists, restart_deadline, rng=rng)
            
            if solution is None:
                rng2 = Random(seed + 54321)
                restart_deadline2 = min(total_deadline - 5, time.time() + 5)
                solution = solve_csp(n, adj, lists, restart_deadline2, rng=rng2)
        
        if solution is not None:
            if is_valid(n, adj, lists, solution):
                out.append(solution)
            else:
                out.append(greedy_fallback(n, adj, lists))
        else:
            out.append(greedy_fallback(n, adj, lists))

    final_out = []
    for lst in out:
        final_out.append([int(c) for c in lst])
    
    return final_out

# EVOLVE-BLOCK-END
