"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    START_TIME = time.time()
    DEADLINE = START_TIME + 115.0

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

    def solve_instance(n, edges, lists, time_limit):
        # Build adjacency as bitmasks for fast neighbor checks
        adj_mask = [0] * n
        for u, v in edges:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        
        # Convert lists to bitmask domains
        # Colors are 0..4, so we need 5 bits.
        init_domains = [0] * n
        for v in range(n):
            mask = 0
            for c in lists[v]:
                mask |= (1 << c)
            init_domains[v] = mask
        
        # Check for empty initial domains
        for v in range(n):
            if init_domains[v] == 0:
                return None
        
        # Helper to get count of bits
        def popcount(x):
            return x.bit_count() if hasattr(x, 'bit_count') else bin(x).count('1')
        
        # AC-3 with bitmasks
        def ac3(domains, adj_mask, n):
            # Queue of arcs (u, v) to check
            # We use a list as a queue for simplicity, though a deque would be faster.
            # Given small n, this is fine.
            queue = []
            for u in range(n):
                m = adj_mask[u]
                while m:
                    lb = m & (-m)
                    v = lb.bit_length() - 1
                    queue.append((u, v))
                    m ^= lb
            
            while queue:
                u, v = queue.pop()
                # Check if domain[u] and domain[v] are compatible.
                # For "not equal" constraint:
                # A value c in domain[u] is supported in domain[v] if there exists c' in domain[v] such that c' != c.
                # This fails only if domain[v] is a singleton {c}.
                # So, if domain[u] contains c, and domain[v] == (1 << c), then c is not supported.
                # We must remove all such unsupported values from domain[u].
                
                du = domains[u]
                dv = domains[v]
                
                # Find values in du that are not supported by dv.
                # A value c is unsupported if dv is exactly {c}.
                # So, unsupported = du & (dv if popcount(dv)==1 else 0)
                # Actually, if dv has size > 1, every c in du is supported (since dv has some other value).
                # If dv is singleton {c}, then c in du is unsupported.
                
                if popcount(dv) == 1:
                    # dv is a single color c
                    # If u also has c, and u has no other colors? No, we just remove c from u if it exists.
                    # Wait, the constraint is color(u) != color(v).
                    # If v is forced to be c, then u cannot be c.
                    # So we remove c from domain[u].
                    unsupported = du & dv
                    if unsupported:
                        new_du = du & ~dv
                        if new_du != du:
                            domains[u] = new_du
                            if new_du == 0:
                                return False
                            # Add neighbors of u to queue
                            m2 = adj_mask[u]
                            while m2:
                                lb = m2 & (-m2)
                                w = lb.bit_length() - 1
                                if w != v:
                                    queue.append((w, u))
                                m2 ^= lb
            return True

        # Apply AC-3
        domains = init_domains[:]
        if not ac3(domains, adj_mask, n):
            return None
        
        # Check for empty domains after AC-3
        for v in range(n):
            if domains[v] == 0:
                return None
        
        # If AC-3 reduced all domains to singletons, we have a solution.
        all_singletons = True
        for v in range(n):
            if popcount(domains[v]) > 1:
                all_singletons = False
                break
        if all_singletons:
            return [domains[v].bit_length() - 1 for v in range(n)]
        
        # Backtracking with MRV and Forward Checking
        # Use a non-recursive approach or careful recursion to avoid stack limits?
        # n is small (<100), so recursion is fine.
        
        assigned = [-1] * n
        unassigned = list(range(n))
        
        def backtrack(domains, assigned, unassigned):
            if not unassigned:
                return True
            
            # Check time
            if time.time() > time_limit:
                return False
            
            # MRV: find vertex with minimum remaining values
            min_count = 5
            best_idx = -1
            candidates = []
            for i, v in enumerate(unassigned):
                c = popcount(domains[v])
                if c < min_count:
                    min_count = c
                    candidates = [i]
                    if min_count == 0:
                        break
                elif c == min_count:
                    candidates.append(i)
                
            if min_count == 0:
                return False
            
            # Pick the vertex with minimum degree among those with min MRV
            # to maximize pruning.
            # We can just pick the first one for simplicity, or sort by degree.
            # Let's sort candidates by degree of the vertex in the *unassigned* subgraph?
            # Or just total degree. Total degree is static.
            if len(candidates) > 1:
                # Sort by degree descending to pick high-degree nodes first?
                # Or ascending? Usually, high degree nodes are more constrained.
                # Let's just pick the first one to keep it fast.
                best_idx = candidates[0]
            else:
                best_idx = candidates[0]
                
            v = unassigned[best_idx]
            unassigned.pop(best_idx)
            
            # Get values to try
            mask = domains[v]
            vals = []
            m = mask
            while m:
                lb = m & (-m)
                idx = lb.bit_length() - 1
                vals.append(idx)
                m ^= lb
            
            # Order by least constraining value
            # A value is less constraining if it removes fewer options from neighbors.
            def count_eliminated(val):
                count = 0
                m2 = adj_mask[v]
                while m2:
                    lb = m2 & (-m2)
                    u = lb.bit_length() - 1
                    if assigned[u] == -1 and domains[u] & (1 << val):
                        count += 1
                    m2 ^= lb
                return count
            
            vals.sort(key=count_eliminated)
            
            for val in vals:
                val_bit = 1 << val
                
                # Check consistency with assigned neighbors
                ok = True
                nb_mask = adj_mask[v]
                m = nb_mask
                while m:
                    lb = m & (-m)
                    u = lb.bit_length() - 1
                    if assigned[u] == val:
                        ok = False
                        break
                    m ^= lb
                if not ok:
                    continue
                
                assigned[v] = val
                
                # Forward checking
                saved = []
                ok_fc = True
                nb_mask = adj_mask[v]
                m = nb_mask
                while m:
                    lb = m & (-m)
                    u = lb.bit_length() - 1
                    if assigned[u] == -1:
                        if domains[u] & val_bit:
                            saved.append((u, val_bit))
                            domains[u] &= ~val_bit
                            if domains[u] == 0:
                                ok_fc = False
                                break
                    m ^= lb
                
                if ok_fc:
                    if backtrack(domains, assigned, unassigned):
                        return True
                
                # Restore forward checking changes
                for u, bit in saved:
                    domains[u] |= bit
                
                assigned[v] = -1
            
            unassigned.insert(best_idx, v)
            return False

        # Main solve
        if backtrack(domains, assigned, unassigned):
            return [assigned[i] for i in range(n)]
        
        return None

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > DEADLINE:
            # If we run out of time, we must still return a valid structure.
            # We will fill remaining with a greedy attempt.
            pass

        n, edges, lists = instance(splits, seed)
        
        # Allocate time per instance
        time_per_instance = 25.0
        t_start = time.time()
        
        coloring = solve_instance(n, edges, lists, t_start + time_per_instance)
        
        if coloring is None:
            # Fallback: Greedy with degree ordering
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            order = sorted(range(n), key=lambda x: -len(adj[x]))
            col = [0] * n
            for v in order:
                taken = set()
                for u in adj[v]:
                    taken.add(col[u])
                free = [c for c in lists[v] if c not in taken]
                if free:
                    col[v] = free[0]
                else:
                    # Constraint violation, pick first available from list
                    col[v] = lists[v][0]
            coloring = col
            
        # Ensure types are Python int
        # Convert to list of ints
        out.append([int(c) for c in coloring])
        
    return out
# EVOLVE-BLOCK-END
