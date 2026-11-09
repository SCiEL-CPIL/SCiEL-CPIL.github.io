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
        # Build adjacency as lists of neighbors for AC-3 and backtracking
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Convert lists to bitmask domains
        domains = [0] * n
        for v in range(n):
            mask = 0
            for c in lists[v]:
                mask |= (1 << c)
            domains[v] = mask
        
        # AC-3 Preprocessing
        # Queue of arcs (u, v) to check
        from collections import deque
        queue = deque()
        for u in range(n):
            for v in adj[u]:
                queue.append((u, v))
        
        def revise(u, v, doms):
            # Returns True if domain of u was reduced
            du = doms[u]
            dv = doms[v]
            # Check if any color in du is incompatible with all colors in dv
            # Since it's just "not equal", a color c in du is supported if there exists c' in dv such that c != c'
            # This is only false if dv contains ONLY c.
            # So if dv == (1 << c) and du has c, then c is removed from du.
            # More generally, if dv is a singleton {c}, then c is removed from du.
            # If dv has >1 colors, all colors in du are supported (since du has at most 3 colors, and dv has at least 2, 
            # unless du and dv are both singletons and equal, but then du would be empty after removal? 
            # Wait. If dv = {0, 1} and du = {0, 1, 2}. 
            # 0 is supported by 1. 1 is supported by 0. 2 is supported by 0. 
            # So no reduction.
            # Reduction only happens if dv is a singleton {c} and c in du.
            if dv & (dv - 1) == 0: # dv is a power of 2 (singleton)
                if du & dv:
                    new_du = du & ~dv
                    if new_du != du:
                        doms[u] = new_du
                        return True
            return False

        while queue:
            u, v = queue.popleft()
            if revise(u, v, domains):
                if domains[u] == 0:
                    return None
                for x in adj[u]:
                    if x != v:
                        queue.append((x, u))
        
        # Backtracking with MRV and Forward Checking
        assigned = [-1] * n
        unassigned = list(range(n))
        
        # Helper to get list of values from mask
        def get_vals(mask):
            vals = []
            m = mask
            while m:
                lb = m & (-m)
                idx = lb.bit_length() - 1
                vals.append(idx)
                m ^= lb
            return vals

        # Helper to get popcount
        def popcount(x):
            return bin(x).count('1')

        def backtrack(domains, assigned, unassigned):
            if not unassigned:
                return True
            
            # MRV: find vertex with min domain size
            min_count = 4 # Max list size is 3, so 4 is infinity
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
            
            # Deterministic choice: pick the first candidate (lowest index in unassigned list)
            # Or pick the one with lowest vertex index? 
            # Let's pick the one with lowest vertex index for determinism and cache locality?
            # The candidates are indices into the unassigned list.
            # Let's just pick the first one found (which is deterministic based on unassigned order).
            # To make it more robust, we can sort candidates by vertex index.
            best_idx = min(candidates, key=lambda i: unassigned[i])
            
            v = unassigned[best_idx]
            unassigned.pop(best_idx)
            
            vals = get_vals(domains[v])
            
            for val in vals:
                val_bit = 1 << val
                
                # Check consistency with assigned neighbors
                ok = True
                for u in adj[v]:
                    if assigned[u] == val:
                        ok = False
                        break
                if not ok:
                    continue
                
                assigned[v] = val
                
                # Forward checking
                saved = []
                ok_fc = True
                for u in adj[v]:
                    if assigned[u] == -1:
                        if domains[u] & val_bit:
                            saved.append((u, val_bit))
                            domains[u] &= ~val_bit
                            if domains[u] == 0:
                                ok_fc = False
                                break
                
                if ok_fc:
                    if backtrack(domains, assigned, unassigned):
                        return True
                
                # Restore
                for u, bit in saved:
                    domains[u] |= bit
                
                assigned[v] = -1
            
            unassigned.insert(best_idx, v)
            return False

        # Run backtrack
        # We use a copy of domains for the search
        search_domains = domains[:]
        if backtrack(search_domains, assigned, unassigned):
            return [assigned[i] for i in range(n)]
        
        return None

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > DEADLINE:
            # If we run out of time, we must still return a valid structure.
            # However, the contract says any broken rule makes the result invalid.
            # Returning a list of 4 lists of correct length is required for validity.
            # If we can't solve it, we can't guarantee correctness.
            # But we assume the solver is fast enough.
            pass

        n, edges, lists = instance(splits, seed)
        
        # Allocate time
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
                    col[v] = lists[v][0]
            coloring = col
            
        # Ensure types are Python int
        out.append(coloring)
        
    return out
# EVOLVE-BLOCK-END
