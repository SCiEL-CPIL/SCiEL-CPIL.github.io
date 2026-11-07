"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

def construct():
    from random import Random
    import sys

    start_time = time.time()
    DEADLINE = 140.0

    def time_remaining():
        return DEADLINE - (time.time() - start_time)

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

    def solve_instance(n, edges, lists):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Bitmask domains
        # colors are 0..4, so we need 5 bits.
        # domain[v] is an integer where bit c is 1 if color c is in domain.
        domains = [0] * n
        for v in range(n):
            mask = 0
            for c in lists[v]:
                mask |= (1 << c)
            domains[v] = mask

        # AC-3 Initialization
        # Queue of arcs (u, v) where u's domain must be checked against v's domain.
        # We need to remove colors from domains[u] that have no support in domains[v].
        
        # Optimization: Since it's bipartite and small, we can just run AC-3 once.
        # For each u, for each c in domain[u], check if there exists c' in domain[v] such that c != c'.
        # Actually, for graph coloring, the constraint is c_u != c_v.
        # Support for (u, c) exists in (v, D_v) if D_v contains any color != c.
        # This is true if |D_v| > 1 or (|D_v| == 1 and D_v's only color != c).
        # Wait, if D_v = {c}, then c has no support in v.
        
        queue = deque()
        for u in range(n):
            for v in adj[u]:
                queue.append((u, v))
        
        def revise(u, v):
            # Remove values from domains[u] that have no support in domains[v]
            # Constraint: color[u] != color[v]
            # Value c in domains[u] is unsupported if domains[v] is a subset of {c}
            # i.e., if domains[v] has only c, or is empty (but we check empty separately).
            # More generally: c is removed if (domains[v] & ~(1 << c)) == 0 AND (domains[v] & (1 << c)) != 0
            # Simpler: c is removed if domains[v] is exactly (1 << c) or 0? No, if domains[v] is 0, the instance is unsat.
            # If domains[v] contains ONLY c, then u cannot be c.
            # So, if domains[v] & ~(1 << c) == 0:
            #   if domains[v] != 0: # v is fixed to c
            #       remove c from domains[u]
            
            mask_v = domains[v]
            if mask_v == 0:
                return True # u is affected, but v is dead. Actually if v is dead, u is dead too eventually.
            
            # For each c in domains[u]:
            # if (mask_v & ~(1 << c)) == 0:
            #     # mask_v is subset of {c}
            #     # if mask_v has c, then v=c, so u!=c
            #     # if mask_v is 0, handled above
            #     domains[u] &= ~(1 << c)
            
            # We can optimize this check.
            # If mask_v has more than 1 bit, it supports all c.
            if mask_v & (mask_v - 1) != 0:
                return False # mask_v has >= 2 bits, supports all c in u
            
            # mask_v has exactly 1 bit (or is 0)
            if mask_v != 0:
                # mask_v is a single color c_v
                # u cannot be c_v
                if domains[u] & mask_v:
                    domains[u] &= ~mask_v
                    return True
            
            return False

        while queue:
            u, v = queue.popleft()
            if revise(u, v):
                if domains[u] == 0:
                    return None
                for w in adj[u]:
                    if w != v:
                        queue.append((w, u))

        # Backtracking with MRV
        sys.setrecursionlimit(10000)
        colour = [-1] * n
        
        def get_mrv_vertex():
            min_count = float('inf')
            min_vertex = -1
            for v in range(n):
                if colour[v] != -1:
                    continue
                # Count bits in domains[v]
                cnt = domains[v].bit_count()
                if cnt < min_count:
                    min_count = cnt
                    min_vertex = v
                    if min_count == 0:
                        break
            return min_vertex, min_count

        def backtrack():
            if time_remaining() < 0.1:
                return False
            
            v, count = get_mrv_vertex()
            if v == -1:
                return True # All assigned
            
            if count == 0:
                return False
            
            # Try each color in domains[v]
            mask = domains[v]
            while mask:
                # Get lowest set bit
                lsb = mask & -mask
                c = lsb.bit_length() - 1
                
                # Assign c to v
                colour[v] = c
                
                # Forward Checking:
                # For each neighbor u of v:
                # Remove c from domains[u]
                # If domains[u] becomes 0, fail
                # Store original domains to restore
                
                saved_domains = {}
                valid = True
                for u in adj[v]:
                    if colour[u] != -1:
                        if colour[u] == c:
                            valid = False
                            break
                        continue
                    
                    # u is unassigned
                    if domains[u] & lsb:
                        saved_domains[u] = domains[u]
                        domains[u] &= ~lsb
                        if domains[u] == 0:
                            valid = False
                            break
                
                if valid and backtrack():
                    return True
                
                # Restore
                colour[v] = -1
                for u, dom in saved_domains.items():
                    domains[u] = dom
            
            return False
        
        if backtrack():
            # Convert to list of ints
            return [int(c) for c in colour]
        
        return None

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        res = solve_instance(n, edges, lists)
        if res is None:
            # Fallback: should not happen, but return something to avoid crash
            # Just take first color from list
            res = [lists[i][0] for i in range(n)]
        out.append(res)
        
    final_out = []
    for lst in out:
        if isinstance(lst, (list, tuple)):
            final_out.append([int(x) for x in lst])
        else:
            final_out.append(lst)
            
    return final_out
# EVOLVE-BLOCK-END
