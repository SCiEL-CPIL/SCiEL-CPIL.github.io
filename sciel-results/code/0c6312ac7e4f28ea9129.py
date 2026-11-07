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
        domains = [0] * n
        for i, lst in enumerate(lists):
            mask = 0
            for c in lst:
                mask |= (1 << c)
            domains[i] = mask

        color = [-1] * n

        # Precompute bit counts for fast MRV
        bit_count = [0] * 32
        for i in range(32):
            bit_count[i] = bin(i).count('1')

        # AC-3 implementation for maintaining arc consistency
        def enforce_ac3(initial_queue=None):
            """
            Enforces arc consistency. 
            If initial_queue is None, initializes with all arcs.
            Returns False if a domain becomes empty.
            Modifies domains in place.
            """
            if initial_queue is None:
                queue = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            queue.append((u, v))
                            queue.append((v, u))
            else:
                queue = list(initial_queue)

            while queue:
                u, v = queue.pop()
                du = domains[u]
                dv = domains[v]
                new_du = du
                # Check if any value in du has no support in dv
                # A value c in du has no support in dv if dv is a subset of {c} (i.e., dv & ~{c} == 0)
                # Since dv is a bitmask, this means dv == 0 or dv == (1 << c)
                # We iterate through bits of du
                temp = du
                while temp:
                    lsb = temp & -temp
                    c = lsb.bit_length() - 1
                    if (dv & ~lsb) == 0:
                        new_du &= ~lsb
                    temp ^= lsb
                
                if new_du != du:
                    domains[u] = new_du
                    if new_du == 0:
                        return False
                    for w in adj[u]:
                        if w != v:
                            queue.append((w, u))

            return True

        # Initial AC-3
        if not enforce_ac3():
            return [0] * n

        # Backtracking with MAC (Maintaining Arc Consistency)
        def backtrack():
            # MRV: Select variable with minimum remaining values
            min_len = 1000
            best_v = -1
            for v in range(n):
                if color[v] == -1:
                    d_len = bit_count[domains[v]]
                    if d_len == 0:
                        return False
                    if d_len < min_len:
                        min_len = d_len
                        best_v = v
                        if min_len == 1:
                            break
            
            if best_v == -1:
                return True

            v = best_v
            saved_domain = domains[v]
            
            # Get list of available colors
            available_colors = []
            temp = saved_domain
            while temp:
                lsb = temp & -temp
                c = lsb.bit_length() - 1
                available_colors.append(c)
                temp ^= lsb
            
            # LCV: Sort colors by number of values removed from neighbors (ascending)
            def count_removed(c):
                count = 0
                for u in adj[v]:
                    if color[u] == -1:
                        if domains[u] & (1 << c):
                            count += 1
                return count
            
            available_colors.sort(key=count_removed)
            
            # Try each color in LCV order
            for c in available_colors:
                color[v] = c
                domains[v] = (1 << c)
                
                # Save domains of neighbors to restore
                neighbor_domains = []
                valid = True
                
                # Forward Checking: Remove c from neighbors' domains
                # We also need to run AC-3 to propagate further
                # To optimize, we can just run AC-3 on the neighbors
                # But AC-3 modifies domains. We need to save them.
                
                # Save all domains? No, just save the ones that change.
                # Simpler: Save the entire domains array? 
                # For n<=408, saving 408 ints is fast.
                domains_snapshot = domains[:]
                
                # Assign v
                domains[v] = (1 << c)
                
                # Run AC-3 starting from neighbors of v
                # We need to check arcs (w, v) and (v, w)?
                # Actually, since v is assigned, we only need to check if neighbors have support in v.
                # v has only one value c. So a neighbor u has support in v iff c is in domains[u].
                # If c is not in domains[u], then u must have other values? No, u cannot use c.
                # Wait, if u's domain contains c, and v is assigned c, then c is removed from u.
                # This is Forward Checking.
                # Then we need to check if removing c from u causes u to lose support in its other neighbors.
                # This is AC-3.
                
                # Initialize AC-3 queue with neighbors of v
                ac_queue = []
                for u in adj[v]:
                    if color[u] == -1:
                        # Add (u, v) to queue? No, v is assigned.
                        # We need to check if u has a valid value given v's assignment.
                        # This is equivalent to checking arc (u, v).
                        # If u's domain becomes empty, fail.
                        # If u's domain shrinks, check u's other neighbors.
                        ac_queue.append((u, v))
                        
                if not enforce_ac3(ac_queue):
                    valid = False
                
                if valid and backtrack():
                    return True
                
                # Restore domains
                domains[:] = domains_snapshot
                color[v] = -1
                
            domains[v] = saved_domain
            color[v] = -1
            return False

        if backtrack():
            return [int(c) for c in color]
        else:
            return [0] * n

    out = []
    start_time = time.time()
    deadline = start_time + 110.0

    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline:
            # Fallback to avoid crash
            n, edges, lists = instance(splits, seed)
            out.append([0] * n)
            continue
            
        n, edges, lists = instance(splits, seed)
        res = solve(n, edges, lists)
        out.append(res)
        
    return out
# EVOLVE-BLOCK-END
