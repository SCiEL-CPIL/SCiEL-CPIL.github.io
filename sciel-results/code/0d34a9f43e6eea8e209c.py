import sys
import time

def construct():
    from random import Random
    
    # Increase recursion limit to be safe
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

    def solve_instance(n, edges, lists):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Pre-compute degrees
        degrees = [len(adj[i]) for i in range(n)]
        
        # Domains as bitmasks. Bit i is 1 if colour i is available.
        # Colours are 0..4, so bits 0..4.
        domains = [0] * n
        for i in range(n):
            mask = 0
            for c in lists[i]:
                mask |= (1 << c)
            domains[i] = mask
        
        assigned = [False] * n
        assignment = [-1] * n
        
        # Start time for deadline guard
        start_time = time.time()
        deadline = start_time + 110.0

        def check_deadline():
            return time.time() > deadline

        # AC-3 Implementation
        # Queue of arcs (u, v) to check
        # If domain of v changes, add (u, v) for all u in adj[v]
        def revise(u, v):
            # Returns True if domain of u was revised
            # Check if for every colour in domains[u], there is a compatible colour in domains[v]
            # Since it's a "not equal" constraint, domains[u] and domains[v] just need to not be identical singletons?
            # No, for "not equal", if domains[u] = {c} and domains[v] = {c}, conflict.
            # If domains[u] = {c, d} and domains[v] = {c}, then d is still valid.
            # So, we only remove c from domains[u] if domains[v] is a subset of {c}?
            # Actually, for "not equal", a value c in domains[u] is supported by domains[v] if domains[v] contains any value != c.
            # So c is unsupported ONLY IF domains[v] == {c} (i.e., domains[v] has only c and c==c).
            # Wait, if domains[v] = {c}, then any u with c in its domain must remove c.
            # If domains[v] = {c, d}, then c is supported by d, and d is supported by c.
            # So, we only need to check if domains[v] is a singleton.
            # If domains[v] is a singleton {c}, and c is in domains[u], remove c from domains[u].
            
            if domains[v] & (domains[v] - 1) == 0: # domains[v] is a power of 2 (singleton)
                mask_v = domains[v]
                if domains[u] & mask_v:
                    domains[u] &= ~mask_v
                    return True
            return False

        def ac3(initial_arcs=None):
            if initial_arcs is None:
                initial_arcs = []
            queue = list(initial_arcs)
            while queue:
                u, v = queue.pop()
                if revise(u, v):
                    if domains[u] == 0:
                        return False
                    for w in adj[u]:
                        if w != v:
                            queue.append((w, u))
            return True

        def backtrack(depth):
            if check_deadline():
                return False
            
            if depth == n:
                return True
            
            # MRV Heuristic: Find unassigned vertex with smallest domain size
            min_size = float('inf')
            best_v = -1
            best_deg = -1
            for v in range(n):
                if not assigned[v]:
                    size = bin(domains[v]).count('1')
                    if size < min_size or (size == min_size and degrees[v] > best_deg):
                        min_size = size
                        best_v = v
                        best_deg = degrees[v]
            
            if best_v == -1:
                return True
            
            if min_size == 0:
                return False
            
            v = best_v
            
            # Get list of colours in domain
            # Iterate through bits
            # To improve, we can order colours by "Least Constraining Value" (LCV)
            # LCV: choose the colour that rules out the fewest choices in neighbours.
            # Since constraint is "not equal", removing colour c from neighbour u reduces domain size of u by 1 if c in domain[u].
            # So LCV is the colour c that minimizes sum(1 if c in domain[u] for u in adj[v]).
            # This is equivalent to maximizing the number of neighbours that do NOT have c.
            
            colors_in_domain = []
            temp_mask = domains[v]
            while temp_mask:
                lsb = temp_mask & -temp_mask
                c = lsb.bit_length() - 1
                colors_in_domain.append(c)
                temp_mask ^= lsb
            
            # Calculate LCV score
            # score[c] = number of neighbours that have c in their domain
            # We want to try c with lowest score first.
            scores = [0] * 5
            for u in adj[v]:
                if not assigned[u]:
                    for c in range(5):
                        if domains[u] & (1 << c):
                            scores[c] += 1
            
            # Sort colors by score
            colors_in_domain.sort(key=lambda c: scores[c])
            
            for c in colors_in_domain:
                if check_deadline():
                    return False
                
                # Assign c to v
                assigned[v] = True
                assignment[v] = c
                old_domain_v = domains[v]
                domains[v] = (1 << c)
                
                # Prepare for undo: store old domains of neighbours
                changes = []
                conflict = False
                
                # Forward Checking + AC-3
                # For each unassigned neighbour u, remove c from domain[u]
                # Then run AC-3 on the affected graph
                # Since we only removed one value, we can just check if any neighbour becomes empty or if AC-3 fails.
                
                for u in adj[v]:
                    if not assigned[u]:
                        old_dom_u = domains[u]
                        if domains[u] & (1 << c):
                            domains[u] &= ~(1 << c)
                            changes.append((u, old_dom_u))
                            if domains[u] == 0:
                                conflict = True
                                break
                
                if not conflict:
                    # Run AC-3 on the subgraph affected by changes
                    # We need to add arcs (w, u) for all u in changes, and w in adj[u]
                    # And also (u, w) ? No, revise(u, w) checks u against w.
                    # If domain of u changed, we need to revise (w, u) for all w in adj[u].
                    arcs = []
                    for u, _ in changes:
                        for w in adj[u]:
                            if w != v: # v is assigned, no need to revise against v? 
                                # Actually, v is assigned to c. If w has c, it should have been removed.
                                # But we only removed c from direct neighbours of v.
                                # If w is not a neighbour of v, but is a neighbour of u, and u lost c, 
                                # does w need to change? 
                                # Only if w's domain was reduced due to u's domain being a singleton?
                                # AC-3 handles this.
                                arcs.append((w, u))
                    
                    if arcs:
                        if not ac3(arcs):
                            conflict = True
                
                if conflict:
                    # Undo
                    for u, old_dom in changes:
                        domains[u] = old_dom
                    domains[v] = old_domain_v
                    assigned[v] = False
                    assignment[v] = -1
                    continue
                
                if backtrack(depth + 1):
                    return True
                
                # Undo
                for u, old_dom in changes:
                    domains[u] = old_dom
                domains[v] = old_domain_v
                assigned[v] = False
                assignment[v] = -1
            
            return False

        # Run solver
        if backtrack(0):
            return [int(assignment[i]) for i in range(n)]
        else:
            return [0] * n

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        coloring = solve_instance(n, edges, lists)
        out.append(coloring)
    return out
