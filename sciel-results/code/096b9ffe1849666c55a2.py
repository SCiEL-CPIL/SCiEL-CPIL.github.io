"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

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

    def solve_exact(n, edges, lists):
        """Exact backtracking with MRV, forward checking, and AC-3 propagation."""
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Convert to lists for faster iteration
        adj_list = [sorted(s) for s in adj]
        
        # Initial domains as sets
        domains = [set(l) for l in lists]
        
        # Color assignment: -1 means uncolored
        color = [-1] * n
        
        # Count remaining uncolored
        uncolored = set(range(n))
        
        # AC-3 style arc consistency propagation
        def propagate(queue):
            """Apply arc consistency. Returns False if domain becomes empty."""
            while queue:
                u, v = queue.pop()
                if u in uncolored and v in uncolored:
                    du = domains[u]
                    dv = domains[v]
                    # Remove values from du that have no support in dv
                    # i.e., cu is removed if cu == cv for all cv in dv
                    # This is equivalent to: if du is subset of {cv for cv in dv if cv == cu}
                    # Simpler: if du and dv are singletons and equal, remove it.
                    # General case: for each cu, if no cv in dv such that cu != cv, remove cu.
                    to_remove = set()
                    for cu in du:
                        # Check if there exists a cv in dv such that cu != cv
                        has_support = False
                        for cv in dv:
                            if cu != cv:
                                has_support = True
                                break
                        if not has_support:
                            to_remove.add(cu)
                    if to_remove:
                        domains[u] = du - to_remove
                        if not domains[u]:
                            return False
                        for w in adj_list[u]:
                            if w != v and w in uncolored:
                                queue.append((w, u))
            return True
        
        def backtrack():
            if not uncolored:
                return True
            
            # MRV: find variable with smallest domain
            min_var = None
            min_size = float('inf')
            doms = domains
            for v in uncolored:
                d = len(doms[v])
                if d == 0:
                    return False
                if d < min_size:
                    min_size = d
                    min_var = v
                    if d == 1:
                        break
            
            # Try colors in the domain of min_var
            # Order: least constraining value
            colors = list(domains[min_var])
            def lcv(c):
                count = 0
                for nb in adj_list[min_var]:
                    if nb in uncolored and c in domains[nb]:
                        count += 1
                return count
            colors.sort(key=lcv)
            
            for c in colors:
                # Assign color c to min_var
                color[min_var] = c
                uncolored.discard(min_var)
                
                # Save old domains for undo
                saved = {}
                valid = True
                for nb in adj_list[min_var]:
                    if nb in uncolored:
                        if c in domains[nb]:
                            saved[nb] = domains[nb].copy()
                            domains[nb].discard(c)
                            if not domains[nb]:
                                valid = False
                                break
                
                if valid:
                    # Run AC-3 propagation
                    queue = []
                    for nb in adj_list[min_var]:
                        if nb in uncolored:
                            for w in adj_list[nb]:
                                if w != min_var and w in uncolored:
                                    queue.append((w, nb))
                    if propagate(queue) and backtrack():
                        return True
                
                # Undo
                for nb, old_domain in saved.items():
                    domains[nb] = old_domain
                uncolored.add(min_var)
                color[min_var] = -1
            
            return False
        
        if backtrack():
            return color
        return None

    def solve_greedy(n, edges, lists):
        """Greedy coloring as fallback."""
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        colour = [None] * n
        for v in sorted(range(n), key=lambda u: -len(adj[u])):
            taken = {colour[u] for u in adj[v] if colour[u] is not None}
            free = [c for c in lists[v] if c not in taken]
            colour[v] = free[0] if free else lists[v][0]
        return colour

    deadline = time.time() + 115.0  # Safety margin
    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline - 5.0:
            # Not enough time left, use greedy for remaining
            n, edges, lists = instance(splits, seed)
            result = solve_greedy(n, edges, lists)
            out.append([int(c) for c in result])
            continue
            
        n, edges, lists = instance(splits, seed)
        
        # Try exact backtracking first
        result = solve_exact(n, edges, lists)
        
        if result is None or time.time() > deadline - 10.0:
            # Fallback to greedy
            result = solve_greedy(n, edges, lists)
        
        # Ensure all values are Python int
        out.append([int(c) for c in result])
    
    return out
# EVOLVE-BLOCK-END
