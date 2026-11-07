"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from random import Random
    from collections import deque

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

    def solve_instance(n, edges, lists, deadline):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Domains: list of sets of available colors
        domains = [set(lists[v]) for v in range(n)]
        
        # Check for initial empty domains
        for v in range(n):
            if not domains[v]:
                return None

        # AC-3 Implementation
        def ac3(domains, adj, n, queue=None):
            if queue is None:
                queue = deque()
                for u in range(n):
                    for v in adj[u]:
                        queue.append((u, v))
            
            while queue:
                u, v = queue.popleft()
                # Check if domain[u] is reduced by constraint (u, v)
                # We remove from domains[u] any color c such that there is no color d in domains[v] with d != c
                # Since constraint is u != v, c is removed from u if domains[v] == {c}
                # More generally: c is in domains[u] and for all d in domains[v], d == c.
                # i.e., if domains[v] is a subset of {c}, then c cannot be in domains[u].
                
                # Optimization: Only process if domains[v] has changed? 
                # Standard AC-3 adds (x, z) for all z in adj[x] \ {y} when y is revised.
                
                revised = False
                to_remove = []
                for c in domains[u]:
                    # If there is no valid color in domains[v] different from c, then c is not supported
                    # valid color d in domains[v] means d != c.
                    # So we need at least one d in domains[v] such that d != c.
                    # This fails if domains[v] is empty (handled by parent) or domains[v] == {c}.
                    if len(domains[v]) == 1 and list(domains[v])[0] == c:
                        to_remove.append(c)
                
                if to_remove:
                    for c in to_remove:
                        domains[u].discard(c)
                    revised = True
                    if not domains[u]:
                        return False
                    # Add neighbors of u (except v) to queue
                    for z in adj[u]:
                        if z != v:
                            queue.append((z, u))
            return True

        # Initial AC-3 pass
        if not ac3(domains, adj, n):
            return None

        # Backtracking with MRV and Forward Checking
        assigned = [None] * n
        
        def get_mrv_var():
            min_size = float('inf')
            var = -1
            for v in range(n):
                if assigned[v] is None:
                    size = len(domains[v])
                    if size < min_size:
                        min_size = size
                        var = v
                        if size <= 1:
                            break
            return var

        def backtrack():
            if time.time() > deadline:
                return False
            
            var = get_mrv_var()
            if var == -1:
                return True # All assigned
            
            # LVC Heuristic: Order values by how few constraints they impose on neighbors
            # For each color c in domains[var], count how many neighbors have c in their domain.
            # Prefer c with minimum count.
            values = list(domains[var])
            def value_score(c):
                count = 0
                for u in adj[var]:
                    if assigned[u] is None and c in domains[u]:
                        count += 1
                return count
            
            values.sort(key=value_score)
            
            for c in values:
                if c not in domains[var]:
                    continue
                
                # Assign
                assigned[var] = c
                domains[var].clear()
                
                # Forward Checking: Remove c from neighbors' domains
                saved_neighbors = []
                ok = True
                for u in adj[var]:
                    if assigned[u] is None:
                        if c in domains[u]:
                            saved_neighbors.append(u)
                            domains[u].discard(c)
                            if not domains[u]:
                                ok = False
                                break
                        else:
                            # No change, but we might need to propagate if domains[u] became singleton?
                            # AC-3 handles full propagation. FC is weaker but faster.
                            # Let's stick to FC for speed, but if FC fails, we might need AC.
                            # Given the density, FC is usually enough for these sparse graphs.
                            pass
                
                if ok:
                    # Optional: Run a local AC-3 check on affected nodes? 
                    # For now, just recurse.
                    if backtrack():
                        return True
                
                # Undo
                assigned[var] = None
                domains[var].add(c)
                for u in saved_neighbors:
                    domains[u].add(c)
                
                if time.time() > deadline:
                    return False
            
            return False

        if backtrack():
            return [int(assigned[v]) for v in range(n)]
        else:
            # Fallback: should not happen for these instances, but return dummy to avoid crash
            # The problem states all are colourable. If we fail, we return invalid.
            return [0] * n

    start = time.time()
    deadline = start + 115.0  # Safety margin before 120s limit

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline - 5:
            n, edges, lists = instance(splits, seed)
            # Emergency fallback: just pick first available color, likely invalid but valid structure
            out.append([int(lists[v][0]) for v in range(n)])
            continue
        
        n, edges, lists = instance(splits, seed)
        result = solve_instance(n, edges, lists, deadline)
        
        # Validate result strictly before adding
        # Check length
        if len(result) != n:
            result = [int(lists[v][0]) for v in range(n)]
        else:
            # Check types and values
            valid = True
            for i, c in enumerate(result):
                if not isinstance(c, int) or isinstance(c, bool):
                    valid = False
                    break
                if c not in lists[i]:
                    valid = False
                    break
            
            if valid:
                # Check edges
                for u, v in edges:
                    if result[u] == result[v]:
                        valid = False
                        break
            
            if not valid:
                # If our solver failed, try a simple greedy or just return dummy
                # Since they are guaranteed colourable, this is a bug in solver or timeout.
                # Returning dummy ensures "valid structure" but score 0 for this instance.
                result = [int(lists[v][0]) for v in range(n)]
        
        out.append(result)
    
    return out
# EVOLVE-BLOCK-END
