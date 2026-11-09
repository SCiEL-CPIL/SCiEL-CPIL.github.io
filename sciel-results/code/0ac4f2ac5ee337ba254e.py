"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    import heapq

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

    def solve(n, edges, lists):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Initialize domains as bitmasks
        # Color c is represented by bit 1 << c
        domains = [0] * n
        for i, lst in enumerate(lists):
            mask = 0
            for c in lst:
                mask |= (1 << c)
            domains[i] = mask
        
        # AC-3 Propagation
        # Queue of arcs (u, v) where u is the variable being checked against neighbor v
        # We need to check if any value in domains[u] is not supported by domains[v]
        # Since constraints are "not equal", a value c in u is supported by v if domains[v] has any bit other than c.
        # So, domains[u] can be pruned to remove c if domains[v] is exactly (1 << c).
        
        import collections
        
        def ac3(domains):
            queue = collections.deque()
            # Initialize queue with all arcs
            for u in range(n):
                for v in adj[u]:
                    queue.append((u, v))
            
            while queue:
                u, v = queue.popleft()
                # Check if domains[u] can be reduced given domains[v]
                # A value c in u is invalid if c is in u AND c is the ONLY value in v?
                # No, c is invalid if c in u and c in v and v has no other values?
                # Actually, constraint is u != v.
                # Value c in u is supported if there exists a value in v such that c != value_in_v.
                # This is true unless domains[v] is a subset of {c}.
                # So if domains[v] == (1 << c), then c cannot be in u.
                
                new_domain_u = domains[u]
                # Iterate through bits in domains[u]
                # Efficient way: 
                # If domains[v] has only one bit, say 1<<k, then k must be removed from u.
                # If domains[v] has >1 bits, all values in u are supported (since u != v is satisfiable for any u_val if v has another val).
                
                # Optimization: Only prune if domains[v] is a single value.
                if domains[v] and (domains[v] & (domains[v] - 1)) == 0:
                    # domains[v] is a power of 2, i.e., single value
                    # Find that value
                    val = domains[v]
                    if new_domain_u & val:
                        new_domain_u &= ~val
                        if new_domain_u == 0:
                            return False
                        if new_domain_u != domains[u]:
                            domains[u] = new_domain_u
                            # Add all neighbors of u (except v) to queue
                            for w in adj[u]:
                                if w != v:
                                    queue.append((w, u))
            return True

        # Initial propagation
        if not ac3(domains):
            return None # Should not happen if instance is valid and colorable, but safe
            
        # Backtracking Search
        # State: domains (list of ints), assigned (list of -1 or color)
        
        def backtrack(domains, assigned):
            # Find unassigned variable with MRV
            min_size = 100
            min_var = -1
            for i in range(n):
                if assigned[i] == -1:
                    cnt = bin(domains[i]).count('1')
                    if cnt < min_size:
                        min_size = cnt
                        min_var = i
                        if cnt <= 1:
                            break
                            
            if min_var == -1:
                # All assigned
                return assigned[:]
                
            if min_size == 0:
                return None
                
            # Try each value in domains[min_var]
            # Order: Least Constraining Value (LCV) is ideal, but simple sorted is okay.
            # Let's try values in order of appearance.
            d = domains[min_var]
            values = []
            temp_d = d
            while temp_d:
                lsb = temp_d & -temp_d
                val = lsb.bit_length() - 1
                values.append(val)
                temp_d ^= lsb
                
            for val in values:
                new_assigned = assigned[:]
                new_assigned[min_var] = val
                new_domains = domains[:]
                
                # Assign value to min_var
                new_domains[min_var] = 1 << val
                
                # Propagate
                if ac3(new_domains):
                    result = backtrack(new_domains, new_assigned)
                    if result is not None:
                        return result
                        
            return None

        assigned = [-1] * n
        result = backtrack(domains, assigned)
        
        if result is None:
            # Fallback
            return [lists[v][0] for v in range(n)]
            
        return result

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        coloring = solve(n, edges, lists)
        # Ensure types are Python int
        coloring = [int(c) for c in coloring]
        out.append(coloring)
    return out
# EVOLVE-BLOCK-END
