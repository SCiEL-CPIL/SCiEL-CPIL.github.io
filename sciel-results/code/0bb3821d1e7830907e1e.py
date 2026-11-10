"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

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

    def solve_csp(n, edges, lists):
        # Build adjacency
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # AC-3 initialization: domains are the lists
        domains = [set(lst) for lst in lists]
        
        # Queue of arcs (i, j) where i's domain might be reduced by j's domain
        # We need to check all arcs initially
        queue = []
        for u in range(n):
            for v in adj[u]:
                queue.append((u, v))
        
        # AC-3 algorithm
        while queue:
            u, v = queue.pop()
            if revise(domains, u, v):
                if not domains[u]:
                    return None  # Inconsistent
                for w in adj[u]:
                    if w != v:
                        queue.append((w, u))
        
        # Backtracking search with forward checking
        # Order vertices by domain size (MRV) - but since we need to assign all, we use a static order
        # For planar bipartite graphs, a good heuristic is degree or MRV.
        # Let's use a simple recursive backtracking.
        
        # To speed up, we can order vertices.
        # Since the graph is bipartite and planar, and lists are size 3, it should be easy.
        
        assignment = [None] * n
        
        def backtrack(idx):
            # Select unassigned vertex with smallest domain (MRV)
            min_size = float('inf')
            var = -1
            for i in range(n):
                if assignment[i] is None:
                    if len(domains[i]) < min_size:
                        min_size = len(domains[i])
                        var = i
            if var == -1:
                return True  # All assigned
            
            # Try each value in domain
            for val in domains[var]:
                # Check if val is consistent with assigned neighbors
                consistent = True
                for nei in adj[var]:
                    if assignment[nei] is not None and assignment[nei] == val:
                        consistent = False
                        break
                if not consistent:
                    continue
                
                # Assign
                assignment[var] = val
                
                # Forward checking: prune domains of unassigned neighbors
                saved_domains = []
                consistent_fc = True
                for nei in adj[var]:
                    if assignment[nei] is None:
                        saved_domains.append((nei, set(domains[nei])))
                        domains[nei].discard(val)
                        if not domains[nei]:
                            consistent_fc = False
                            break
                
                if consistent_fc:
                    if backtrack(idx + 1):
                        return True
                
                # Undo forward checking
                for nei, dom in saved_domains:
                    domains[nei] = dom
                assignment[var] = None
            
            return False
        
        # We need to be careful with modifying domains in place during backtracking.
        # The above code saves/restores domains, which is correct but might be slow.
        # For these small graphs, it should be fine.
        
        if backtrack(0):
            return assignment
        else:
            return None

    def greedy(n, edges, palette):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        colour = [None] * n
        for v in sorted(range(n), key=lambda u: -len(adj[u])):
            taken = {colour[u] for u in adj[v]}
            free = [c for c in palette(v) if c not in taken]
            colour[v] = free[0] if free else palette(v)[0]
        return colour

    out = []
    deadline = time.time() + 110  # Safety margin
    
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline:
            # If we run out of time, we need to return something valid.
            # The task says "Any broken rule makes the whole result invalid".
            # So we must return a list of 4 lists.
            # If we haven't solved all, we can't just return partial.
            # However, the problem is small enough that this shouldn't happen.
            # If it does, we might try a faster but less robust method for remaining.
            pass
            
        n, edges, lists = instance(splits, seed)
        
        # Try CSP solver first
        colouring = solve_csp(n, edges, lists)
        
        if colouring is None:
            # Fallback to greedy (should not happen if CSP is correct and graph is colourable)
            # But greedy might produce invalid colouring if it picks wrong colors.
            # The task says greedy scored 1/4, meaning it failed on 3.
            # If CSP fails, it's a bug. But let's add a fallback just in case.
            # Actually, if CSP returns None, it means it proved infeasible, which contradicts Alon-Tarsi.
            # So CSP should never return None for these instances.
            # If it does, we try greedy.
            colouring = greedy(n, edges, lambda v: lists[v])
            
        # Ensure types are Python int
        colouring = [int(c) for c in colouring]
        out.append(colouring)
        
    return out
# EVOLVE-BLOCK-END
