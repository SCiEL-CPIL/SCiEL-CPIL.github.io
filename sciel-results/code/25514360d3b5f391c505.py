"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

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

    def solve_instance(n, edges, lists):
        # Build adjacency list
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Initial domains
        domains = [set(l) for l in lists]
        
        # Order vertices by degree (higher degree first) to help with early constraint propagation
        # But MRV is dynamic, so we start with a static order for initial selection if domains are equal
        order = sorted(range(n), key=lambda v: -len(adj[v]))
        
        color = [-1] * n
        
        # Recursive backtracking with MRV
        # To avoid recursion depth issues, we use an iterative approach or increase limit
        # Given n <= 408, recursion depth is fine, but we'll use a stack for safety or just increase limit
        sys.setrecursionlimit(5000)
        
        start_time = time.time()
        deadline = start_time + 110.0 # Leave 10s margin
        
        def backtrack(unassigned):
            if not unassigned:
                return True
            
            # MRV: pick vertex with smallest domain size
            min_v = None
            min_size = float('inf')
            for v in unassigned:
                size = len(domains[v])
                if size < min_size:
                    min_size = size
                    min_v = v
                    if size == 1:
                        break
                    if size == 0:
                        return False
            
            if min_size == 0:
                return False
            
            v = min_v
            # Try colors in the domain
            # Heuristic: try colors that appear least frequently in neighbors? 
            # For now, just iterate.
            for c in list(domains[v]):
                # Check if c is valid for neighbors
                valid = True
                for u in adj[v]:
                    if u in unassigned:
                        if c in domains[u]:
                            # This doesn't immediately invalidate, but we must ensure consistency later
                            # Actually, we just need to check if neighbor is already colored
                            pass
                    else:
                        if color[u] == c:
                            valid = False
                            break
                if not valid:
                    continue
                
                # Assign v = c
                color[v] = c
                new_domains = [set(d) for d in domains] # Copy is expensive, let's do in-place with undo
                # Actually, copying sets for 400 vertices at each step is O(N^2 * Branch). 
                # Better to do in-place modification and undo.
                
                # Let's restructure to use in-place domains
                # We pass domains as mutable state
                
                # Undo logic:
                saved_domains = []
                for u in adj[v]:
                    if u in unassigned:
                        saved_domains.append((u, domains[u].copy()))
                        domains[u].discard(c)
                        if not domains[u]:
                            # Restore and fail
                            for uu, dd in saved_domains:
                                domains[uu] = dd
                            color[v] = -1
                            break
                else:
                    # No immediate conflict, recurse
                    new_unassigned = set(unassigned)
                    new_unassigned.remove(v)
                    if backtrack(new_unassigned):
                        return True
                    # Undo
                    for uu, dd in saved_domains:
                        domains[uu] = dd
                color[v] = -1
                
            return False
        
        # Initial call
        # We need to handle the "unassigned" set carefully.
        # A simple list/set of unassigned vertices.
        initial_unassigned = set(range(n))
        
        # Optimization: Static ordering might be better for initial MRV tie-breaking
        # But dynamic MRV is usually superior.
        
        # To speed up, we can use a more efficient data structure for domains,
        # but sets are fine for N=400.
        
        # Let's refine the backtrack function to avoid copying domains unnecessarily
        # The previous implementation copied domains in the failed branch.
        
        # Let's rewrite backtrack to be cleaner
        def backtrack_clean(unassigned):
            if time.time() > deadline:
                return False
            
            if not unassigned:
                return True
            
            # MRV
            min_v = None
            min_size = float('inf')
            # To make it deterministic and fast, we can just iterate
            for v in unassigned:
                size = len(domains[v])
                if size < min_size:
                    min_size = size
                    min_v = v
                    if size <= 1:
                        break
            
            if min_size == 0:
                return False
            
            v = min_v
            
            # Sort colors by least constraining value (LCV) - optional, might help
            # For now, just iterate
            for c in domains[v]:
                # Check neighbors
                conflict = False
                for u in adj[v]:
                    if u not in unassigned:
                        if color[u] == c:
                            conflict = True
                            break
                if conflict:
                    continue
                
                # Assign
                color[v] = c
                changes = []
                for u in adj[v]:
                    if u in unassigned:
                        if c in domains[u]:
                            changes.append(u)
                            domains[u].discard(c)
                            if not domains[u]:
                                # Conflict
                                for uu in changes:
                                    # We need to restore, but we didn't save the old state.
                                    # We need to save old state or re-derive.
                                    # Re-deriving is hard. Saving is O(Degree).
                                    # Let's save old state.
                                    pass
                                # This approach is messy. Let's stick to the previous one but optimize.
                                break
                
                if not changes:
                    # No changes needed? Or changes were made but no conflict?
                    pass
                
                # This "in-place with undo" is tricky if we don't save state.
                # Let's go back to the simple copy approach but optimize it.
                # N=400. Depth=400. Copying 400 sets of size 3 is fast.
                # 400 * 400 * 3 operations = 480,000 operations. Very fast.
                
                pass
            
            return False
        
        # Let's use the robust simple copy approach, it's safe and fast enough.
        def backtrack_simple(unassigned):
            if time.time() > deadline:
                return False
            if not unassigned:
                return True
            
            # MRV
            min_v = None
            min_size = float('inf')
            for v in unassigned:
                size = len(domains[v])
                if size < min_size:
                    min_size = size
                    min_v = v
                    if size <= 1:
                        break
            
            if min_size == 0:
                return False
            
            v = min_v
            # Try colors
            # LCV heuristic: try colors that rule out the fewest choices for neighbors
            # This is expensive to compute. Skip for now.
            for c in domains[v]:
                # Check hard constraints with colored neighbors
                ok = True
                for u in adj[v]:
                    if u not in unassigned:
                        if color[u] == c:
                            ok = False
                            break
                if not ok:
                    continue
                
                # Propagate
                new_domains = [set(d) for d in domains]
                new_unassigned = set(unassigned)
                new_unassigned.remove(v)
                
                consistent = True
                for u in adj[v]:
                    if u in new_unassigned:
                        new_domains[u].discard(c)
                        if not new_domains[u]:
                            consistent = False
                            break
                
                if consistent:
                    color[v] = c
                    if backtrack_simple(new_unassigned):
                        return True
                    color[v] = -1
                # else: discard this color
                
            return False
        
        # Initial domains are already sets
        success = backtrack_simple(set(range(n)))
        
        if not success:
            # Should not happen due to Alon-Tarsi, but if timeout or bug, return best effort
            # Fill with -1 or 0? The task says "int colour in 0..4".
            # If we fail, we return a dummy list to maintain type, though it will score 0.
            for i in range(n):
                if color[i] == -1:
                    color[i] = lists[i][0]
        
        return [int(c) for c in color]

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        out.append(solve_instance(n, edges, lists))
    
    return out
# EVOLVE-BLOCK-END
