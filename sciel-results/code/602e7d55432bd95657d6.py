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

    def solve_csp(n, edges, lists):
        """
        Solve the list-coloring problem using forward checking with backtracking.
        Uses a heap for MRV selection to improve efficiency.
        Returns a list of colors (integers) of length n, or None if no solution found.
        """
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Initialize domains as lists for faster iteration and copying
        domains = [list(lst) for lst in lists]
        assigned = [False] * n
        color = [None] * n
        
        # Heap for MRV: (domain_size, vertex_id)
        # We need to handle stale entries in the heap, so we check validity when popping
        heap = []
        for v in range(n):
            heapq.heappush(heap, (len(domains[v]), v))

        # To allow fast "is assigned" check and efficient domain updates,
        # we will store domains as sets for O(1) removal, but keep a list for iteration?
        # Actually, sets are good for removal. Let's use sets for domains.
        domains = [set(lst) for lst in lists]
        
        # Re-push to heap with correct sizes if we switch to sets (sizes are same)
        # But we need to update heap when domains change.
        # A simpler approach for n~400: Just use a list and find min? 
        # No, heap is better. But updating heap is tricky with stale entries.
        # Let's stick to the standard "lazy deletion" approach for heap.
        
        # Reset state
        assigned = [False] * n
        color = [None] * n
        domains = [set(lst) for lst in lists]
        
        heap = []
        for v in range(n):
            heapq.heappush(heap, (len(domains[v]), v))
            
        # Count of unassigned
        unassigned_count = n

        def pop_mrv():
            """
            Pop the valid MRV vertex from the heap.
            Returns vertex id or None if no unassigned vertices.
            """
            nonlocal unassigned_count
            while heap:
                size, v = heapq.heappop(heap)
                if not assigned[v] and len(domains[v]) == size:
                    return v
                # If stale (assigned or size changed), discard
            return None

        def push_mrv(v):
            heapq.heappush(heap, (len(domains[v]), v))

        # Iterative backtracking
        # We need to save state for backtracking.
        # State: assigned[], color[], domains[]
        # Since domains are sets, we need to copy them or use a trail.
        # Trail is more efficient.
        
        # Trail entries: (vertex, color_removed)
        trail = []
        
        # Stack for backtracking: (vertex, next_color_index, saved_domain_of_v)
        # Actually, we can just store the vertex and the list of colors to try.
        
        # Let's use a recursive approach but with the heap for MRV selection?
        # Recursion depth is n (max 408), which is safe for Python (default limit 1000).
        # The bottleneck was likely the O(N) scan for MRV.
        # So, recursive + heap for MRV selection is a good balance.
        
        def backtrack():
            nonlocal unassigned_count
            
            v = pop_mrv()
            if v is None:
                return True # No unassigned vertices left
            
            unassigned_count -= 1
            saved_domain_v = set(domains[v])
            assigned[v] = True
            
            for c in saved_domain_v:
                color[v] = c
                
                # Forward Check
                changes = []
                consistent = True
                for u in adj[v]:
                    if not assigned[u] and c in domains[u]:
                        domains[u].remove(c)
                        changes.append(u)
                        if len(domains[u]) == 0:
                            consistent = False
                            break
                
                if consistent:
                    # Push updated neighbors to heap (lazy update)
                    for u in changes:
                        push_mrv(u)
                    
                    if backtrack():
                        # Undo changes before returning success? 
                        # Actually, if we return success, we don't need to undo.
                        # But we must ensure that if we return False, we undo.
                        # Wait, if backtrack returns True, the solution is found.
                        # We can just return True. The state is left modified, but that's fine.
                        return True
                else:
                    # Contradiction, undo changes
                    for u in changes:
                        domains[u].add(c)
                    # Note: we don't need to push to heap here because we are backtracking
                    # and the next iteration will pop valid entries or push new ones if needed.
                    # However, the heap might contain stale entries for 'u' with larger sizes.
                    # This is handled by pop_mrv checking size.
                
                # If we get here, this color c failed.
                # We need to restore domains of neighbors if we modified them?
                # In the 'consistent' branch, if backtrack failed, it should have undone its own changes.
                # But did it? 
                # My recursive backtrack does NOT undo changes on failure explicitly in the code above 
                # if I just return False.
                # I need to ensure that when backtrack() returns False, the state is restored 
                # to what it was BEFORE the call to backtrack().
                
                # Let's restructure:
                # The standard pattern is:
                # 1. Save state
                # 2. Apply change
                # 3. Recurse
                # 4. If success, return
                # 5. If fail, Undo change
                # 6. Try next option
                
                # So, if consistent and backtrack() returns False, we must undo the changes 
                # made in this step (removing c from neighbors).
                
                # Wait, in the code above:
                # if consistent:
                #    ...
                #    if backtrack(): return True
                #    # Implicitly, if backtrack() returns False, we fall through to undo?
                #    # No, I need to explicitly undo the 'changes' made in this step if backtrack fails.
                
                # Let's rewrite the loop body clearly.
            
            # Undo assignment of v
            assigned[v] = False
            color[v] = None
            domains[v] = saved_domain_v
            unassigned_count += 1
            
            return False
        
        # Let's refine the backtrack function to handle undo correctly.
        
        def backtrack_v2():
            nonlocal unassigned_count
            
            v = pop_mrv()
            if v is None:
                return True
            
            unassigned_count -= 1
            saved_domain_v = set(domains[v])
            assigned[v] = True
            
            for c in saved_domain_v:
                color[v] = c
                
                changes = []
                consistent = True
                for u in adj[v]:
                    if not assigned[u] and c in domains[u]:
                        domains[u].remove(c)
                        changes.append(u)
                        if len(domains[u]) == 0:
                            consistent = False
                            break
                
                if consistent:
                    for u in changes:
                        push_mrv(u)
                    
                    if backtrack_v2():
                        return True
                    else:
                        # Backtrack failed, undo changes made in this step
                        for u in changes:
                            domains[u].add(c)
                        # No need to push to heap, pop_mrv handles stale entries
                # If not consistent, changes were already undone in the loop break?
                # No, if consistent is False, we broke out of the neighbor loop.
                # We must undo the changes made before the break.
                else:
                    for u in changes:
                        domains[u].add(c)
                
            # All colors failed
            assigned[v] = False
            color[v] = None
            domains[v] = saved_domain_v
            unassigned_count += 1
            
            return False

        if backtrack_v2():
            return color
        else:
            return None

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        coloring = solve_csp(n, edges, lists)
        if coloring is None:
            # Fallback: should not happen per Alon-Tarsi, but just in case
            # Use a simple greedy as fallback
            adj = [set() for _ in range(n)]
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            coloring = [None] * n
            for v in sorted(range(n), key=lambda u: -len(adj[u])):
                taken = {coloring[u] for u in adj[v] if coloring[u] is not None}
                free = [c for c in lists[v] if c not in taken]
                coloring[v] = free[0] if free else lists[v][0]
        # Ensure types are Python int
        out.append([int(c) for c in coloring])
    return out
# EVOLVE-BLOCK-END
