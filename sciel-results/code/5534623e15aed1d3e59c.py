"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import deque

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

    def solve_instance(n, edges, lists, deadline, rng):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Degree for variable ordering tie-breaking
        degree = [len(adj[i]) for i in range(n)]

        # Initialize domains as bitmasks
        # Color c is represented by bit (1 << c)
        domains = [0] * n
        for i in range(n):
            mask = 0
            for c in lists[i]:
                mask |= (1 << c)
            domains[i] = mask
        
        assignment = [-1] * n
        sys.setrecursionlimit(10000)

        # AC-3 with bitmasks
        def arc_consistency(initial_queue=None):
            """Apply AC-3. Returns False if any domain becomes empty."""
            if initial_queue is None:
                queue = deque()
                for u in range(n):
                    for v in adj[u]:
                        queue.append((v, u))
            else:
                queue = deque(initial_queue)
            
            while queue:
                if time.time() > deadline:
                    return 'timeout'
                
                xi, xj = queue.popleft()
                
                # Check if Dom(xi) has a support in Dom(xj) for every value in Dom(xi)
                # Equivalent to: Does Dom(xi) contain any value c such that c is NOT in Dom(xj)?
                # If so, remove those c from Dom(xi).
                # Actually, arc consistency for "not equal" constraint:
                # Dom(xi) is consistent with Dom(xj) if for every c in Dom(xi), there exists d in Dom(xj) such that c != d.
                # This fails only if Dom(xj) is empty, or if Dom(xj) = {c} and c in Dom(xi).
                # If Dom(xj) has size >= 2, any c in Dom(xi) has a support (any d != c).
                # So we only need to check if Dom(xj) is singleton.
                
                if domains[xj] == 0:
                    # Neighbor domain empty, xi must be empty too? No, if neighbor is empty, the problem is unsolvable.
                    # Actually, if Dom(xj) is empty, the instance is unsolvable.
                    return False

                # If Dom(xj) is a singleton {c}, then c cannot be in Dom(xi).
                # Check if Dom(xj) is power of 2 (singleton)
                if domains[xj] & (domains[xj] - 1) == 0:
                    # It is a singleton
                    c_bit = domains[xj]
                    if domains[xi] & c_bit:
                        # Remove c from Dom(xi)
                        new_dom = domains[xi] & ~c_bit
                        if new_dom != domains[xi]:
                            domains[xi] = new_dom
                            if domains[xi] == 0:
                                return False
                            # Add neighbors of xi to queue, excluding xj
                            for xk in adj[xi]:
                                if xk != xj:
                                    queue.append((xk, xi))
            
            return True

        # Try AC-3 first
        ac_result = arc_consistency()
        if ac_result is False:
            return None
        if ac_result == 'timeout':
            return None

        # Helper to get count of bits
        def popcount(x):
            return bin(x).count('1')

        # Backtracking search
        def backtrack():
            if time.time() > deadline:
                return 'timeout'
            
            # Find unassigned vertex with smallest domain size (MRV)
            best_v = -1
            best_count = float('inf')
            best_deg = -1
            for v in range(n):
                if assignment[v] == -1:
                    count = popcount(domains[v])
                    if count == 0:
                        return False
                    if count < best_count or (count == best_count and degree[v] > best_deg):
                        best_count = count
                        best_v = v
                        best_deg = degree[v]
            
            if best_v == -1:
                return True  # All assigned
            
            # Extract colors from domain bitmask
            dom = domains[best_v]
            
            # Try each color in the domain
            while dom:
                if time.time() > deadline:
                    return 'timeout'
                
                # Get lowest set bit
                lsb = dom & -dom
                c = lsb.bit_length() - 1
                dom ^= lsb # Remove lsb from local dom iteration
                
                # Save state for undo
                old_dom_v = domains[best_v]
                old_assignment = assignment[best_v]
                
                # Assign
                assignment[best_v] = c
                domains[best_v] = lsb
                
                # Propagate: Remove c from neighbors' domains
                valid = True
                for u in adj[best_v]:
                    if assignment[u] == -1:
                        if domains[u] & lsb:
                            new_dom_u = domains[u] & ~lsb
                            if new_dom_u == 0:
                                # Undo and break
                                valid = False
                                break
                            domains[u] = new_dom_u
                
                if valid:
                    # Run AC-3 on affected arcs
                    # Affected arcs are (neighbor, best_v) for all unassigned neighbors
                    affected = [(u, best_v) for u in adj[best_v] if assignment[u] == -1]
                    ac_res = arc_consistency(affected)
                    
                    if ac_res is True:
                        result = backtrack()
                        if result is True:
                            return True
                        if result == 'timeout':
                            # Undo
                            assignment[best_v] = old_assignment
                            domains[best_v] = old_dom_v
                            return 'timeout'
                        # result is False, undo and try next color
                    elif ac_res is False:
                        # Undo
                        pass
                    else: # timeout
                        assignment[best_v] = old_assignment
                        domains[best_v] = old_dom_v
                        return 'timeout'
                
                # Undo changes to neighbors
                # We need to restore domains of neighbors that we modified.
                # Since we didn't track which ones were modified, we can just re-run AC-3 or restore from a backup?
                # Better: Store changes in a list? Or simply restore domains from a saved snapshot?
                # Snapshotting full domains array is O(N). N is small (~400). This is acceptable.
                
                # Let's optimize undo by saving only changed domains?
                # Actually, for this size, snapshotting `domains` is fast enough and less error prone.
                # But we are inside recursion. Let's snapshot before modification.
                
                # Correction: The code above modified `domains` in place.
                # To undo properly, I need to have saved the state of `domains` before the modification.
                # Let's refactor the loop to save a copy of domains.
                
                # This approach is getting complex with in-place modifications.
                # Let's restart the undo logic:
                # 1. Save `old_domains = domains[:]`
                # 2. Modify
                # 3. If fail, `domains[:] = old_domains`
                
                # Let's rewrite the inner loop with snapshotting.
                # (Self-correction: The previous block was flawed. I will implement the snapshot version below.)
                pass # Placeholder, will implement properly in final code

        # Refactored Backtracking with Snapshotting
        def backtrack_refactored():
            if time.time() > deadline:
                return 'timeout'
            
            # Find MRV
            best_v = -1
            best_count = float('inf')
            best_deg = -1
            for v in range(n):
                if assignment[v] == -1:
                    count = popcount(domains[v])
                    if count == 0:
                        return False
                    if count < best_count or (count == best_count and degree[v] > best_deg):
                        best_count = count
                        best_v = v
                        best_deg = degree[v]
            
            if best_v == -1:
                return True
            
            dom = domains[best_v]
            
            # Shuffle colors for randomized search order if needed? 
            # For now, deterministic order, but we can retry with different seeds later if it fails.
            
            # Extract colors
            colors = []
            temp = dom
            while temp:
                lsb = temp & -temp
                c = lsb.bit_length() - 1
                colors.append(c)
                temp ^= lsb
            
            for c in colors:
                if time.time() > deadline:
                    return 'timeout'
                
                lsb = 1 << c
                
                # Snapshot
                old_domains = domains[:]
                old_assignment = assignment[best_v]
                
                # Assign
                assignment[best_v] = c
                domains[best_v] = lsb
                
                # Propagate
                valid = True
                for u in adj[best_v]:
                    if assignment[u] == -1:
                        if domains[u] & lsb:
                            new_dom_u = domains[u] & ~lsb
                            if new_dom_u == 0:
                                valid = False
                                break
                            domains[u] = new_dom_u
                
                if valid:
                    affected = [(u, best_v) for u in adj[best_v] if assignment[u] == -1]
                    ac_res = arc_consistency(affected)
                    
                    if ac_res is True:
                        result = backtrack_refactored()
                        if result is True:
                            return True
                        if result == 'timeout':
                            domains[:] = old_domains
                            assignment[best_v] = old_assignment
                            return 'timeout'
                        # Else result is False, continue to next color
                    elif ac_res is False:
                        pass
                    else: # timeout
                        domains[:] = old_domains
                        assignment[best_v] = old_assignment
                        return 'timeout'
                
                # Undo
                domains[:] = old_domains
                assignment[best_v] = old_assignment
            
            return False

        # Initial AC-3
        if arc_consistency() is not True:
            return None

        result = backtrack_refactored()
        if result is True:
            # Convert bitmasks/assignments to list of ints
            # assignment[v] is already the color int
            return assignment
        return None

    out = []
    start_time = time.time()
    hard_deadline = start_time + 145.0

    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        
        if time.time() > hard_deadline - 5.0:
             out.append([0] * n)
             continue
        
        remaining_instances = 4 - len(out)
        time_left = hard_deadline - time.time()
        instance_deadline = time.time() + (time_left / remaining_instances) * 0.9
        
        colouring = None
        rng_solver = Random(0)
        
        # Try to solve
        colouring = solve_instance(n, edges, lists, instance_deadline, rng_solver)
        
        if colouring is None:
            # Retry with different random seeds if the solver uses randomness (currently it doesn't for ordering, 
            # but we can add a tie-breaker randomness if we want. For now, retrying is just a safety net 
            # in case of 'timeout' or logical errors, though the algorithm is deterministic.)
            for retry_seed in range(1, 10):
                if time.time() > instance_deadline:
                    break
                # Note: The current solver is deterministic. 
                # To make retries useful, we would need to introduce randomness in variable ordering or color ordering.
                # Let's add a small hook for that in the solver if we can't solve it.
                # For now, just retry the same deterministic solver? No, that's pointless.
                # If it failed, it will fail again.
                # However, the prompt says "random list... random graph". The solver logic is deterministic.
                # If it times out, a retry won't help unless we change the search strategy.
                # Given the small size, it should solve quickly.
                pass
                
        if colouring is None:
            colouring = [0] * n
        
        colouring = [int(c) for c in colouring]
        out.append(colouring)

    return out

# EVOLVE-BLOCK-END
