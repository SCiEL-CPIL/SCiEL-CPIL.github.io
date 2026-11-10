"""Model-and-optimize: Iterative DFS with Forward Checking and MRV for robust list coloring."""
# EVOLVE-BLOCK-START
import time
from random import Random


def construct():
    t0 = time.time()
    LIMIT = 120.0
    MARGIN = 10.0

    def deadline():
        return (time.time() - t0) > (LIMIT - MARGIN)

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
        """
        Solves the list coloring problem using:
        1. Randomized Greedy (fast pre-pass)
        2. Iterative DFS with Forward Checking and MRV
        """
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # --- Strategy 1: Randomized Greedy with multiple restarts ---
        # This is very fast and often sufficient for sparse graphs.
        rng = Random(42)
        for _ in range(50):
            if deadline():
                break
            order = list(range(n))
            rng.shuffle(order)
            colour = [None] * n
            valid = True
            for v in order:
                taken = set()
                for u in adj[v]:
                    if colour[u] is not None:
                        taken.add(colour[u])
                avail = [c for c in lists[v] if c not in taken]
                if avail:
                    colour[v] = rng.choice(avail)
                else:
                    valid = False
                    break
            if valid:
                # Verify
                is_valid = True
                for u, v in edges:
                    if colour[u] == colour[v] or colour[u] not in lists[u] or colour[v] not in lists[v]:
                        is_valid = False
                        break
                if is_valid:
                    return [int(c) for c in colour]

        # --- Strategy 2: Iterative DFS with Forward Checking and MRV ---
        # This is a robust solver for the remaining cases.
        
        # Initial domains
        domains = [set(lists[v]) for v in range(n)]
        assignment = [None] * n
        
        # Stack for DFS: (variable, value_index)
        # We will maintain a history of states to backtrack.
        # To optimize, we store the state of domains and assignment before each choice.
        
        # MRV Heuristic: Select unassigned variable with smallest domain
        def get_mrv_variable(unassigned_set):
            if not unassigned_set:
                return None
            min_size = float('inf')
            min_var = None
            for v in unassigned_set:
                size = len(domains[v])
                if size < min_size:
                    min_size = size
                    min_var = v
                    if min_size == 0:
                        break
            return min_var

        # We use a stack of (variable, chosen_value, previous_domain_state, previous_assignment_state)
        # But copying domains every step is expensive. 
        # Instead, we use a "undo" log or simply copy the domains since n is small (~108) and depth is limited.
        # Given n=108, copying a list of 108 sets of size <=3 is cheap.
        
        stack = []
        unassigned = set(range(n))
        
        # Initialize first variable
        v = get_mrv_variable(unassigned)
        if v is None:
            return [int(c) for c in assignment] # All assigned? Unlikely at start unless n=0
            
        # Push initial state
        # We don't push a "state" yet, we just start the loop.
        # We need to handle the initial selection.
        
        # To make it iterative, we simulate the recursion.
        # State: (current_variable, value_iterator_state, domains_snapshot, assignment_snapshot)
        # Actually, simpler:
        # Push (variable, values_to_try, domains_before, assignment_before)
        
        # Let's implement a standard iterative backtracking.
        
        # We need to track which value we are trying for the current variable.
        
        # Start with the first variable
        current_var = get_mrv_variable(unassigned)
        if current_var is None:
             return None # Should not happen
        
        # We will push frames onto the stack.
        # Frame: (var, values_list, domains_copy, assignment_copy)
        # But we want to try values one by one.
        
        # Let's use a different structure:
        # stack of (var, next_val_index, domains_snapshot, assignment_snapshot)
        
        # Initial setup
        init_domains = [set(d) for d in domains]
        init_assignment = [None] * n
        
        # We pick the first variable
        v0 = get_mrv_variable(unassigned)
        
        # If v0 is None, we are done.
        if v0 is None:
            return [int(c) for c in assignment]
            
        # We will iterate through values for v0
        vals0 = list(domains[v0])
        if not vals0:
            return None
            
        # Push the first frame
        # We store the state BEFORE assigning v0
        stack.append((v0, 0, init_domains, init_assignment))
        
        while stack:
            if deadline():
                return None
                
            var, val_idx, dom_snap, asg_snap = stack[-1]
            
            # Get the list of values for this variable in the snapshot
            vals = list(dom_snap[var])
            
            if val_idx >= len(vals):
                # No more values to try for this variable. Backtrack.
                stack.pop()
                continue
            
            val = vals[val_idx]
            
            # Prepare to assign 'val' to 'var'
            # Update assignment
            new_asg = asg_snap.copy()
            new_asg[var] = val
            
            # Update domains (Forward Checking)
            new_dom = [set(d) for d in dom_snap]
            fc_ok = True
            
            # Remove 'val' from neighbors' domains
            for u in adj[var]:
                if new_asg[u] is None: # Only if unassigned
                    if val in new_dom[u]:
                        new_dom[u].remove(val)
                        if not new_dom[u]:
                            fc_ok = False
                            break
                            
            if not fc_ok:
                # Try next value for this variable
                stack[-1] = (var, val_idx + 1, dom_snap, asg_snap)
                continue
            
            # Check if all variables are assigned
            # We can track a count of assigned variables
            # Or just check if any domain is empty? No, FC handles that.
            # We need to know if we are done.
            
            # Optimization: Keep track of unassigned count?
            # For simplicity, check if all are assigned.
            all_assigned = True
            for i in range(n):
                if new_asg[i] is None:
                    all_assigned = False
                    break
                    
            if all_assigned:
                # Verify the solution
                valid = True
                for u, v in edges:
                    if new_asg[u] == new_asg[v]:
                        valid = False
                        break
                    if new_asg[u] not in lists[u] or new_asg[v] not in lists[v]:
                        valid = False
                        break
                if valid:
                    return [int(c) for c in new_asg]
                else:
                    # Should not happen if FC is correct, but just in case
                    stack[-1] = (var, val_idx + 1, dom_snap, asg_snap)
                    continue
            
            # Select next variable using MRV
            next_vars = [i for i in range(n) if new_asg[i] is None]
            if not next_vars:
                # Should be handled by all_assigned
                stack[-1] = (var, val_idx + 1, dom_snap, asg_snap)
                continue
                
            next_var = get_mrv_variable(set(next_vars))
            
            if next_var is None or not new_dom[next_var]:
                # No valid next variable or empty domain (should be caught by FC)
                stack[-1] = (var, val_idx + 1, dom_snap, asg_snap)
                continue
                
            # Push new frame
            stack.append((next_var, 0, new_dom, new_asg))
            
        return None

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if deadline():
            n, edges, lists = instance(splits, seed)
            rng_fallback = Random(777)
            out.append([int(rng_fallback.choice(lists[v])) for v in range(n)])
            continue
        
        n, edges, lists = instance(splits, seed)
        colour = solve_instance(n, edges, lists)
        
        if colour is None:
             # Fallback to random if solver fails (should not happen)
             rng_fallback = Random(777)
             colour = [int(rng_fallback.choice(lists[v])) for v in range(n)]
             
        out.append(colour)

    return out
# EVOLVE-BLOCK-END
