def is_2_colorable(n, edges):
    if not edges:
        return True
    
    m = len(edges)
    
    # Precompute vertex to edges mapping
    v_to_e = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            v_to_e[v].append(i)
            
    assignment = [-1] * n
    
    # We will use a recursive backtracking.
    # To make it fast, we implement unit propagation.
    
    def check_edge(i):
        """
        Checks status of edge i.
        Returns:
        0: OK (mixed or empty unassigned)
        1: Unit (exactly 1 unassigned, others same color -> forces the unassigned)
        -1: Conflict (all assigned and monochromatic)
        """
        unassigned_v = None
        count_unassigned = 0
        c0 = False
        c1 = False
        
        for v in edges[i]:
            if assignment[v] == -1:
                count_unassigned += 1
                unassigned_v = v
            elif assignment[v] == 0:
                c0 = True
            else:
                c1 = True
                
        if count_unassigned == 0:
            if c0 and c1:
                return 0
            else:
                return -1
        elif count_unassigned == 1:
            if c0 and c1:
                return 0
            elif c0:
                # Must be 1
                return (1, unassigned_v) # Force
            elif c1:
                # Must be 0
                return (0, unassigned_v) # Force
            else:
                # All unassigned? No, count is 1. So 4 are unassigned? 
                # Wait, if count_unassigned == 1, the other 4 are assigned.
                # If none are 0 and none are 1, that's impossible if they are assigned.
                # So this case shouldn't happen if logic is correct.
                return 0
        else:
            return 0

    # This check_edge is a bit messy with return types.
    # Let's stick to a simpler propagation loop.
    
    def solve():
        # Backtracking with unit propagation
        # We'll use a stack for assignments to allow undoing.
        # Or just recursive with global state and undo.
        
        # To keep it simple and correct, we'll use a standard DPLL structure.
        # Given small size, we can afford some overhead.
        
        def propagate():
            # Returns False if conflict, True if no conflict.
            # Also returns list of forced assignments made?
            # No, let's just apply them directly and track them for undo.
            forced = []
            changed = True
            while changed:
                changed = False
                for i in range(m):
                    unassigned_v = None
                    count_unassigned = 0
                    c0 = False
                    c1 = False
                    
                    for v in edges[i]:
                        if assignment[v] == -1:
                            count_unassigned += 1
                            unassigned_v = v
                        elif assignment[v] == 0:
                            c0 = True
                        else:
                            c1 = True
                            
                    if count_unassigned == 0:
                        if not (c0 and c1):
                            return False, forced
                    elif count_unassigned == 1:
                        if c0 and not c1:
                            # Force unassigned_v to 1
                            if assignment[unassigned_v] == -1:
                                assignment[unassigned_v] = 1
                                forced.append(unassigned_v)
                                changed = True
                            elif assignment[unassigned_v] != 1:
                                return False, forced
                        elif c1 and not c0:
                            # Force unassigned_v to 0
                            if assignment[unassigned_v] == -1:
                                assignment[unassigned_v] = 0
                                forced.append(unassigned_v)
                                changed = True
                            elif assignment[unassigned_v] != 0:
                                return False, forced
            return True, forced

        def backtrack():
            # Check if all assigned
            # Actually, if no conflicts and no more forces, we pick a variable.
            
            # First, try to propagate
            # But propagation depends on current assignment.
            # We should propagate at the start of each node.
            
            # This structure is getting complex to manage undo.
            # Alternative: Since N is small (<=26), we can just iterate through variables 0..N-1.
            # Pick first unassigned.
            # Try 0, recurse.
            # Try 1, recurse.
            # Before recursing, check if this assignment causes immediate conflict.
            # And use propagation to force others.
            
            # Let's implement a cleaner DPLL.
            
            pass

    # Given the complexity of writing a perfect DPLL in this format, 
    # and the fact that N is very small, a simpler backtracking with 
    # "most constrained variable" heuristic might suffice and be faster to code correctly.
    
    # Heuristic:
    # 1. Check for any edge that is monochromatic (conflict).
    # 2. Check for any edge that has 4 assigned same color, 1 unassigned -> force the 1.
    # 3. If no forces, pick an unassigned variable. 
    #    Heuristic: Pick variable that appears in the most "critical" edges?
    #    Or just the first one.
    
    def backtrack(assigned_count):
        if assigned_count == n:
            # Final check
            for e in edges:
                colors = [assignment[v] for v in e]
                if all(c == 0 for c in colors) or all(c == 1 for c in colors):
                    return False
            return True
            
        # 1. Check for conflicts and forces
        forced_var = -1
        forced_val = -1
        for e in edges:
            unassigned_in_e = [v for v in e if assignment[v] == -1]
            if len(unassigned_in_e) == 0:
                colors = [assignment[v] for v in e]
                if all(c == 0 for c in colors) or all(c == 1 for c in colors):
                    return False
            elif len(unassigned_in_e) == 1:
                v = unassigned_in_e[0]
                # Check if other 4 are same color
                others = [assignment[u] for u in e if u != v]
                if all(c == 0 for c in others):
                    forced_var = v
                    forced_val = 1
                elif all(c == 1 for c in others):
                    forced_var = v
                    forced_val = 0
                    
        if forced_var != -1:
            assignment[forced_var] = forced_val
            res = backtrack(assigned_count + 1)
            assignment[forced_var] = -1
            return res
            
        # 2. Pick a variable
        # Heuristic: Pick variable with highest degree?
        # Precompute degrees
        # Or just pick first unassigned.
        var_to_assign = -1
        for v in range(n):
            if assignment[v] == -1:
                var_to_assign = v
                break
                
        if var_to_assign == -1:
            return True
            
        # Try 0
        assignment[var_to_assign] = 0
        # Check if this creates a monochromatic edge
        valid = True
        for e in edges:
            if var_to_assign in e:
                # Check if all others are 0
                # If so, conflict
                # We can just let the next recursion handle it?
                # No, pruning early is better.
                # But checking all edges is O(E*5).
                # Let's just recurse. The base case or force check will catch it.
                # Actually, if we set v=0, and an edge has v and four 0s, 
                # the next call will see 5 zeros and return False.
                # This is fine.
                pass
                
        if backtrack(assigned_count + 1):
            return True
            
        # Try 1
        assignment[var_to_assign] = 1
        if backtrack(assigned_count + 1):
            return True
            
        assignment[var_to_assign] = -1
        return False
        
    return backtrack(0)

def construct():
    import random
    from itertools import combinations
    
    # We want to find a minimal non-2-colorable hypergraph.
    # Strategy:
    # 1. Generate a random hypergraph that is definitely non-2-colorable.
    #    High density helps.
    # 2. Verify it's non-2-colorable.
    # 3. Iteratively remove edges if possible.
    
    best_edges = None
    best_n = 0
    
    # Try a few different n values
    for n in [14, 15, 16]:
        all_edges = list(combinations(range(n), 5))
        
        # Start with a random subset, biased towards high density
        # Or start with all and remove? 
        # Removing from all is safer to ensure non-colorability initially.
        # But all edges is huge. 
        # Let's start with a random set of size ~50-100.
        
        found = False
        current_edges = []
        
        for attempt in range(10):
            random.seed(42 + n + attempt)
            # Sample edges
            sample_size = 80
            edges = random.sample(all_edges, sample_size)
            edges = [list(e) for e in edges]
            
            if not is_2_colorable(n, edges):
                current_edges = edges
                found = True
                break
        
        if not found:
            continue
            
        # Minimize
        # Iterate through edges and try to remove them.
        # To be efficient, we can shuffle and process in batches.
        
        changed = True
        while changed:
            changed = False
            # Shuffle to avoid bias
            random.shuffle(current_edges)
            
            # Iterate through edges
            for i in range(len(current_edges) - 1, -1, -1):
                e_to_remove = current_edges[i]
                temp_edges = current_edges[:i] + current_edges[i+1:]
                
                # Check if still non-2-colorable
                if not is_2_colorable(n, temp_edges):
                    # Can remove
                    current_edges.pop(i)
                    changed = True
                    # Restart the pass? 
                    # If we remove one, the status of other edges might change.
                    # Restarting is safer and often finds more reductions.
                    break
        
        if best_edges is None or len(current_edges) < len(best_edges):
            best_edges = current_edges
            best_n = n
            
    if best_edges is None:
        # Fallback
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
        
    return {'n': best_n, 'edges': best_edges}
