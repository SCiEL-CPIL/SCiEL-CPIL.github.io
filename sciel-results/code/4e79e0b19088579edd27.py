import itertools
import random

def is_2_colorable(n, edges):
    # edges: list of tuples/lists of 5 vertices
    # Returns True if 2-colorable, False otherwise
    
    # Pre-process: map vertices to indices 0..n-1
    # Use a list for colors: -1 for unassigned, 0 for Red, 1 for Blue
    
    def check_conflict(colors):
        for e in edges:
            c0, c1, c2, c3, c4 = colors[e[0]], colors[e[1]], colors[e[2]], colors[e[3]], colors[e[4]]
            # If all assigned
            if -1 not in (c0, c1, c2, c3, c4):
                # Check if monochromatic
                if c0 == c1 == c2 == c3 == c4:
                    return True
        return False

    def propagate(colors):
        # Returns (success, new_colors, changed)
        # If conflict, success=False
        # If assignment made, changed=True
        changed = True
        while changed:
            changed = False
            for e in edges:
                c0, c1, c2, c3, c4 = colors[e[0]], colors[e[1]], colors[e[2]], colors[e[3]], colors[e[4]]
                assigned = [c for c in (c0, c1, c2, c3, c4) if c != -1]
                unassigned_idx = [i for i, c in enumerate((c0, c1, c2, c3, c4)) if c == -1]
                
                if len(assigned) == 5:
                    if len(set(assigned)) == 1:
                        return False, colors, True # Conflict
                elif len(assigned) == 4:
                    # 4 assigned, 1 unassigned
                    if len(set(assigned)) == 1:
                        # All 4 same, 5th must be different
                        val = 1 - assigned[0]
                        idx = unassigned_idx[0]
                        v = e[idx]
                        if colors[v] == -1:
                            colors[v] = val
                            changed = True
                        elif colors[v] != val:
                            return False, colors, True # Conflict
        
        return True, colors, changed

    def backtrack(colors, v_idx):
        if v_idx == n:
            return True
        
        v = v_idx
        # Try 0 and 1
        for c in (0, 1):
            new_colors = colors[:]
            new_colors[v] = c
            success, new_colors, _ = propagate(new_colors)
            if success:
                if backtrack(new_colors, v_idx + 1):
                    return True
        return False

    colors = [-1] * n
    success, colors, _ = propagate(colors)
    if not success:
        return False
    return backtrack(colors, 0)

def construct():
    n = 9
    all_edges = [list(e) for e in itertools.combinations(range(n), 5)]
    
    # Start with all edges
    current_edges = all_edges[:]
    
    # We will try to remove edges
    # To find a smaller minimal hypergraph, we can shuffle the order of removal attempts
    # and run the minimization process multiple times, keeping the best result
    
    best_edges = current_edges[:]
    
    # Run minimization a few times with different random orders
    for _ in range(10):
        edges_to_try = current_edges[:]
        random.shuffle(edges_to_try)
        
        working_edges = set(tuple(e) for e in current_edges)
        
        for e in edges_to_try:
            e_tuple = tuple(e)
            if e_tuple not in working_edges:
                continue
            
            # Try removing e
            candidate_edges = [list(x) for x in working_edges if x != e_tuple]
            
            # Check if candidate is 2-colorable
            # If it IS 2-colorable, we cannot remove e (it was necessary for non-2-colorability)
            # If it is NOT 2-colorable, we can remove e
            
            if is_2_colorable(n, candidate_edges):
                # It became 2-colorable, so we must keep e
                pass
            else:
                # It remains non-2-colorable, so we can remove e
                working_edges.remove(e_tuple)
                
        # Convert back to list
        result_edges = [list(e) for e in working_edges]
        
        if len(result_edges) < len(best_edges):
            best_edges = result_edges
            current_edges = result_edges[:] # Update base for next iteration? 
            # Actually, starting from the new smaller set might not be better for finding even smaller ones?
            # Minimalization is a local search. Starting from the minimal set found, 
            # re-shuffling and trying to remove more might not work if it's already minimal.
            # But maybe a different order allows different removals?
            # If it's minimal, no single edge removal works. 
            # So further iterations on the same set will yield the same set.
            # So we should probably restart from the original dense set or the best found so far?
            # If we restart from best found, and it's minimal, we won't improve.
            # So we should restart from the ORIGINAL set each time? 
            # Or just do one pass?
            # A single pass of "try to remove each edge in random order" is a standard minimalization heuristic.
            # Doing it multiple times from the start with different orders is better.
            pass

    return {'n': n, 'edges': best_edges}
