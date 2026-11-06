def is_2_colorable(n, edges):
    # colors: -1 unassigned, 0 black, 1 white
    # We will use a list of lists or just a list and copy it.
    
    def propagate(colors):
        changed = True
        while changed:
            changed = False
            for e in edges:
                c0 = sum(1 for v in e if colors[v] == 0)
                c1 = sum(1 for v in e if colors[v] == 1)
                uncolored = [v for v in e if colors[v] == -1]
                
                if c0 == 4 and uncolored:
                    colors[uncolored[0]] = 1
                    changed = True
                elif c1 == 4 and uncolored:
                    colors[uncolored[0]] = 0
                    changed = True
                elif c0 == 5 or c1 == 5:
                    return None # Contradiction
        return colors

    def backtrack(idx, colors):
        if idx == n:
            # Check all edges
            for e in edges:
                c0 = sum(1 for v in e if colors[v] == 0)
                c1 = sum(1 for v in e if colors[v] == 1)
                if c0 == 5 or c1 == 5:
                    return False
            return True
        
        # Try color 0
        new_colors = colors.copy()
        new_colors[idx] = 0
        propagated = propagate(new_colors)
        if propagated is not None and backtrack(idx + 1, propagated):
            return True
            
        # Try color 1
        new_colors = colors.copy()
        new_colors[idx] = 1
        propagated = propagate(new_colors)
        if propagated is not None and backtrack(idx + 1, propagated):
            return True
        
        return False

    initial_colors = [-1] * n
    return backtrack(0, initial_colors)
