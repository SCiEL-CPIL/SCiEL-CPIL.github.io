    def is_2_colorable(n, edges):
        if not edges:
            return True
        
        edge_list = [tuple(sorted(e)) for e in edges]
        m = len(edge_list)
        colors = [-1] * n
        
        # Precompute vertex to edges
        vertex_edges = [[] for _ in range(n)]
        for i, e in enumerate(edge_list):
            for v in e:
                vertex_edges[v].append(i)
        
        def propagate():
            """
            Returns False if contradiction.
            Modifies colors.
            Returns list of (vertex, old_color) to undo? 
            Or just return True/False and let caller handle state?
            Caller needs to undo.
            """
            changed = True
            while changed:
                changed = False
                for i in range(m):
                    e = edge_list[i]
                    c0 = 0
                    c1 = 0
                    c_un = 0
                    un_v = -1
                    for v in e:
                        col = colors[v]
                        if col == 0:
                            c0 += 1
                        elif col == 1:
                            c1 += 1
                        else:
                            c_un += 1
                            un_v = v
                    
                    if c0 == 5 or c1 == 5:
                        return False
                    if c0 == 4 and c_un == 1 and colors[un_v] == -1:
                        colors[un_v] = 1
                        changed = True
                    elif c1 == 4 and c_un == 1 and colors[un_v] == -1:
                        colors[un_v] = 0
                        changed = True
            return True

        def backtrack():
            # Find next unassigned
            next_v = -1
            for v in range(n):
                if colors[v] == -1:
                    next_v = v
                    break
            
            if next_v == -1:
                return True
            
            # Save state to restore later
            # Since n is small, we can just copy the list
            saved_colors = colors[:]
            
            # Try 0
            colors[next_v] = 0
            if propagate() and backtrack():
                return True
            
            # Try 1
            colors[next_v] = 1
            if propagate() and backtrack():
                return True
            
            # Restore state
            colors[:] = saved_colors
            return False
        
        return backtrack()
    