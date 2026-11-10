    def backtrack():
        # Find unassigned vertex with most constraints (MRV)
        min_count = n + 1
        best_v = -1
        for v in range(n):
            if assignment[v] == -1:
                deg = 0
                for e in edges_list:
                    if v in e:
                        deg += 1
                if deg < min_count:
                    min_count = deg
                    best_v = v
                    
        if best_v == -1:
            return True
            
        old_assignment = assignment.copy()
        
        # Try 0
        assignment[best_v] = 0
        if propagate():
            if backtrack():
                return True
        
        # Restore
        for i in range(n):
            assignment[i] = old_assignment[i]
            
        # Try 1
        assignment[best_v] = 1
        if propagate():
            if backtrack():
                return True
        
        # Restore
        for i in range(n):
            assignment[i] = old_assignment[i]
            
        return False
