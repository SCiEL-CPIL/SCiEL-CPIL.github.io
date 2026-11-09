    def dpll():
        if time.time() > deadline:
            return False
            
        # Loop for UP and PLE until fixed point
        while True:
            if not propagate():
                return False
            if all(a != -1 for a in assignment):
                for c in clauses:
                    if not is_satisfied(c):
                        return False
                return True
            if not pure_literal_elimination():
                break
        
        # Branch
        var = -1
        for i in range(n):
            if assignment[i] == -1:
                var = i
                break
                
        if var == -1:
            # All assigned
            for c in clauses:
                if not is_satisfied(c):
                    return False
            return True

        # Try True
        assignment[var] = True
        if dpll():
            return True
        assignment[var] = False
        if dpll():
            return True
        assignment[var] = -1
        return False
