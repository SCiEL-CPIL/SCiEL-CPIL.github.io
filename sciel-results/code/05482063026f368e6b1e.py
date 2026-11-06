def unit_propagate(clauses, assignment):
    changed = True
    while changed:
        changed = False
        new_clauses = []
        for clause in clauses:
            # Determine status of each literal
            # 1. Satisfied?
            # 2. All False? (Conflict)
            # 3. One Unassigned? (Unit)
            
            unassigned = []
            for l in clause:
                var = abs(l) - 1
                if var in assignment:
                    val = assignment[var]
                    if (val and l > 0) or (not val and l < 0):
                        # Satisfied
                        unassigned = None # Break early
                        break
                    else:
                        # False
                        pass
                else:
                    unassigned.append(l)
            
            if unassigned is None:
                continue # Satisfied, drop clause
            
            if len(unassigned) == 0:
                return None, None, True # Conflict
            
            if len(unassigned) == 1:
                l = unassigned[0]
                var = abs(l) - 1
                assignment[var] = (l > 0)
                changed = True
                # Clause is satisfied by this assignment, so drop it
            else:
                # Keep the unassigned literals
                new_clauses.append(unassigned)
                
    return new_clauses, assignment, False
