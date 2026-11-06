import numpy as np
import random
import time

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    C = len(clauses)
    
    # Convert to 0-based indices
    # clauses[i] = [lit1, lit2, lit3] where lit is var*2 + (0 if pos, 1 if neg)? 
    # Let's stick to signed 1-based literals as input, but convert to 0-based internally for speed.
    # lit > 0 means var = lit - 1, sign = +1
    # lit < 0 means var = -lit - 1, sign = -1
    
    def lit_to_var_sign(lit):
        if lit > 0:
            return lit - 1, 1
        else:
            return -lit - 1, -1

    # Preprocess: build clause list with 0-based vars and signs
    # clause_vars[i] = [(var, sign), (var, sign), (var, sign)]
    clause_vars = []
    for clause in clauses:
        cv = []
        for lit in clause:
            v, s = lit_to_var_sign(lit)
            cv.append((v, s))
        clause_vars.append(cv)
    
    # Data structures for DPLL
    # assignment: list of None, True, False
    # active_clauses: list of indices of clauses not yet satisfied
    
    def dpll(assignment, active_clauses):
        # 1. Check if all clauses are satisfied
        if not active_clauses:
            return assignment
        
        # 2. Check for conflicts (empty clause)
        # In our representation, if a clause has 0 literals left, it's unsat.
        # We handle this by filtering clauses that become empty during propagation.
        
        # 3. Unit Propagation
        changed = True
        while changed:
            changed = False
            new_active = []
            conflict = False
            
            for ci in active_clauses:
                cv = clause_vars[ci]
                # Filter out satisfied literals
                new_cv = []
                sat = False
                for (v, s) in cv:
                    val = assignment[v]
                    if val is not None:
                        if (s == 1 and val) or (s == -1 and not val):
                            sat = True
                            break
                        # If assigned and not satisfying, it's a false literal, ignore it
                    else:
                        new_cv.append((v, s))
                
                if sat:
                    continue # Clause satisfied, remove from active
                
                if not new_cv:
                    # Conflict: clause has no unassigned literals and was not satisfied
                    conflict = True
                    break
                
                if len(new_cv) == 1:
                    # Unit clause: must assign the remaining variable
                    v, s = new_cv[0]
                    assignment[v] = (s == 1)
                    changed = True
                    # Note: We don't add to new_active here because it's now satisfied 
                    # (or will be handled in next iteration if we keep it, but it's sat now)
                    # Actually, if we assign it, the clause is satisfied.
                    continue
                
                new_active.append(ci)
            
            if conflict:
                return None
            active_clauses = new_active
            
            if not active_clauses:
                return assignment
        
        # 4. Pure Literal Elimination
        # Find variables that appear only with one sign in active clauses
        # If a variable is pure, assign it to satisfy those clauses.
        
        # Build count of positive and negative occurrences for unassigned vars in active clauses
        pos_count = [0] * n
        neg_count = [0] * n
        
        for ci in active_clauses:
            cv = clause_vars[ci]
            for (v, s) in cv:
                if assignment[v] is None:
                    if s == 1:
                        pos_count[v] += 1
                    else:
                        neg_count[v] += 1
        
        pure_vars = []
        for v in range(n):
            if assignment[v] is None:
                if pos_count[v] > 0 and neg_count[v] == 0:
                    pure_vars.append((v, True))
                elif neg_count[v] > 0 and pos_count[v] == 0:
                    pure_vars.append((v, False))
        
        if pure_vars:
            # Assign all pure literals
            for v, val in pure_vars:
                assignment[v] = val
            
            # Recurse or loop back to propagation
            # It's safer to loop back to propagation to handle any new unit clauses
            return dpll(assignment, active_clauses)
        
        # 5. Variable Selection
        # Pick a variable with highest occurrence in active clauses (Most Constrained)
        var_occurrences = [0] * n
        for ci in active_clauses:
            cv = clause_vars[ci]
            for (v, s) in cv:
                if assignment[v] is None:
                    var_occurrences[v] += 1
        
        best_var = -1
        max_occ = -1
        for v in range(n):
            if assignment[v] is None and var_occurrences[v] > max_occ:
                max_occ = var_occurrences[v]
                best_var = v
        
        if best_var == -1:
            # All variables assigned, but active_clauses not empty?
            # This should not happen if propagation worked correctly.
            # If we get here, it means there are no unassigned vars but clauses are active.
            # This implies a conflict that wasn't caught? Or just empty active clauses?
            # If active_clauses is not empty, and all vars assigned, we check satisfaction.
            # But we already checked satisfaction in step 1.
            # So this case is a conflict.
            return None
        
        # 6. Branching
        # Try True
        assignment[best_var] = True
        result = dpll(assignment, active_clauses)
        if result is not None:
            return result
        
        # Try False
        assignment[best_var] = False
        result = dpll(assignment, active_clauses)
        if result is not None:
            return result
        
        # Backtrack
        assignment[best_var] = None
        return None

    start_time = time.time()
    deadline = start_time + 290.0
    
    # Initialize assignment
    assignment = [None] * n
    active_clauses = list(range(C))
    
    # Run DPLL
    result = dpll(assignment, active_clauses)
    
    if result is not None:
        # Convert None to False (shouldn't happen if solved)
        return [bool(x) for x in result]
    
    # If DPLL fails (should not happen for SAT instances), fallback to random
    return [False] * n
