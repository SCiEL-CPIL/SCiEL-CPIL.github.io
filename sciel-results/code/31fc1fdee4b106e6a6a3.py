import time
import sys

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 290.0
    
    # DPLL Solver
    # Clauses are lists of signed integers (1-based, so var = abs(lit))
    # We will work with 0-based variables internally for array indexing, but keep literals as is or convert.
    # Let's convert to 0-based signed literals: var in [0, n-1], sign +/- 1.
    # lit = var + 1 if positive, -(var + 1) if negative.
    
    # Preprocess: convert to 0-based
    c_0based = []
    for c in clauses:
        new_c = []
        for lit in c:
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            new_c.append((v, s))
        c_0based.append(new_c)
        
    m = len(c_0based)
    
    # State:
    # assignment: list of 0, 1, or -1 for each variable (0 = unassigned, 1 = True, -1 = False)
    # active_clauses: list of indices of clauses that are not yet satisfied or falsified
    # clause_status: 0 = active, 1 = satisfied, -1 = falsified
    
    def dpll(assignment, active_clauses):
        # 1. Check for conflict
        for ci in active_clauses:
            clause = c_0based[ci]
            all_false = True
            for v, s in clause:
                if assignment[v] == 0:
                    all_false = False
                    break
                if assignment[v] == s:
                    all_false = False
                    break
            if all_false:
                return None # Conflict
        
        # 2. Unit Propagation
        changed = True
        while changed:
            changed = False
            new_active = []
            for ci in active_clauses:
                clause = c_0based[ci]
                # Count unassigned and check if satisfied
                unassigned = []
                is_sat = False
                is_fals = True
                for v, s in clause:
                    av = assignment[v]
                    if av == 0:
                        unassigned.append((v, s))
                        is_fals = False
                    elif av == s:
                        is_sat = True
                        is_fals = False
                        break
                    else:
                        # av == -s, so literal is false
                        pass
                
                if is_sat:
                    continue # Clause satisfied, remove from active
                if is_fals:
                    return None # Conflict
                
                if len(unassigned) == 1:
                    # Unit clause: must assign this variable
                    v, s = unassigned[0]
                    if assignment[v] != 0:
                        # Already assigned, check consistency
                        if assignment[v] != s:
                            return None
                        else:
                            continue # Already satisfied by previous assignment? No, if it was unassigned in this loop iteration, it's 0.
                    assignment[v] = s
                    changed = True
                    # After assigning, we need to re-evaluate all active clauses, so we break and restart loop
                    # Or we can continue and let the next pass handle it. 
                    # To be efficient, we restart the while loop immediately.
                    # But we need to ensure we don't miss other unit clauses in the same pass?
                    # Standard UP: apply all forced assignments, then check for conflicts.
                    # Let's just mark changed and continue scanning to find other units?
                    # Actually, assigning one variable might make another clause unit.
                    # It's safer to restart the while loop after any assignment to ensure all consequences are propagated.
                    # However, for n=50, even a simple re-scan is fine.
                    pass 
                else:
                    new_active.append(ci)
            
            if changed:
                active_clauses = new_active
                continue
            
            # If no changes, update active_clauses to new_active and break
            active_clauses = new_active
            break
            
        if not active_clauses:
            return assignment[:] # All clauses satisfied
        
        # 3. Pure Literal Elimination
        # Count positive and negative occurrences in active clauses
        pos_count = [0] * n
        neg_count = [0] * n
        for ci in active_clauses:
            clause = c_0based[ci]
            for v, s in clause:
                if assignment[v] == 0:
                    if s == 1:
                        pos_count[v] += 1
                    else:
                        neg_count[v] += 1
        
        changed = True
        while changed:
            changed = False
            for v in range(n):
                if assignment[v] != 0:
                    continue
                if pos_count[v] > 0 and neg_count[v] == 0:
                    assignment[v] = 1
                    changed = True
                    # Update counts? For simplicity, restart PLE loop
                    break
                elif neg_count[v] > 0 and pos_count[v] == 0:
                    assignment[v] = -1
                    changed = True
                    break
            
            if changed:
                # Re-run UP and PLE
                # Recurse or loop back?
                # To keep it simple and fast for n=50, we can just return dpll(assignment, active_clauses)
                # But that might be deep. Let's just restart the function logic.
                # Actually, PLE doesn't change clause satisfaction status directly, it just assigns variables.
                # So we need to re-run UP.
                # Let's implement PLE as a pre-step to branching, but if it assigns, we must UP.
                # So:
                return dpll(assignment, active_clauses)
        
        # 4. Branching
        # Select a variable
        # Heuristic: Variable in the most active clauses? Or part of the shortest clause?
        # Let's pick a variable from the first active clause.
        # Better: Pick variable with highest frequency in active clauses.
        var_freq = [0] * n
        for ci in active_clauses:
            clause = c_0based[ci]
            seen = set()
            for v, s in clause:
                if assignment[v] == 0 and v not in seen:
                    var_freq[v] += 1
                    seen.add(v)
        
        best_var = -1
        max_freq = -1
        for v in range(n):
            if assignment[v] == 0 and var_freq[v] > max_freq:
                max_freq = var_freq[v]
                best_var = v
        
        if best_var == -1:
            # Should not happen if active_clauses is not empty and UP/PLE didn't solve it
            # Find any unassigned var in active clauses
            for ci in active_clauses:
                for v, s in c_0based[ci]:
                    if assignment[v] == 0:
                        best_var = v
                        break
                if best_var != -1:
                    break
        
        if best_var == -1:
            return assignment[:] # All assigned, should be satisfied
        
        # Try True
        assignment[best_var] = 1
        result = dpll(assignment, active_clauses)
        if result is not None:
            return result
        
        # Try False
        assignment[best_var] = -1
        result = dpll(assignment, active_clauses)
        if result is not None:
            return result
        
        # Backtrack
        assignment[best_var] = 0
        return None

    # Initial call
    initial_assignment = [0] * n
    initial_active = list(range(m))
    
    solution = dpll(initial_assignment, initial_active)
    
    if solution is not None:
        # Convert 0-based assignment (1/-1) to list of booleans
        # 1 -> True, -1 -> False
        return [val == 1 for val in solution]
    
    # Fallback: If DPLL fails (shouldn't for satisfiable instances, but just in case of bugs or timeout),
    # return a random assignment or best effort.
    # Since the task is "planted satisfiable", DPLL should find it.
    # If it times out, we return what we have or random.
    # Given n=50, it won't time out.
    return [False] * n
