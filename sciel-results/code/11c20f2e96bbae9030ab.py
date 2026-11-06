import time
import random
import numpy as np

def solve(instance):
    n = instance["n"]
    orig_clauses = instance["clauses"]
    m = len(orig_clauses)
    
    # 1. Preprocessing: Unit Propagation
    assignment = [None] * n
    active_clauses = [c[:] for c in orig_clauses]
    
    # Unit Propagation
    changed = True
    while changed:
        changed = False
        for i in range(len(active_clauses)):
            if len(active_clauses[i]) == 0:
                return [random.random() < 0.5 for _ in range(n)]
            
            if len(active_clauses[i]) == 1:
                lit = active_clauses[i][0]
                var = abs(lit) - 1
                val = lit > 0
                
                if assignment[var] is not None:
                    if assignment[var] != val:
                        return [random.random() < 0.5 for _ in range(n)]
                    else:
                        active_clauses[i] = []
                        changed = True
                        continue
                
                assignment[var] = val
                changed = True
                
                for j in range(len(active_clauses)):
                    if len(active_clauses[j]) == 0:
                        continue
                    new_clause = []
                    clause_satisfied = False
                    for l in active_clauses[j]:
                        v = abs(l) - 1
                        if v == var:
                            if (l > 0 and val) or (l < 0 and not val):
                                clause_satisfied = True
                                break
                            else:
                                continue
                        else:
                            new_clause.append(l)
                    
                    if clause_satisfied:
                        active_clauses[j] = []
                    else:
                        active_clauses[j] = new_clause
                break
    
    if all(a is not None for a in assignment):
        for c in orig_clauses:
            sat = False
            for l in c:
                v = abs(l) - 1
                if assignment[v] == (l > 0):
                    sat = True
                    break
            if not sat:
                return [random.random() < 0.5 for _ in range(n)]
        return assignment

    # 2. Robust WalkSAT
    # Initialize assignment
    for i in range(n):
        if assignment[i] is None:
            assignment[i] = random.random() < 0.5
            
    # Prepare data structures
    # var_to_clauses: list of lists of clause indices
    var_to_clauses = [[] for _ in range(n)]
    # clause_vars: list of lists of variable indices (0-based)
    # clause_signs: list of lists of signs (+1/-1)
    clause_vars = []
    clause_signs = []
    
    for i, c in enumerate(orig_clauses):
        cv = []
        cs = []
        for l in c:
            v = abs(l) - 1
            # Safety check: if variable is out of bounds, skip or handle?
            # Assuming valid input based on task description, but guard just in case.
            if 0 <= v < n:
                cv.append(v)
                cs.append(1 if l > 0 else -1)
                var_to_clauses[v].append(i)
            else:
                # If out of bounds, this literal is never satisfied/unsatisfied by our assignment?
                # Actually, if the instance is malformed, we can't solve it properly.
                # We'll treat it as a constant that is never satisfied to be safe? 
                # Or just ignore? Ignoring might make it satisfiable when it shouldn't be.
                # Given the crash, this might be the issue.
                # For now, we assume valid input but the crash was due to uninitialized numpy.
                # If we use lists, we don't have uninitialized memory.
                pass
        clause_vars.append(cv)
        clause_signs.append(cs)
        
    # Initial satisfaction check
    sat_count = [0] * m
    for i in range(m):
        for j in range(len(clause_vars[i])):
            v = clause_vars[i][j]
            s = clause_signs[i][j]
            val = assignment[v]
            if (s == 1 and val) or (s == -1 and not val):
                sat_count[i] += 1
                
    num_unsat = sum(1 for c in sat_count if c == 0)
    
    if num_unsat == 0:
        return assignment
        
    start_time = time.time()
    deadline = start_time + 290.0
    
    rng = random.Random(42)
    
    best_num_unsat = num_unsat
    best_assignment = assignment[:]
    
    phase = assignment[:]
    
    steps = 0
    current_restart_limit = 10000
    max_restart_limit = 500000
    
    while time.time() < deadline:
        if num_unsat == 0:
            return assignment
        
        if num_unsat < best_num_unsat:
            best_num_unsat = num_unsat
            best_assignment = assignment[:]
            
        if steps > current_restart_limit:
            # Restart with phase saving
            assignment = phase[:]
            
            # Recompute sat_count
            sat_count = [0] * m
            for i in range(m):
                for j in range(len(clause_vars[i])):
                    v = clause_vars[i][j]
                    s = clause_signs[i][j]
                    val = assignment[v]
                    if (s == 1 and val) or (s == -1 and not val):
                        sat_count[i] += 1
            num_unsat = sum(1 for c in sat_count if c == 0)
            
            steps = 0
            current_restart_limit = int(current_restart_limit * 1.5)
            if current_restart_limit > max_restart_limit:
                current_restart_limit = max_restart_limit
            continue
        
        # Pick a random unsatisfied clause
        # Optimization: Maintain a list of unsat clauses? 
        # For m ~ 450, scanning is cheap.
        unsat_indices = [i for i, c in enumerate(sat_count) if c == 0]
        if not unsat_indices:
            return assignment
            
        unsat_clause_idx = rng.choice(unsat_indices)
        
        # Get variables in this clause
        cv = clause_vars[unsat_clause_idx]
        cs = clause_signs[unsat_clause_idx]
        
        # Heuristic: Weighted random
        best_score = -float('inf')
        best_var = None
        
        for k in range(len(cv)):
            var_idx = cv[k]
            
            # Get all clauses containing this variable
            c_indices = var_to_clauses[var_idx]
            
            # Calculate gain/loss
            gains = 0
            losses = 0
            
            for c_idx in c_indices:
                old_count = sat_count[c_idx]
                
                # Check if flipping var_idx changes the satisfaction of this clause
                # The literal is at some position in clause c_idx
                # We need to know the sign of var_idx in clause c_idx
                # We didn't store (clause, pos) for each var efficiently.
                # Let's find the sign.
                # Since n is small, we can just check the clause.
                
                # Find sign of var_idx in clause c_idx
                sign_in_clause = 0
                for j in range(len(clause_vars[c_idx])):
                    if clause_vars[c_idx][j] == var_idx:
                        sign_in_clause = clause_signs[c_idx][j]
                        break
                
                if sign_in_clause == 0:
                    continue # Should not happen if var_to_clauses is correct
                
                # Current val
                cur_val = assignment[var_idx]
                # Is this literal satisfied currently?
                cur_lit_sat = (sign_in_clause == 1 and cur_val) or (sign_in_clause == -1 and not cur_val)
                
                # New val
                new_val = not cur_val
                # Is this literal satisfied after flip?
                new_lit_sat = (sign_in_clause == 1 and new_val) or (sign_in_clause == -1 and not new_val)
                
                # Delta
                if cur_lit_sat and not new_lit_sat:
                    # This literal was contributing 1 to sat_count, now 0
                    # If old_count was 1, it becomes 0 (Loss)
                    # If old_count > 1, it stays > 0
                    if old_count == 1:
                        losses += 1
                elif not cur_lit_sat and new_lit_sat:
                    # This literal was contributing 0, now 1
                    # If old_count was 0, it becomes 1 (Gain)
                    if old_count == 0:
                        gains += 1
                
            score = gains * 1.0 - losses * 1.5
            
            if score > best_score:
                best_score = score
                best_var = var_idx
                
        if best_var is None:
            best_var = cv[0]
            
        # Flip the variable
        assignment[best_var] = not assignment[best_var]
        phase[best_var] = assignment[best_var]
        
        # Update sat_count
        for c_idx in var_to_clauses[best_var]:
            # Find sign of best_var in clause c_idx
            sign_in_clause = 0
            for j in range(len(clause_vars[c_idx])):
                if clause_vars[c_idx][j] == best_var:
                    sign_in_clause = clause_signs[c_idx][j]
                    break
            
            if sign_in_clause == 0:
                continue
            
            old_count = sat_count[c_idx]
            
            cur_val = assignment[best_var]
            # Note: assignment[best_var] is already flipped.
            # We need to compare new state vs old state.
            # Old val was not cur_val.
            
            # New literal satisfaction
            new_lit_sat = (sign_in_clause == 1 and cur_val) or (sign_in_clause == -1 and not cur_val)
            
            # Old literal satisfaction
            old_lit_sat = not new_lit_sat # Because flipping the only variable in the clause that changed?
            # Wait, if the clause has other variables, they didn't change.
            # So the change in sat_count is exactly the change in this literal's contribution.
            # If new_lit_sat is True and old was False -> +1
            # If new_lit_sat is False and old was True -> -1
            
            if new_lit_sat:
                # It is now satisfied. Was it before?
                # Old val = not cur_val
                old_lit_sat_check = (sign_in_clause == 1 and (not cur_val)) or (sign_in_clause == -1 and cur_val)
                if not old_lit_sat_check:
                    sat_count[c_idx] += 1
            else:
                # It is now unsatisfied. Was it before?
                old_lit_sat_check = (sign_in_clause == 1 and (not cur_val)) or (sign_in_clause == -1 and cur_val)
                if old_lit_sat_check:
                    sat_count[c_idx] -= 1
            
        # Update num_unsat
        # Instead of recounting, we can track changes, but recounting m=450 is fast.
        # Or just update num_unsat based on the clauses we touched.
        # Let's just recount for simplicity and safety, as m is small.
        num_unsat = sum(1 for c in sat_count if c == 0)
        
        steps += 1
        
    return best_assignment
