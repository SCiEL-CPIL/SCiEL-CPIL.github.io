import time
import random

def solve(instance):
    n = instance["n"]
    orig_clauses = instance["clauses"]
    
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

    # 2. WalkSAT with Incremental Updates, Weighted Heuristic, Phase Saving, and Unsat List
    for i in range(n):
        if assignment[i] is None:
            assignment[i] = random.random() < 0.5
    
    m = len(orig_clauses)
    clause_sat = [False] * m
    sat_count = [0] * m
    var_to_clauses = [[] for _ in range(n)]
    var_literal_in_clause = [{} for _ in range(n)]
    
    for i, c in enumerate(orig_clauses):
        count = 0
        for l in c:
            v = abs(l) - 1
            var_to_clauses[v].append(i)
            var_literal_in_clause[v][i] = l
            if assignment[v] == (l > 0):
                count += 1
        sat_count[i] = count
        clause_sat[i] = (count > 0)
        
    num_unsat = sum(1 for s in clause_sat if not s)
    unsat_clauses = [i for i in range(m) if not clause_sat[i]]
    
    if num_unsat == 0:
        return assignment
    
    start_time = time.time()
    deadline = start_time + 290.0
    
    restart_limit = 1000
    rng = random.Random(42)
    
    best_num_unsat = num_unsat
    best_assignment = assignment[:]
    
    # Phase saving
    phase = assignment[:]
    
    steps = 0
    current_restart_limit = restart_limit
    
    while time.time() < deadline:
        if num_unsat == 0:
            return assignment
        
        if num_unsat < best_num_unsat:
            best_num_unsat = num_unsat
            best_assignment = assignment[:]
            
        if steps > current_restart_limit:
            # Restart with phase saving
            for i in range(n):
                assignment[i] = phase[i]
            
            for i, c in enumerate(orig_clauses):
                count = 0
                for l in c:
                    v = abs(l) - 1
                    if assignment[v] == (l > 0):
                        count += 1
                sat_count[i] = count
                clause_sat[i] = (count > 0)
            
            num_unsat = sum(1 for s in clause_sat if not s)
            unsat_clauses = [i for i in range(m) if not clause_sat[i]]
            
            steps = 0
            current_restart_limit = int(current_restart_limit * 1.5)
            if current_restart_limit > 100000:
                current_restart_limit = 100000
            continue
        
        # Pick a random unsatisfied clause from the list
        if not unsat_clauses:
            return assignment
            
        idx_pos = rng.randrange(len(unsat_clauses))
        unsat_clause_idx = unsat_clauses[idx_pos]
        
        clause = orig_clauses[unsat_clause_idx]
        
        # Weighted heuristic
        if rng.random() < 0.3:
            lit = rng.choice(clause)
            var_idx = abs(lit) - 1
        else:
            best_score = -float('inf')
            best_var = None
            for lit in clause:
                var_idx = abs(lit) - 1
                satisfied_gain = 0
                unsatisfied_loss = 0
                
                for c_idx in var_to_clauses[var_idx]:
                    lit_in_c = var_literal_in_clause[var_idx][c_idx]
                    old_count = sat_count[c_idx]
                    
                    if assignment[var_idx] == (lit_in_c > 0):
                        if old_count == 1:
                            unsatisfied_loss += 1
                    else:
                        if old_count == 0:
                            satisfied_gain += 1
                
                score = satisfied_gain * 1.0 - unsatisfied_loss * 1.5
                
                if score > best_score:
                    best_score = score
                    best_var = var_idx
            
            if best_var is None:
                best_var = abs(rng.choice(clause)) - 1
            var_idx = best_var
            
        # Flip the variable
        old_val = assignment[var_idx]
        new_val = not old_val
        assignment[var_idx] = new_val
        phase[var_idx] = new_val
        
        # Update sat_count and clause_sat
        for c_idx in var_to_clauses[var_idx]:
            lit_in_c = var_literal_in_clause[var_idx][c_idx]
            was_sat = clause_sat[c_idx]
            old_count = sat_count[c_idx]
            
            if old_val == (lit_in_c > 0):
                new_count = old_count - 1
            else:
                new_count = old_count + 1
                
            new_sat = (new_count > 0)
            
            if was_sat and not new_sat:
                num_unsat += 1
                unsat_clauses.append(c_idx)
            elif not was_sat and new_sat:
                num_unsat -= 1
                # Remove c_idx from unsat_clauses
                # Find index and swap-remove
                try:
                    pos = unsat_clauses.index(c_idx)
                    unsat_clauses[pos] = unsat_clauses[-1]
                    unsat_clauses.pop()
                except ValueError:
                    pass # Should not happen
            
            sat_count[c_idx] = new_count
            clause_sat[c_idx] = new_sat
            
        steps += 1
        
    # Final polish
    if time.time() < deadline - 10:
        polish_deadline = deadline - 5
        for _ in range(100):
            if num_unsat == 0:
                return assignment
            if time.time() > polish_deadline:
                break
            
            if num_unsat < best_num_unsat:
                best_num_unsat = num_unsat
                best_assignment = assignment[:]
            
            if not unsat_clauses:
                return assignment
                
            idx_pos = rng.randrange(len(unsat_clauses))
            unsat_clause_idx = unsat_clauses[idx_pos]
            
            clause = orig_clauses[unsat_clause_idx]
            
            best_score = -float('inf')
            best_var = None
            for lit in clause:
                var_idx = abs(lit) - 1
                satisfied_gain = 0
                unsatisfied_loss = 0
                
                for c_idx in var_to_clauses[var_idx]:
                    lit_in_c = var_literal_in_clause[var_idx][c_idx]
                    old_count = sat_count[c_idx]
                    
                    if assignment[var_idx] == (lit_in_c > 0):
                        if old_count == 1:
                            unsatisfied_loss += 1
                    else:
                        if old_count == 0:
                            satisfied_gain += 1
                
                score = satisfied_gain * 1.0 - unsatisfied_loss * 2.0
                
                if score > best_score:
                    best_score = score
                    best_var = var_idx
            
            if best_var is not None:
                old_val = assignment[best_var]
                new_val = not old_val
                assignment[best_var] = new_val
                phase[best_var] = new_val
                
                for c_idx in var_to_clauses[best_var]:
                    lit_in_c = var_literal_in_clause[best_var][c_idx]
                    was_sat = clause_sat[c_idx]
                    old_count = sat_count[c_idx]
                    
                    if old_val == (lit_in_c > 0):
                        new_count = old_count - 1
                    else:
                        new_count = old_count + 1
                        
                    new_sat = (new_count > 0)
                    
                    if was_sat and not new_sat:
                        num_unsat += 1
                        unsat_clauses.append(c_idx)
                    elif not was_sat and new_sat:
                        num_unsat -= 1
                        try:
                            pos = unsat_clauses.index(c_idx)
                            unsat_clauses[pos] = unsat_clauses[-1]
                            unsat_clauses.pop()
                        except ValueError:
                            pass
                    
                    sat_count[c_idx] = new_count
                    clause_sat[c_idx] = new_sat
    
    return best_assignment
