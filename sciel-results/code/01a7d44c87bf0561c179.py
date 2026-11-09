import time
import random

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Pre-process literals
    # clause_vars[i] = list of variable indices (0-based)
    # clause_signs[i] = list of booleans (True if positive)
    clause_vars = []
    clause_signs = []
    for c in clauses:
        vs = []
        ss = []
        for l in c:
            vs.append(abs(l) - 1)
            ss.append(l > 0)
        clause_vars.append(vs)
        clause_signs.append(ss)
        
    m = len(clauses)
    
    # Map variable to list of (clause_index, sign_in_clause)
    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j in range(len(clause_vars[i])):
            v = clause_vars[i][j]
            s = clause_signs[i][j]
            var_to_clauses[v].append((i, s))

    def is_satisfied(clause_idx, assignment):
        c_vars = clause_vars[clause_idx]
        c_signs = clause_signs[clause_idx]
        for v, s in zip(c_vars, c_signs):
            if assignment[v] == s:
                return True
        return False

    start_time = time.time()
    time_limit = 290.0
    
    best_assignment = None
    min_unsat_count = float('inf')
    
    run_id = 0
    while time.time() - start_time < time_limit:
        run_id += 1
        rng = random.Random(run_id)
        
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Initialize clause_sat and unsat_indices
        clause_sat = [False] * m
        unsat_indices = []
        for i in range(m):
            if is_satisfied(i, assignment):
                clause_sat[i] = True
            else:
                unsat_indices.append(i)
                
        current_unsat_count = len(unsat_indices)
        if current_unsat_count < min_unsat_count:
            min_unsat_count = current_unsat_count
            best_assignment = assignment[:]
            
        if not unsat_indices:
            return assignment
            
        # WalkSAT parameters
        p = 0.55 # Probability of random flip
        max_steps = 200000 # Generous budget, time check will stop it
        
        for step in range(max_steps):
            if time.time() - start_time > time_limit - 5:
                break
                
            if not unsat_indices:
                return assignment
                
            # Pick random unsat clause
            c_idx = rng.choice(unsat_indices)
            c_vars = clause_vars[c_idx]
            
            # Select variable to flip
            if rng.random() < p:
                v_to_flip = c_vars[rng.randrange(len(c_vars))]
            else:
                # Heuristic: Pick variable that minimizes damage
                best_v = None
                best_score = -float('inf')
                
                for v in c_vars:
                    # Calculate score for flipping v
                    # Score = (clauses satisfied by flip) - (clauses unsatisfied by flip)
                    # Note: c_idx is currently unsat, so flipping any v in c_vars 
                    # will satisfy c_idx (since it's a 3-SAT clause, at least one literal 
                    # must be true if we set the variable correctly? 
                    # Wait. If clause is (x OR y OR z), and it's unsat, then x=F, y=F, z=F.
                    # If we flip x to T, clause becomes T. So c_idx is always satisfied by flipping any of its vars.
                    # So +1 for c_idx.
                    
                    score = 1.0 # For satisfying c_idx
                    
                    # Check other clauses containing v
                    for u_idx, sign_in_u in var_to_clauses[v]:
                        if u_idx == c_idx:
                            continue
                        
                        # Current state of u_idx
                        was_sat = clause_sat[u_idx]
                        
                        # If we flip v, does u_idx change?
                        # u_idx is satisfied if any literal is true.
                        # If was_sat is False, u_idx is unsat.
                        # If we flip v, does it become sat?
                        # It becomes sat if the literal v in u_idx becomes true.
                        # Literal v in u_idx is true if assignment[v] XOR sign_in_u is True?
                        # Let's check:
                        # If sign_in_u is True (positive), literal is true if assignment[v] is True.
                        # If sign_in_u is False (negative), literal is true if assignment[v] is False.
                        # So literal is true iff assignment[v] == sign_in_u.
                        
                        # Current assignment[v] is A.
                        # New assignment[v] will be not A.
                        # Current literal value: L_old = (A == sign_in_u)
                        # New literal value: L_new = ((not A) == sign_in_u)
                        
                        # If was_sat is True:
                        #   Clause is sat. If we flip v, and v was the *only* satisfying literal, it becomes unsat.
                        #   If there was another satisfying literal, it stays sat.
                        #   To know this, we'd need to check other literals.
                        #   However, a simpler approximation often used:
                        #   Count how many unsat clauses v appears in (potential gains).
                        #   Count how many sat clauses v appears in (potential losses).
                        #   This is the "MinConflicts" style heuristic.
                        
                        # Let's use the precise check for small n.
                        if not was_sat:
                            # Clause is currently unsat.
                            # If flipping v satisfies it, +1
                            # It satisfies it if the new value of v makes the literal true.
                            new_val = not assignment[v]
                            if new_val == sign_in_u:
                                score += 1.0
                        else:
                            # Clause is currently sat.
                            # If flipping v makes it unsat, -1
                            # It makes it unsat if v was the ONLY satisfying literal.
                            # Check if any other literal in u_idx is satisfied.
                            other_sat = False
                            u_vars = clause_vars[u_idx]
                            u_signs = clause_signs[u_idx]
                            for k, uv in enumerate(u_vars):
                                if uv == v:
                                    continue
                                if assignment[uv] == u_signs[k]:
                                    other_sat = True
                                    break
                            if not other_sat:
                                score -= 1.0
                                
                    if score > best_score:
                        best_score = score
                        best_v = v
                        
                v_to_flip = best_v if best_v is not None else c_vars[0]
                
            # Perform flip
            assignment[v_to_flip] = not assignment[v_to_flip]
            
            # Update clause_sat and unsat_indices
            for u_idx, sign_in_u in var_to_clauses[v_to_flip]:
                was_sat = clause_sat[u_idx]
                is_now_sat = is_satisfied(u_idx, assignment)
                
                if was_sat == is_now_sat:
                    continue
                    
                clause_sat[u_idx] = is_now_sat
                
                if was_sat and not is_now_sat:
                    # Became unsat
                    unsat_indices.append(u_idx)
                elif not was_sat and is_now_sat:
                    # Became sat
                    # Remove from unsat_indices
                    # O(N) removal
                    try:
                        unsat_indices.remove(u_idx)
                    except ValueError:
                        pass # Should not happen if logic is correct

            # Update best
            current_unsat_count = len(unsat_indices)
            if current_unsat_count < min_unsat_count:
                min_unsat_count = current_unsat_count
                best_assignment = assignment[:]

    return best_assignment
