import random
import time

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290.0  # 10s safety margin
    
    # Precompute variable to clauses mapping
    var_to_clauses = [[] for _ in range(n)]
    for i, c in enumerate(clauses):
        for l in c:
            var_to_clauses[abs(l)-1].append(i)
            
    rng = random.Random(0)
    
    # Helper to check if clause is satisfied
    def is_satisfied(clause, assignment):
        for l in clause:
            if assignment[abs(l)-1] == (l > 0):
                return True
        return False

    # Main loop with restarts
    while time.time() < deadline:
        # Random initialization
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Initialize states
        clause_is_sat = [False] * len(clauses)
        var_unsat_count = [0] * n
        unsat_clause_indices = []
        
        for i, c in enumerate(clauses):
            sat = is_satisfied(c, assignment)
            clause_is_sat[i] = sat
            if not sat:
                unsat_clause_indices.append(i)
                for l in c:
                    var_unsat_count[abs(l)-1] += 1
                    
        if not unsat_clause_indices:
            return assignment
            
        # WalkSAT Loop
        steps_since_improve = 0
        max_steps_no_improve = 20000 # Allow longer runs if improving
        
        while time.time() < deadline and steps_since_improve < max_steps_no_improve:
            if not unsat_clause_indices:
                return assignment
                
            # Pick random unsat clause
            c_idx = rng.choice(unsat_clause_indices)
            clause = clauses[c_idx]
            
            # Heuristic: Pick variable with highest unsat count in this clause
            # With 10% probability, pick a random variable in the clause (kick)
            if rng.random() < 0.1:
                l = rng.choice(clause)
                best_var = abs(l) - 1
            else:
                best_var = -1
                best_count = -1
                for l in clause:
                    v_idx = abs(l) - 1
                    if var_unsat_count[v_idx] > best_count:
                        best_count = var_unsat_count[v_idx]
                        best_var = v_idx
                        
            if best_var == -1:
                break
                
            # Flip best_var
            old_val = assignment[best_var]
            assignment[best_var] = not old_val
            
            # Update clause states for all clauses containing best_var
            improved = False
            
            for ci in var_to_clauses[best_var]:
                c = clauses[ci]
                was_sat = clause_is_sat[ci]
                
                # Check new state
                is_sat = False
                for l in c:
                    if assignment[abs(l)-1] == (l > 0):
                        is_sat = True
                        break
                
                if was_sat and not is_sat:
                    # Clause became unsat
                    clause_is_sat[ci] = False
                    unsat_clause_indices.append(ci)
                    for l in c:
                        var_unsat_count[abs(l)-1] += 1
                elif not was_sat and is_sat:
                    # Clause became sat
                    improved = True
                    clause_is_sat[ci] = True
                    # Remove ci from unsat_clause_indices
                    try:
                        unsat_clause_indices.remove(ci)
                    except ValueError:
                        pass
                    for l in c:
                        var_unsat_count[abs(l)-1] -= 1
                # else: state unchanged, no count update needed
                
            if improved:
                steps_since_improve = 0
            else:
                steps_since_improve += 1
                
    # If time runs out, return the last assignment found
    return assignment
