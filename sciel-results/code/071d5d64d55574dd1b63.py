import random
import time

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Pre-processing for efficiency
    # 1. Map variables to the indices of clauses they appear in
    var_to_clauses = [[] for _ in range(n)]
    # Also precompute: for each (var, clause_idx), store the literal sign
    var_clause_lit = {}  # (var_idx, clause_idx) -> literal (signed)
    
    for i, c in enumerate(clauses):
        for l in c:
            var_idx = abs(l) - 1
            var_to_clauses[var_idx].append(i)
            var_clause_lit[(var_idx, i)] = l
            
    # Precompute literal satisfaction lookup
    # For each literal, precompute which assignment value satisfies it
    # lit > 0: satisfied when assign[var] = True
    # lit < 0: satisfied when assign[var] = False
    
    def is_lit_sat(lit, assign):
        var = abs(lit) - 1
        val = assign[var]
        return (lit > 0 and val) or (lit < 0 and not val)
    
    def get_clause_sat_status(clause, assign):
        for l in clause:
            var = abs(l) - 1
            val = assign[var]
            if (l > 0 and val) or (l < 0 and not val):
                return True
        return False
    
    # Initial random assignment
    rng = random.Random(42)
    
    # Try multiple restarts to ensure robustness
    max_restarts = 50
    start_time = time.time()
    deadline = start_time + 290.0
    
    best_assign = None
    best_unsat = float('inf')
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Initialize assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Initialize clause status
        clause_sat = [False] * len(clauses)
        unsat_count = 0
        for i, c in enumerate(clauses):
            if get_clause_sat_status(c, assignment):
                clause_sat[i] = True
            else:
                unsat_count += 1
                
        if unsat_count == 0:
            return assignment
            
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assign = assignment[:]
            
        # WalkSAT Loop
        walk_prob = 0.3
        max_steps = 5000 + restart * 1000
        step = 0
        no_improve_steps = 0
        last_unsat = unsat_count
        
        while step < max_steps:
            if time.time() > deadline:
                break
                
            if unsat_count == 0:
                return assignment
                
            # Select an unsatisfied clause
            unsat_indices = [i for i in range(len(clauses)) if not clause_sat[i]]
            if not unsat_indices:
                return assignment
                
            c_idx = rng.choice(unsat_indices)
            c = clauses[c_idx]
            
            # Select variable to flip
            if rng.random() < walk_prob:
                # Walk step
                lit = rng.choice(c)
                var_idx = abs(lit) - 1
            else:
                # Greedy step
                best_gain = float('inf')
                best_var = None
                
                for lit in c:
                    var_idx = abs(lit) - 1
                    current_val = assignment[var_idx]
                    new_val = not current_val
                    
                    gain = 0
                    for ci in var_to_clauses[var_idx]:
                        cc = clauses[ci]
                        is_sat_now = clause_sat[ci]
                        lit_in_cc = var_clause_lit[(var_idx, ci)]
                        
                        new_lit_sat = (lit_in_cc > 0 and new_val) or (lit_in_cc < 0 and not new_val)
                        
                        if is_sat_now:
                            old_lit_sat = (lit_in_cc > 0 and current_val) or (lit_in_cc < 0 and not current_val)
                            if old_lit_sat and not new_lit_sat:
                                # Check if any other lit in cc is satisfied
                                others_sat = False
                                for l in cc:
                                    if abs(l) - 1 != var_idx:
                                        if is_lit_sat(l, assignment):
                                            others_sat = True
                                            break
                                if not others_sat:
                                    gain += 1
                        else:
                            if new_lit_sat:
                                gain -= 1
                                
                    if gain < best_gain:
                        best_gain = gain
                        best_var = var_idx
                        
                if best_var is not None:
                    var_idx = best_var
                else:
                    lit = rng.choice(c)
                    var_idx = abs(lit) - 1
            
            # Perform the flip
            assignment[var_idx] = not assignment[var_idx]
            
            # Update clause statuses efficiently
            for ci in var_to_clauses[var_idx]:
                cc = clauses[ci]
                old_sat = clause_sat[ci]
                new_sat = get_clause_sat_status(cc, assignment)
                
                if old_sat and not new_sat:
                    unsat_count += 1
                elif not old_sat and new_sat:
                    unsat_count -= 1
                    
                clause_sat[ci] = new_sat
                
            step += 1
            
            # Track progress
            if unsat_count < last_unsat:
                last_unsat = unsat_count
                no_improve_steps = 0
            else:
                no_improve_steps += 1
                
            if no_improve_steps > 2000:
                break
                
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assign = assignment[:]
                
        # End of inner loop for this restart
        
    return best_assign
