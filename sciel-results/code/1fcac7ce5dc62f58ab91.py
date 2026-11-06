import numpy as np
import random
import time

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290.0
    
    # Convert to 0-based signed literals for internal use
    # lit_internal = abs(lit)-1 if lit>0 else -(abs(lit)-1)-1
    # Actually, let's keep a mapping:
    # var = abs(lit)-1
    # sign = 1 if lit>0 else 0 (where 1 means True, 0 means False)
    
    # Pre-compute variable to clause mapping
    var_clauses = [[] for _ in range(n)]
    for i, clause in enumerate(clauses):
        for lit in clause:
            var = abs(lit) - 1
            var_clauses[var].append(i)
    
    # Build numpy arrays for fast vectorized satisfaction checks
    # For each clause, we have 3 literals.
    # clause_vars: shape (m, 3), values are variable indices (0-based)
    # clause_signs: shape (m, 3), values are 1 if positive, -1 if negative
    # assignment: shape (n,), values are 1 (True) or 0 (False)
    
    clause_vars = np.zeros((m, 3), dtype=np.int32)
    clause_signs = np.zeros((m, 3), dtype=np.int8)
    
    for i, clause in enumerate(clauses):
        for j, lit in enumerate(clause):
            var = abs(lit) - 1
            clause_vars[i, j] = var
            clause_signs[i, j] = 1 if lit > 0 else -1
            
    # Initialize random assignment
    rng = random.Random(42)
    assignment = np.array([1 if rng.random() < 0.5 else 0 for _ in range(n)], dtype=np.int8)
    
    def get_unsat_count():
        # For each clause, check if any literal is satisfied.
        # A literal (var, sign) is satisfied if:
        #   sign == 1 and assignment[var] == 1
        #   sign == -1 and assignment[var] == 0
        # This is equivalent to: (assignment[var] == 1) == (sign == 1)
        # Or: assignment[var] * sign == 1
        # Let's use: sat = (assignment[clause_vars] * clause_signs == 1)
        # sat is (m, 3)
        # A clause is unsatisfied if sum(sat[i,:]) == 0
        sat = (assignment[clause_vars] * clause_signs == 1)
        unsat_mask = sat.sum(axis=1) == 0
        return unsat_mask
    
    unsat_mask = get_unsat_count()
    if not unsat_mask.any():
        return [bool(x) for x in assignment]
    
    best_assignment = assignment.copy()
    best_unsat_count = unsat_mask.sum()
    
    # WalkSAT with restarts
    # We will use a simple strategy:
    # 1. Pick a random unsatisfied clause.
    # 2. Pick a variable in that clause.
    #    - With probability p (walk_prob), pick randomly.
    #    - Otherwise, pick the variable that minimizes the number of satisfied clauses it would break.
    # 3. Flip the variable.
    # 4. Update unsat_mask.
    
    # To make step 4 fast, we only need to re-evaluate clauses containing the flipped variable.
    # However, maintaining the full unsat_mask vectorized is fast enough for n=150, m~450.
    # Let's estimate: m=450, n=150.
    # Vectorized check is O(m*3) ~ 1350 ops. This is very fast in numpy.
    # So we can just recompute unsat_mask every step.
    
    # Parameters
    walk_prob = 0.4
    restart_interval = 10000
    
    steps_since_restart = 0
    total_steps = 0
    
    while time.time() < deadline:
        if not unsat_mask.any():
            return [bool(x) for x in assignment]
        
        current_unsat_count = unsat_mask.sum()
        if current_unsat_count < best_unsat_count:
            best_unsat_count = current_unsat_count
            best_assignment = assignment.copy()
            
        # Restart logic
        if steps_since_restart >= restart_interval:
            # Random restart
            assignment = np.array([1 if rng.random() < 0.5 else 0 for _ in range(n)], dtype=np.int8)
            unsat_mask = get_unsat_count()
            steps_since_restart = 0
            continue
        
        # Pick a random unsatisfied clause
        unsat_indices = np.where(unsat_mask)[0]
        if len(unsat_indices) == 0:
            return [bool(x) for x in assignment]
        
        c_idx = unsat_indices[rng.randint(0, len(unsat_indices)-1)]
        
        # Get literals in this clause
        clause_vars_c = clause_vars[c_idx]
        clause_signs_c = clause_signs[c_idx]
        
        # Decide which variable to flip
        if rng.random() < walk_prob:
            # Random flip
            lit_idx = rng.randint(0, 2)
            var = clause_vars_c[lit_idx]
        else:
            # Min-conflicts heuristic
            # For each of the 3 variables, count how many satisfied clauses would become unsatisfied.
            # A clause c2 is affected if it contains var.
            # c2 becomes unsatisfied if:
            #   - It was satisfied (not in unsat_mask)
            #   - And after flipping var, it becomes unsatisfied.
            
            min_cost = float('inf')
            best_var = -1
            
            for k in range(3):
                var = clause_vars_c[k]
                cost = 0
                # Iterate over clauses containing var
                for c2_idx in var_clauses[var]:
                    if c2_idx == c_idx:
                        continue
                    # If c2 is currently satisfied
                    if not unsat_mask[c2_idx]:
                        # Check if flipping var breaks c2
                        # c2 is satisfied if sum(sat[c2,:]) > 0
                        # Let's check if var is the only satisfied literal in c2, or if flipping it makes it unsat.
                        # Actually, we can just simulate the flip for this specific clause.
                        # But that's O(3) per clause.
                        
                        # Alternative: 
                        # c2 is satisfied.
                        # Let sat_count = sat[c2,:].sum()
                        # If sat_count > 1, flipping one literal might not break it.
                        # If sat_count == 1, and the satisfied literal is var, then flipping breaks it.
                        
                        # Let's compute sat for c2.
                        # sat_c2 = (assignment[clause_vars[c2]] * clause_signs[c2] == 1)
                        # This is vectorized for one clause.
                        sat_c2 = (assignment[clause_vars[c2]] * clause_signs[c2] == 1)
                        if sat_c2.sum() == 1:
                            # Only one literal is satisfied.
                            # If that literal is var, then flipping var breaks the clause.
                            # Find which index in c2 corresponds to var.
                            # clause_vars[c2] is an array of 3 vars.
                            idxs = np.where(clause_vars[c2] == var)[0]
                            if len(idxs) > 0:
                                # Check if the satisfied literal is indeed var.
                                # sat_c2[idx] is True if satisfied.
                                # We know sat_c2.sum() == 1.
                                # If sat_c2[idxs[0]] is True, then it's var.
                                if sat_c2[idxs[0]]:
                                    cost += 1
                                    # Note: it's possible var appears multiple times in a clause? 
                                    # In 3-SAT, usually distinct variables. Assume distinct.
                                    
                if cost < min_cost:
                    min_cost = cost
                    best_var = var
                    
            var = best_var
            
        # Flip variable
        assignment[var] = 1 - assignment[var]
        
        # Update unsat_mask
        unsat_mask = get_unsat_count()
        
        steps_since_restart += 1
        total_steps += 1
        
    return [bool(x) for x in best_assignment]
