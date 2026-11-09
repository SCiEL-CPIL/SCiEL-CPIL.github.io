"""Synthetic planted 3-SAT solver (150 variables) - Optimized WalkSAT with NumPy vectorization."""
import time
import random
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    # Deadline guard: return best result before 330s limit with safety margin
    start_time = time.time()
    deadline = start_time + 310.0  # 20s margin
    
    # Preprocess: convert clauses to 0-based var indices and signs
    # Store as arrays for NumPy operations
    clause_vars = np.zeros((m, 3), dtype=np.int32)  # variable indices
    clause_signs = np.zeros((m, 3), dtype=np.int8)  # +1 or -1
    
    for ci, clause in enumerate(clauses):
        for li, lit in enumerate(clause):
            clause_vars[ci, li] = abs(lit) - 1
            clause_signs[ci, li] = 1 if lit > 0 else -1
    
    # Build variable to clause mapping
    var_to_clauses = [[] for _ in range(n)]
    for ci in range(m):
        for li in range(3):
            var_to_clauses[clause_vars[ci, li]].append(ci)
    
    # Precompute for each variable, the list of (clause_idx, literal_pos) pairs
    var_clause_pos = []
    for v in range(n):
        pairs = []
        for ci in var_to_clauses[v]:
            for li in range(3):
                if clause_vars[ci, li] == v:
                    pairs.append((ci, li))
        var_clause_pos.append(pairs)
    
    def check_assignment(assignment):
        """Check if assignment satisfies all clauses using NumPy."""
        # assignment is a list of bools, convert to numpy
        # A literal (var, sign) is satisfied if:
        #   sign == 1 and assignment[var] == True, OR
        #   sign == -1 and assignment[var] == False
        # In numpy: for positive sign, assignment[var]; for negative, not assignment[var]
        
        # Create a matrix: sat[ci, li] = whether literal (ci, li) is satisfied
        # assignment[clause_vars[ci,li]] gives the value
        # For positive sign: sat = assignment[var]
        # For negative sign: sat = not assignment[var]
        
        # Vectorized approach:
        # sat_matrix = np.where(clause_signs == 1, assignment_arr[clause_vars], 1 - assignment_arr[clause_vars])
        
        assignment_arr = np.array(assignment, dtype=np.int8)
        # For each literal, check if it's satisfied
        # Positive literal: satisfied if var is True (1)
        # Negative literal: satisfied if var is False (0)
        
        # Create satisfaction matrix
        sat_matrix = np.zeros((m, 3), dtype=np.int8)
        for li in range(3):
            vars_col = clause_vars[:, li]
            signs_col = clause_signs[:, li]
            vals = assignment_arr[vars_col]
            # If sign is +1, satisfied when val==1; if sign is -1, satisfied when val==0
            sat_matrix[:, li] = np.where(signs_col == 1, vals, 1 - vals)
        
        # A clause is satisfied if any literal is satisfied
        clause_sat = np.any(sat_matrix, axis=1)
        
        if np.all(clause_sat):
            return True, -1
        else:
            unsat_idx = np.where(~clause_sat)[0]
            return False, unsat_idx[0]
    
    def count_unsatisfied(assignment):
        """Count unsatisfied clauses using NumPy."""
        assignment_arr = np.array(assignment, dtype=np.int8)
        sat_matrix = np.zeros((m, 3), dtype=np.int8)
        for li in range(3):
            vars_col = clause_vars[:, li]
            signs_col = clause_signs[:, li]
            vals = assignment_arr[vars_col]
            sat_matrix[:, li] = np.where(signs_col == 1, vals, 1 - vals)
        
        clause_sat = np.any(sat_matrix, axis=1)
        return int(np.sum(~clause_sat))
    
    def get_unsat_clauses(assignment):
        """Get list of unsatisfied clause indices."""
        assignment_arr = np.array(assignment, dtype=np.int8)
        sat_matrix = np.zeros((m, 3), dtype=np.int8)
        for li in range(3):
            vars_col = clause_vars[:, li]
            signs_col = clause_signs[:, li]
            vals = assignment_arr[vars_col]
            sat_matrix[:, li] = np.where(signs_col == 1, vals, 1 - vals)
        
        clause_sat = np.any(sat_matrix, axis=1)
        return list(np.where(~clause_sat)[0])
    
    def compute_flip_damage(assignment, var_idx, ci):
        """Compute damage if we flip var_idx, excluding clause ci."""
        damage = 0
        current_val = assignment[var_idx]
        
        for oj, oj_li in var_clause_pos[var_idx]:
            if oj == ci:
                continue
            
            # Check if this literal in clause oj is currently satisfied
            oj_sign = clause_signs[oj, oj_li]
            currently_sat = (current_val == (oj_sign == 1))
            
            if currently_sat:
                # Check if any other literal in clause oj is satisfied
                other_sat = False
                for li in range(3):
                    if li == oj_li:
                        continue
                    v2 = clause_vars[oj, li]
                    s2 = clause_signs[oj, li]
                    if assignment[v2] == (s2 == 1):
                        other_sat = True
                        break
                if not other_sat:
                    damage += 1
        
        return damage
    
    def walksat_step(assignment, rng, weight=0.55):
        """One step of WalkSAT: pick random unsat clause, flip a literal in it."""
        unsat_clauses = get_unsat_clauses(assignment)
        
        if not unsat_clauses:
            return assignment, True
        
        # Pick a random unsatisfied clause
        ci = rng.choice(unsat_clauses)
        
        # For each literal in the clause, compute the impact of flipping
        best_var = -1
        best_score = float('inf')
        
        for li in range(3):
            var_idx = clause_vars[ci, li]
            damage = compute_flip_damage(assignment, var_idx, ci)
            
            if damage < best_score:
                best_score = damage
                best_var = var_idx
        
        # With probability weight, flip the best variable; otherwise flip random
        if best_var >= 0 and rng.random() < weight:
            assignment[best_var] = not assignment[best_var]
        else:
            li = rng.choice([0, 1, 2])
            var_idx = clause_vars[ci, li]
            assignment[var_idx] = not assignment[var_idx]
        
        return assignment, False
    
    def polish(assignment, max_iterations=1000):
        """Local search polish: try flipping each variable if it reduces unsat count."""
        current_unsat = count_unsatisfied(assignment)
        
        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
            
            improved = False
            for v in range(n):
                # Try flipping variable v
                old_val = assignment[v]
                assignment[v] = not old_val
                
                new_unsat = count_unsatisfied(assignment)
                
                if new_unsat < current_unsat:
                    current_unsat = new_unsat
                    improved = True
                else:
                    assignment[v] = old_val
            
            if not improved:
                break
        
        return assignment, current_unsat
    
    # Try multiple random restarts
    best_assignment = None
    best_unsat_count = float('inf')
    
    max_restarts = 50
    max_steps_per_restart = 5000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Random initial assignment
        assignment = [random.random() < 0.5 for _ in range(n)]
        rng = random.Random(restart * 12345 + 67890)
        
        # Quick check
        sat, _ = check_assignment(assignment)
        if sat:
            return assignment
        
        unsat_count = count_unsatisfied(assignment)
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
        
        # WalkSAT steps
        for step in range(max_steps_per_restart):
            if time.time() > deadline:
                break
            
            assignment, solved = walksat_step(assignment, rng, weight=0.55)
            
            if solved:
                return assignment
            
            # Periodically check unsat count
            if step % 50 == 0:
                unsat_count = count_unsatisfied(assignment)
                if unsat_count < best_unsat_count:
                    best_unsat_count = unsat_count
                    best_assignment = assignment[:]
        
        # Final check for this restart
        sat, _ = check_assignment(assignment)
        if sat:
            return assignment
        
        # Apply polish pass
        polished, polished_unsat = polish(assignment)
        if polished_unsat < best_unsat_count:
            best_unsat_count = polished_unsat
            best_assignment = polished[:]
        
        if polished_unsat == 0:
            return polished
    
    # Return best assignment found
    if best_assignment is not None:
        # Try one more polish on the best assignment
        polished, polished_unsat = polish(best_assignment[:], max_iterations=2000)
        if polished_unsat == 0:
            return polished
        return polished
    else:
        # Fallback: random assignment
        return [random.random() < 0.5 for _ in range(n)]
