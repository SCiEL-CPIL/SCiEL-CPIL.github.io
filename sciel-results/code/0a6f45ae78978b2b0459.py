"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Deadline guard
    start_time = time.time()
    timeout = 320.0  # Leave margin for overhead
    
    m = len(clauses)
    if m == 0:
        return [True] * n
    
    # Preprocess clauses into numpy arrays for fast evaluation
    # clause_data: shape (m, 3), values are +/-1
    # clause_vars: shape (m, 3), values are var indices (0-based)
    clause_data = np.empty((m, 3), dtype=np.int8)
    clause_vars = np.empty((m, 3), dtype=np.int32)
    
    for i, c in enumerate(clauses):
        for j, lit in enumerate(c):
            clause_data[i, j] = 1 if lit > 0 else -1
            clause_vars[i, j] = abs(lit) - 1
    
    # Build variable-to-clause index mapping
    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = clause_vars[i, j]
            var_to_clauses[v].append(i)
    
    # Convert to numpy arrays for faster access
    var_to_clauses_arr = []
    for v in range(n):
        var_to_clauses_arr.append(np.array(var_to_clauses[v], dtype=np.int32))
    
    # Initialize random assignment
    rng = random.Random(42)
    assignment = np.array([rng.random() < 0.5 for _ in range(n)], dtype=bool)
    
    # Helper function to compute unsatisfied clause count and clause satisfaction status
    def compute_unsat(assignment):
        # For each clause, check if it's satisfied
        # assignment is boolean array, clause_data is +/-1
        # A literal is satisfied if assignment[var] == (data == 1)
        # i.e., assignment[var] * data > 0 (treating True as 1, False as -1)
        # Actually: literal with value +1 is satisfied if assignment[var] is True
        #           literal with value -1 is satisfied if assignment[var] is False
        # So: satisfied if (data == 1 and assignment[var]) or (data == -1 and not assignment[var])
        # Equivalent: (data * (assignment[var] * 2 - 1)) > 0
        # Let signed_assignment = assignment * 2 - 1 (True -> 1, False -> -1)
        # Then literal is satisfied if clause_data * signed_assignment[var] > 0
        
        signed_assign = assignment.astype(np.int8) * 2 - 1  # shape (n,)
        # For each clause, for each literal, compute clause_data[i,j] * signed_assign[clause_vars[i,j]]
        # Shape: (m, 3)
        lit_values = clause_data * signed_assign[clause_vars]
        # Clause is satisfied if any literal is satisfied (value > 0)
        clause_sat = (lit_values > 0).any(axis=1)  # shape (m,)
        unsat_count = m - clause_sat.sum()
        return unsat_count, clause_sat
    
    # Initial evaluation
    unsat_count, clause_sat = compute_unsat(assignment)
    if unsat_count == 0:
        return assignment.tolist()
    
    # Track best solution found
    best_unsat = unsat_count
    best_assignment = assignment.copy()
    
    # WalkSAT with restarts
    max_restarts = 50
    steps_per_restart = 5000
    
    for restart in range(max_restarts):
        # Check deadline
        if time.time() - start_time > timeout:
            break
        
        # For first restart, use current assignment; for subsequent, randomize
        if restart > 0:
            assignment = np.array([rng.random() < 0.5 for _ in range(n)], dtype=bool)
            unsat_count, clause_sat = compute_unsat(assignment)
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment.copy()
            if unsat_count == 0:
                return assignment.tolist()
        
        for step in range(steps_per_restart):
            # Check deadline periodically
            if step % 100 == 0 and time.time() - start_time > timeout:
                break
            
            if unsat_count == 0:
                return assignment.tolist()
            
            # Pick a random unsatisfied clause
            unsat_indices = np.where(~clause_sat)[0]
            if len(unsat_indices) == 0:
                return assignment.tolist()
            
            clause_idx = unsat_indices[rng.randint(0, len(unsat_indices) - 1)]
            
            # Choose a literal to flip using WalkSAT heuristic
            # With probability p, pick the literal that satisfies the most other clauses
            # With probability 1-p, pick randomly
            p = 0.5
            clause_lits = clause_vars[clause_idx]  # 3 variables
            clause_signs = clause_data[clause_idx]  # 3 signs
            
            if rng.random() < p:
                # Greedy: pick variable that maximizes number of satisfied clauses
                best_var = -1
                best_gain = -1
                for j in range(3):
                    v = clause_lits[j]
                    # If we flip v, which clauses become satisfied?
                    # Clauses containing v where the literal becomes satisfied
                    v_clauses = var_to_clauses_arr[v]
                    # For each such clause, check if flipping v would satisfy it
                    # Current literal value: clause_data[i, j] * signed_assign[v]
                    # After flip: sign changes, so new value = -old value
                    # Clause becomes satisfied if new value > 0, i.e., old value < 0
                    # But we need to check all literals in the clause
                    # Simpler: count how many currently unsatisfied clauses containing v
                    # would be satisfied by flipping v
                    # A clause is satisfied by flipping v if:
                    #   - The clause is currently unsatisfied
                    #   - The literal for v in this clause would become satisfied after flip
                    #   - No other literal in the clause is already satisfied (but clause is unsat, so none are)
                    # Actually if clause is unsat, no literal is satisfied. After flipping v,
                    # the clause is satisfied iff the v-literal becomes satisfied.
                    # v-literal becomes satisfied iff clause_data * new_signed_assign[v] > 0
                    # new_signed_assign[v] = -signed_assign[v]
                    # So: clause_data * (-signed_assign[v]) > 0 => clause_data * signed_assign[v] < 0
                    # i.e., the current v-literal is unsatisfied (which it is, since clause is unsat)
                    # Wait, that's always true if clause is unsat. So any unsat clause containing v
                    # gets satisfied by flipping v.
                    # So gain = number of unsat clauses containing v
                    gain = 0
                    for ci in v_clauses:
                        if not clause_sat[ci]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_var = v
                flip_var = best_var
            else:
                # Random pick from the 3 literals
                j = rng.randint(0, 2)
                flip_var = clause_lits[j]
            
            # Flip the variable
            old_val = assignment[flip_var]
            assignment[flip_var] = not old_val
            
            # Update clause satisfaction for clauses containing flip_var
            signed_assign = assignment.astype(np.int8) * 2 - 1
            v_clauses = var_to_clauses_arr[flip_var]
            
            # Recompute satisfaction for affected clauses
            for ci in v_clauses:
                # Check if clause ci is now satisfied
                lit_vals = clause_data[ci] * signed_assign[clause_vars[ci]]
                new_sat = (lit_vals > 0).any()
                old_sat = clause_sat[ci]
                if new_sat and not old_sat:
                    unsat_count -= 1
                elif not new_sat and old_sat:
                    unsat_count += 1
                clause_sat[ci] = new_sat
            
            # Track best
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment.copy()
            
            if unsat_count == 0:
                return assignment.tolist()
        
        # End of restart
    
    # Return best found
    if best_unsat == 0:
        return best_assignment.tolist()
    return best_assignment.tolist()
# EVOLVE-BLOCK-END
