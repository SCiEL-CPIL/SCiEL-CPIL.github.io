# EVOLVE-BLOCK-START
import random
import numpy as np


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    num_clauses = len(clauses)
    if num_clauses == 0:
        return [True] * n

    rng = random.Random(42)

    # Preprocess clauses
    clause_vars = []
    clause_signs = []
    var_to_clauses = [[] for _ in range(n)]
    
    for i, clause in enumerate(clauses):
        vars_ = [abs(l) - 1 for l in clause]
        signs = [l > 0 for l in clause]
        clause_vars.append(vars_)
        clause_signs.append(signs)
        for v in vars_:
            var_to_clauses[v].append(i)

    # --- Phase 1: Gaussian Elimination Initialization ---
    # Treat clauses as XOR equations: x_v1 XOR x_v2 XOR x_v3 = 0
    # We build a basis from a random subset of clauses.
    
    indices = list(range(num_clauses))
    rng.shuffle(indices)
    subset_indices = indices[:min(n, num_clauses)]
    
    basis = {}
    assignment = [False] * n
    assigned = [False] * n
    
    for idx in subset_indices:
        vars_in_clause = set(clause_vars[idx])
        
        # Reduce using existing basis
        changed = True
        while changed:
            changed = False
            for v in list(vars_in_clause):
                if v in basis:
                    vars_in_clause = vars_in_clause.symmetric_difference(basis[v])
                    changed = True
                    break
        
        if not vars_in_clause:
            continue
            
        pivot = rng.choice(list(vars_in_clause))
        basis[pivot] = vars_in_clause
        
        # Heuristic: Assign pivot = 0 (assuming others are 0)
        assignment[pivot] = False
        assigned[pivot] = True

    # Assign random values to unassigned variables
    for i in range(n):
        if not assigned[i]:
            assignment[i] = rng.random() < 0.5

    assignment = np.array(assignment, dtype=bool)

    # --- Phase 2: Targeted Repair (Weighted WalkSAT) ---
    
    # Precompute matrix for fast satisfaction check
    max_clause_len = 3
    clause_matrix = np.zeros((num_clauses, max_clause_len), dtype=int)
    clause_sign_matrix = np.zeros((num_clauses, max_clause_len), dtype=bool)
    clause_lengths = np.zeros(num_clauses, dtype=int)
    
    for i, (vars_, signs) in enumerate(zip(clause_vars, clause_signs)):
        length = len(vars_)
        clause_lengths[i] = length
        clause_matrix[i, :length] = vars_
        clause_sign_matrix[i, :length] = signs
        
    valid_mask = np.arange(max_clause_len)[None, :] < clause_lengths[:, None]
    
    # Function to check satisfaction and get unsat indices
    def get_unsatisfied_indices(assignment):
        var_indices = clause_matrix.copy()
        var_indices[~valid_mask] = 0
        var_values = assignment[var_indices]
        lit_satisfied = var_values == clause_sign_matrix
        lit_satisfied[~valid_mask] = True
        clause_satisfied = lit_satisfied.any(axis=1)
        return np.where(~clause_satisfied)[0]

    # Initialize conflict counts
    # conflict_count[v] = number of unsatisfied clauses containing v
    unsat_indices = get_unsatisfied_indices(assignment)
    conflict_counts = np.zeros(n, dtype=int)
    for idx in unsat_indices:
        for v in clause_vars[idx]:
            conflict_counts[v] += 1
    
    unsat_count = len(unsat_indices)
    
    max_iterations = 50000
    stuck_counter = 0
    best_unsat_count = unsat_count
    best_assignment = assignment.copy()
    
    for iteration in range(max_iterations):
        if unsat_count == 0:
            return assignment.tolist()
            
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment.copy()
            stuck_counter = 0
        else:
            stuck_counter += 1
            
        if stuck_counter > 2000:
            assignment = best_assignment.copy()
            unsat_indices = get_unsatisfied_indices(assignment)
            unsat_count = len(unsat_indices)
            if unsat_count == 0:
                return assignment.tolist()
            # Recalculate conflict counts
            conflict_counts = np.zeros(n, dtype=int)
            for idx in unsat_indices:
                for v in clause_vars[idx]:
                    conflict_counts[v] += 1
            stuck_counter = 0
            continue

        if unsat_count == 0:
            return assignment.tolist()

        # Pick a random unsatisfied clause
        clause_idx = rng.choice(unsat_indices.tolist())
        vars_in_clause = clause_vars[clause_idx]
        
        # Heuristic: Flip variable with highest conflict count in this clause
        # With 50% probability, flip random variable to escape local minima
        
        if rng.random() < 0.5:
            var_idx = rng.choice(vars_in_clause)
        else:
            # Find variable with max conflict count
            max_conf = -1
            best_var = vars_in_clause[0]
            for v in vars_in_clause:
                if conflict_counts[v] > max_conf:
                    max_conf = conflict_counts[v]
                    best_var = v
            var_idx = best_var

        # Flip variable
        old_val = assignment[var_idx]
        assignment[var_idx] = not old_val
        
        # Update conflict counts
        # For each clause containing var_idx:
        #   If clause was unsatisfied and becomes satisfied: conflict_counts for all vars in clause decrement by 1
        #   If clause was satisfied and becomes unsatisfied: conflict_counts for all vars in clause increment by 1
        
        # We need to know the state of clauses containing var_idx.
        # We can check them individually.
        
        for c_idx in var_to_clauses[var_idx]:
            # Check if clause c_idx is satisfied
            # Get vars and signs
            c_vars = clause_vars[c_idx]
            c_signs = clause_signs[c_idx]
            
            # Check satisfaction
            sat = False
            for v, s in zip(c_vars, c_signs):
                if assignment[v] == s:
                    sat = True
                    break
            
            # Determine if it was satisfied before the flip
            # We can infer this from conflict_counts? No.
            # We need to store clause satisfaction status?
            # Or just re-evaluate?
            
            # Let's re-evaluate satisfaction for this clause before and after.
            # But we already flipped.
            # Let's calculate delta.
            
            # Actually, it's easier to just recalculate conflict counts for the affected clauses.
            # But that's O(degree * clause_len).
            
            # Simpler: Just update conflict_counts for the specific clause.
            # If clause is NOW satisfied:
            #   It was NOT satisfied before (because we only flip vars in unsat clauses? No, we flip vars in ANY clause? No, we picked an unsat clause, but the flip might affect other clauses).
            #   Wait, the variable var_idx is in clause c_idx.
            #   If c_idx is satisfied NOW, it might have been satisfied BEFORE or NOT.
            #   If it was satisfied BEFORE and is satisfied NOW, no change.
            #   If it was satisfied BEFORE and is NOT satisfied NOW, conflict_counts increase.
            #   If it was NOT satisfied BEFORE and is satisfied NOW, conflict_counts decrease.
            #   If it was NOT satisfied BEFORE and is NOT satisfied NOW, no change.
            
            # We need to know the "before" state.
            # We can check if any other literal in the clause is satisfied.
            # If ANY other literal is satisfied, the clause is satisfied regardless of var_idx.
            # So, if any other literal is satisfied, the clause was satisfied BEFORE and is satisfied NOW.
            # If NO other literal is satisfied, then:
            #   If var_idx is satisfied NOW, clause is satisfied NOW.
            #       Was it satisfied BEFORE? Only if var_idx was satisfied BEFORE.
            #       Since we flipped, if it is satisfied NOW, it was NOT satisfied BEFORE.
            #       So it went Unsat -> Sat. Decrement counts.
            #   If var_idx is NOT satisfied NOW, clause is NOT satisfied NOW.
            #       Was it satisfied BEFORE? Only if var_idx was satisfied BEFORE.
            #       Since we flipped, if it is NOT satisfied NOW, it WAS satisfied BEFORE.
            #       So it went Sat -> Unsat. Increment counts.
            
            other_sat = False
            for v, s in zip(c_vars, c_signs):
                if v != var_idx and assignment[v] == s:
                    other_sat = True
                    break
                    
            if not other_sat:
                # Clause satisfaction depends entirely on var_idx
                if assignment[var_idx] == c_signs[c_vars.index(var_idx)]:
                    # Now Satisfied
                    # Was Unsat
                    for v in c_vars:
                        conflict_counts[v] -= 1
                else:
                    # Now Unsatisfied
                    # Was Sat
                    for v in c_vars:
                        conflict_counts[v] += 1

        # Update unsat_indices and unsat_count
        # This is expensive. We can just recalculate every K steps or if we think it's worth it.
        # For now, let's recalculate every 10 steps to keep it simple and robust.
        if iteration % 10 == 0:
            unsat_indices = get_unsatisfied_indices(assignment)
            unsat_count = len(unsat_indices)
            
    return best_assignment.tolist()

# EVOLVE-BLOCK-END
