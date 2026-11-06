# EVOLVE-BLOCK-START
import random
import numpy as np


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    num_clauses = len(clauses)
    if num_clauses == 0:
        return [True] * n
    
    rng = random.Random(42)
    
    # Precompute clause structure
    clause_vars = []
    clause_signs = []
    for clause in clauses:
        vars_ = [abs(l) - 1 for l in clause]
        signs = [l > 0 for l in clause]
        clause_vars.append(vars_)
        clause_signs.append(signs)
    
    # Build variable-to-clause index for fast lookup
    # Store (clause_index, sign) to allow fast update of satisfaction counts
    var_to_clauses = [[] for _ in range(n)]
    for i, (vars_, signs) in enumerate(zip(clause_vars, clause_signs)):
        for v, s in zip(vars_, signs):
            var_to_clauses[v].append((i, s))
            
    # --- Step 1: Spectral / Linear Initialization ---
    # Construct the incidence matrix M (C x N)
    # M[i, j] = 1 if x_j appears positively in clause i
    # M[i, j] = -1 if x_j appears negatively in clause i
    # M[i, j] = 0 otherwise
    M = np.zeros((num_clauses, n), dtype=float)
    for i, (vars_, signs) in enumerate(zip(clause_vars, clause_signs)):
        for v, s in zip(vars_, signs):
            if s:
                M[i, v] = 1.0
            else:
                M[i, v] = -1.0
                
    # Use SVD to find the "dominant" direction in the clause space
    # The top right singular vector often correlates with the planted solution
    try:
        U, s, Vt = np.linalg.svd(M, full_matrices=False)
        # Vt is N x N, top row is the first right singular vector
        v_top = Vt[0]
        # Threshold to boolean
        # If v_top[i] > 0, assign True, else False
        assignment = v_top > 0
    except:
        # Fallback to random or bias-based
        assignment = np.zeros(n, dtype=bool)
        for i in range(n):
            assignment[i] = rng.random() < 0.5

    # --- Step 2: Targeted Repair with Incremental Updates ---
    
    # Calculate initial satisfaction counts for each clause
    # clause_sat_count[i] = number of satisfied literals in clause i
    clause_sat_count = np.zeros(num_clauses, dtype=int)
    
    def compute_initial_counts(assignment):
        for i in range(num_clauses):
            count = 0
            for v, s in zip(clause_vars[i], clause_signs[i]):
                if assignment[v] == s:
                    count += 1
            clause_sat_count[i] = count
            
    compute_initial_counts(assignment)
    
    # Check if already satisfied
    if np.all(clause_sat_count > 0):
        return assignment.tolist()
    
    # Helper to update clause_sat_count when a variable is flipped
    def flip_variable(var_idx):
        assignment[var_idx] = not assignment[var_idx]
        new_val = assignment[var_idx]
        for clause_idx, sign in var_to_clauses[var_idx]:
            if new_val == sign:
                clause_sat_count[clause_idx] += 1
            else:
                clause_sat_count[clause_idx] -= 1

    max_iterations = 50000
    best_unsat = float('inf')
    best_assignment = assignment.copy()
    
    # Track unsatisfied clauses efficiently?
    # For N=150, scanning all clauses to find unsat is O(C) ~ 600.
    # 50000 * 600 = 30M ops. This is fine in Python/NumPy mix?
    # Actually, pure Python loop for scanning might be slow.
    # Let's use NumPy to find unsat indices.
    
    for iteration in range(max_iterations):
        # Find unsatisfied clauses
        unsat_indices = np.where(clause_sat_count == 0)[0]
        unsat_count = len(unsat_indices)
        
        if unsat_count == 0:
            return assignment.tolist()
        
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment.copy()
            
        # If stuck for too long, restart or backtrack?
        # Simple restart to best if no improvement for a while?
        # Let's just keep going, WalkSAT usually escapes.
        
        # Pick a random unsatisfied clause
        clause_idx = rng.choice(unsat_indices.tolist())
        
        # Select a variable to flip from this clause
        # Heuristic: Pick variable that appears in the most other unsatisfied clauses
        clause_vars_list = clause_vars[clause_idx]
        clause_signs_list = clause_signs[clause_idx]
        
        best_var = -1
        best_score = -1
        
        # Create a set of unsat clause indices for fast lookup
        # Since unsat_count is likely small, a list or set is fine
        # But checking membership in a list is O(K).
        # Let's use a boolean mask or set.
        # For speed, since N is small, we can just iterate.
        
        # Optimization: Pre-compute "impact" scores for variables in the clause
        # Score = number of other unsat clauses containing this variable
        # We can approximate this by checking var_to_clauses[v] against unsat_indices
        
        # To avoid O(C) check for each of 3 vars, we can just do it directly.
        # 3 vars * O(degree) checks. Degree ~ 10-20. So ~60 ops. Very fast.
        
        for v in clause_vars_list:
            score = 0
            for ci, s in var_to_clauses[v]:
                if ci == clause_idx:
                    continue
                if clause_sat_count[ci] == 0:
                    score += 1
            
            # Add small random noise to break ties
            score += rng.random() * 0.1
            
            if score > best_score:
                best_score = score
                best_var = v
                
        # If no variable was selected (shouldn't happen), pick first
        if best_var == -1:
            best_var = clause_vars_list[0]
            
        flip_variable(best_var)
        
    return best_assignment.tolist()
# EVOLVE-BLOCK-END
