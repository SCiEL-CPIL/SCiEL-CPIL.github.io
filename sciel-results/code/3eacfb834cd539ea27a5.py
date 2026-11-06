# EVOLVE-BLOCK-START
import random
import numpy as np


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Convert clauses to 0-indexed signed integers
    # Each clause is a list of signed 1-based literals
    # We'll work with numpy arrays for vectorized operations
    
    num_clauses = len(clauses)
    if num_clauses == 0:
        return [True] * n
    
    # Build clause matrix: each row is a clause, entries are signed literals (1-indexed)
    # For Gaussian elimination approach on linear relaxation:
    # We treat each clause as a constraint. In 3-SAT, a clause (l1 OR l2 OR l3) is unsatisfied
    # only when all literals are false.
    
    # Strategy:
    # 1. Initialize assignment using a parity-based heuristic (Gaussian elimination on parity constraints)
    # 2. Repair phase: iteratively flip variables in unsatisfied clauses
    
    rng = random.Random(42)
    
    # Step 1: Parity-based initialization
    # Build a system of parity equations from the clauses
    # For each clause (l1, l2, l3), we can form a parity equation:
    # x_{|l1|} XOR x_{|l2|} XOR x_{|l3|} = b
    # where b depends on the signs. This is a heuristic, not exact, but gives a good starting point.
    
    # Actually, let's use a simpler approach: 
    # Initialize with random assignment, then use local search with vectorized operations
    
    # Better approach: Use the structure of planted 3-SAT
    # Planted instances typically have a unique or few solutions
    # We can use a combination of:
    # 1. Initial random assignment
    # 2. Vectorized satisfaction checking
    # 3. Targeted repair
    
    # Initialize assignment
    assignment = np.array([rng.random() < 0.5 for _ in range(n)], dtype=bool)
    
    # Precompute literal indices and signs for fast lookup
    # For each clause, store (var_indices, signs)
    clause_vars = []
    clause_signs = []
    for clause in clauses:
        vars_ = [abs(l) - 1 for l in clause]
        signs = [l > 0 for l in clause]
        clause_vars.append(vars_)
        clause_signs.append(signs)
    
    # Convert to numpy arrays where possible
    # Since clauses may have different lengths (though 3-SAT means all are 3), we pad
    max_clause_len = max(len(c) for c in clauses)
    clause_matrix = np.zeros((num_clauses, max_clause_len), dtype=int)
    clause_sign_matrix = np.zeros((num_clauses, max_clause_len), dtype=bool)
    clause_lengths = np.zeros(num_clauses, dtype=int)
    
    for i, (vars_, signs) in enumerate(zip(clause_vars, clause_signs)):
        length = len(vars_)
        clause_lengths[i] = length
        clause_matrix[i, :length] = vars_
        clause_sign_matrix[i, :length] = signs
    
    # Function to check which clauses are unsatisfied (vectorized)
    def get_unsatisfied():
        # For each clause, check if all literals are false
        # A literal l is false if:
        #   if l > 0: assignment[var] == False
        #   if l < 0: assignment[var] == True
        # i.e., literal is false iff assignment[var] != (l > 0)
        
        # Get variable values for all clause positions
        # clause_matrix has -1 for padding, so we need to handle that
        valid_mask = np.arange(max_clause_len)[None, :] < clause_lengths[:, None]
        
        # Get current values of variables referenced in clauses
        var_indices = clause_matrix.copy()
        # Replace padding with 0 (will be masked out)
        var_indices[~valid_mask] = 0
        
        # Get assignment values for these variables
        var_values = assignment[var_indices]
        
        # A literal is satisfied if var_values == sign
        # For padded positions, we don't care
        lit_satisfied = var_values == clause_sign_matrix
        lit_satisfied[~valid_mask] = True  # padding positions don't matter
        
        # A clause is satisfied if any literal is satisfied
        clause_satisfied = lit_satisfied.any(axis=1)
        
        # Return indices of unsatisfied clauses
        return np.where(~clause_satisfied)[0]
    
    # Repair phase with targeted flips
    # We'll use a WalkSAT-like approach but with vectorized checks
    
    # First, try a simple iterative repair
    max_iterations = 50000
    stuck_counter = 0
    best_unsat_count = float('inf')
    best_assignment = assignment.copy()
    
    for iteration in range(max_iterations):
        unsat_indices = get_unsatisfied()
        unsat_count = len(unsat_indices)
        
        if unsat_count == 0:
            return assignment.tolist()
        
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment.copy()
        
        if unsat_count == 0:
            break
        
        # If we've been stuck for a while, reset to best found so far
        if iteration % 1000 == 0 and unsat_count >= best_unsat_count and best_unsat_count < unsat_count:
            assignment = best_assignment.copy()
        
        # Pick a random unsatisfied clause
        clause_idx = rng.choice(unsat_indices.tolist())
        clause = clause_vars[clause_idx]
        signs = clause_signs[clause_idx]
        
        # For each literal in the clause, decide whether to flip
        # Heuristic: flip a literal that is currently false
        # With some probability, flip a random literal (to escape local minima)
        
        if rng.random() < 0.5:
            # Flip a random variable in the clause
            var_idx = rng.choice(clause)
            assignment[var_idx] = not assignment[var_idx]
        else:
            # Flip the variable that appears most frequently in unsatisfied clauses
            # (simplified: just pick randomly among the clause's variables)
            var_idx = rng.choice(clause)
            assignment[var_idx] = not assignment[var_idx]
    
    # If we didn't find a solution, return the best assignment we found
    # But we need to return a satisfying assignment or the best effort
    # The task says "return a satisfying assignment", so if we can't find one,
    # we return the best we have (which may not be satisfying, scoring 0 for that instance)
    
    return assignment.tolist()
# EVOLVE-BLOCK-END
