import random
import time
import numpy as np

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Pre-process clauses into a numpy array for fast vectorized operations
    # literals are 1-based signed integers
    # We need to map literal l to index abs(l)-1 and sign (l > 0)
    # For vectorized satisfaction check:
    # A literal l is satisfied if:
    #   if l > 0: assignment[abs(l)-1] is True
    #   if l < 0: assignment[abs(l)-1] is False
    # Let A be the assignment array of booleans (True/False).
    # Let L be the array of literals.
    # Let idx = abs(L) - 1
    # Let pos = L > 0
    # A literal is satisfied if (A[idx] == pos)
    # A clause is satisfied if any literal in it is satisfied.
    
    # Convert clauses to a list of arrays
    clause_list = [np.array(c, dtype=np.int32) for c in clauses]
    max_clause_len = max(len(c) for c in clause_list)
    
    # Pad clauses to same length for easy vectorization, or just loop over them?
    # With 50 variables, clauses are small (3 literals). 
    # Looping over clauses in Python is slow. 
    # Let's try to vectorize across clauses.
    
    # Create a 2D array of literals, padded with 0 (which won't match any variable index if we handle carefully)
    # Actually, 0 is not a valid literal in 1-based indexing. 
    # If we pad with 0, abs(0)-1 = -1. We must ensure we don't access index -1 incorrectly or that it doesn't falsely satisfy.
    # Better: Keep clauses separate but use list comprehensions with numpy inside? 
    # Or, since n=50 is small, the overhead of Python loops over ~100-1000 clauses might be acceptable if the inner check is fast?
    # Let's estimate: 1000 clauses * 3 literals = 3000 ops. 
    # In Python, 3000 ops is ~0.1ms. 20000 steps * 0.1ms = 2s. 
    # With NumPy, we can do all clauses at once.
    
    # Let's create a padded array of literals
    # If number of clauses is M, shape is (M, 3)
    M = len(clauses)
    lit_array = np.zeros((M, max_clause_len), dtype=np.int32)
    for i, c in enumerate(clauses):
        lit_array[i, :len(c)] = c
        
    idx_array = np.abs(lit_array) - 1
    pos_array = lit_array > 0
    
    # Mask for valid literals (non-zero)
    valid_mask = lit_array != 0
    
    def get_unsat_clauses(assignment):
        # assignment is a numpy array of bools
        # A literal at (i, j) is satisfied if assignment[idx_array[i, j]] == pos_array[i, j]
        # We need to be careful with index -1 from padding. 
        # If lit_array[i, j] is 0, idx is -1. 
        # assignment[-1] is the last element. This is dangerous.
        # Let's handle padding by setting idx to 0 for invalid, and then masking out the result.
        
        safe_idx = np.where(valid_mask, idx_array, 0)
        # Get the values from assignment
        vals = assignment[safe_idx]
        # Check satisfaction
        sat_lits = (vals == pos_array) & valid_mask
        # A clause is satisfied if any literal in it is satisfied
        clause_sat = sat_lits.any(axis=1)
        return np.where(~clause_sat)[0]

    def check_satisfied(assignment):
        unsat = get_unsat_clauses(assignment)
        return len(unsat) == 0

    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    # WalkSAT parameters
    # Weight for flipping a literal that appears in the most unsat clauses
    # Standard WalkSAT: Pick an unsat clause, pick a literal. 
    # With probability p, flip the literal that appears in the most unsat clauses.
    # With probability 1-p, pick a random literal from the clause.
    
    # Pre-compute literal frequencies? 
    # To find the "best" literal in a clause, we need to know how many unsat clauses contain each literal.
    # This requires counting.
    
    # Let's implement a simpler but effective variant first: 
    # Random Walk with a slight bias towards satisfying more clauses.
    # Actually, standard WalkSAT is quite strong. Let's optimize it.
    
    # To count occurrences of literals in unsat clauses efficiently:
    # We have unsat_clause_indices.
    # We can extract the literals from those clauses.
    
    def walksat_step(assignment, unsat_idx, rng):
        if len(unsat_idx) == 0:
            return assignment, True
        
        # Pick a random unsat clause
        clause_idx = unsat_idx[rng.randint(0, len(unsat_idx) - 1)]
        clause_lits = lit_array[clause_idx]
        valid_lits = clause_lits[clause_lits != 0]
        
        if len(valid_lits) == 0:
            # Should not happen for valid 3-SAT
            return assignment, False
            
        # Count occurrences of each literal in all unsat clauses
        # This is expensive if done naively. 
        # Alternative: Just pick a random literal? (Random Walk)
        # Or: Pick the literal that flips to satisfy the most other unsat clauses?
        
        # Let's use a heuristic: 
        # For each literal in the chosen clause, calculate the "gain" if flipped.
        # Gain = number of unsat clauses that would become satisfied.
        
        # This requires checking all unsat clauses for each candidate literal.
        # With ~1000 unsat clauses, and 3 candidates, this is 3000 checks.
        # Each check is vectorized?
        
        # Let's try a simpler approach first: Random Walk with Restarts.
        # If Random Walk isn't enough, we can add the "weight" heuristic.
        
        # Random literal selection
        lit = valid_lits[rng.randint(0, len(valid_lits) - 1)]
        var_idx = abs(lit) - 1
        assignment[var_idx] = not assignment[var_idx]
        
        return assignment, False

    # Better: Implement the "Weight" heuristic properly.
    # We can pre-compute for each variable, which clauses it appears in?
    # That's static. But we need dynamic counts of *unsat* clauses.
    
    # Let's stick to a robust Random Walk with Restarts first, as it's simpler and often sufficient for planted 3-SAT with n=50.
    # If that fails, we can iterate. But we need to submit one version.
    # Let's add a simple "improvement" step: 
    # If a flip reduces the number of unsat clauses, keep it. 
    # This is basically MinConflicts or a greedy hill-climb.
    
    def solve_with_heuristic(assignment, rng, max_steps):
        # Calculate initial unsat count
        unsat_idx = get_unsat_clauses(assignment)
        current_unsat_count = len(unsat_idx)
        
        for step in range(max_steps):
            if current_unsat_count == 0:
                return assignment, True
            
            if time.time() > deadline:
                return assignment, False
                
            # Pick a random unsat clause
            clause_idx = unsat_idx[rng.randint(0, len(unsat_idx) - 1)]
            clause_lits = lit_array[clause_idx]
            valid_lits = clause_lits[clause_lits != 0]
            
            if len(valid_lits) == 0:
                continue
                
            # Try all literals in the clause, pick the one that minimizes unsat clauses
            best_lit = None
            best_unsat_count = current_unsat_count + 1 # worse than current
            
            for lit in valid_lits:
                var_idx = abs(lit) - 1
                # Flip
                old_val = assignment[var_idx]
                assignment[var_idx] = not old_val
                
                # Check new unsat count
                new_unsat_idx = get_unsat_clauses(assignment)
                new_count = len(new_unsat_idx)
                
                if new_count < best_unsat_count:
                    best_unsat_count = new_count
                    best_lit = lit
                    best_var_idx = var_idx
                    best_unsat_idx = new_unsat_idx
                    best_old_val = old_val
                    best_new_val = not old_val
                
                # Unflip for next candidate
                assignment[var_idx] = old_val
                
            # Apply the best flip
            if best_lit is not None:
                assignment[best_var_idx] = best_new_val
                current_unsat_count = best_unsat_count
                unsat_idx = best_unsat_idx
            else:
                # No improvement found, pick random
                lit = valid_lits[rng.randint(0, len(valid_lits) - 1)]
                var_idx = abs(lit) - 1
                assignment[var_idx] = not assignment[var_idx]
                unsat_idx = get_unsat_clauses(assignment)
                current_unsat_count = len(unsat_idx)
                
        return assignment, current_unsat_count == 0

    # Strategy: Multiple restarts
    # Each restart gets a random initial assignment
    # Run the heuristic solver for a fixed number of steps or until time limit
    # If solved, return.
    
    rng = random.Random()
    
    # Determine steps per restart based on time
    # Let's try 1000 steps per restart initially
    steps_per_restart = 2000
    
    while time.time() < deadline:
        # Random initialization
        assignment = np.random.rand(n) < 0.5
        
        solved, _ = solve_with_heuristic(assignment, rng, steps_per_restart)
        
        if solved:
            # Convert numpy bool to list of bools
            return assignment.tolist()
            
    # If time runs out, return the last assignment (likely unsat)
    return assignment.tolist()

