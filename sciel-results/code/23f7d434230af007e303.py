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
    
    # Build variable-to-clauses mapping
    var_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = clause_vars[i, j]
            var_clauses[v].append(i)
    
    def evaluate():
        """Compute which clauses are satisfied and the truth values."""
        # assignment_vals: +1 for True, -1 for False
        # satisfied_clause: for each clause, is any literal true?
        # clause_values[i, j] = clause_data[i, j] * assignment_vals[clause_vars[i, j]]
        # satisfied if any value in the row is +1
        
        return None  # We'll use a different approach for speed
    
    def is_satisfied(assignment):
        """Return boolean array of satisfied clauses."""
        # assignment is a numpy array of bools
        # We need to check for each clause if any literal is true
        # literal is true if (assignment[var] == (sign == 1))
        # i.e., assignment[var] == True if sign is +1, assignment[var] == False if sign is -1
        
        # clause_vars: (m, 3) indices
        # clause_data: (m, 3) signs (+1/-1)
        # assignment: (n,) bools
        
        # Get the values of the variables in each clause position
        vals = assignment[clause_vars]  # (m, 3) bools
        # For each literal, it's true if (vals == (clause_data == 1))
        # clause_data == 1 gives a bool array of shape (m, 3)
        sign_bool = (clause_data == 1)  # (m, 3)
        # True if vals matches sign_bool
        literal_true = (vals == sign_bool)  # (m, 3)
        # Clause is satisfied if any literal is true
        return np.any(literal_true, axis=1)  # (m,) bools
    
    def find_best_flip(assignment, unsat_indices):
        """
        For each variable, compute the number of satisfied clauses that would be unsatisfied
        and the number of unsatisfied clauses that would be satisfied if we flip the variable.
        We want to maximize (unsat_becomes_sat - sat_becomes_unsat).
        """
        if len(unsat_indices) == 0:
            return None, 0
        
        # Count for each variable:
        # +1 for each unsatisfied clause where this variable appears with a sign such that
        #    flipping would make this literal true (i.e., currently the literal is false)
        # -1 for each satisfied clause where this variable appears with a sign such that
        #    flipping would make this literal false (i.e., currently the literal is true)
        #    AND the clause has no other true literal (i.e., this is the only true literal)
        
        # For simplicity and speed, let's use a random restart strategy with a good heuristic:
        # 1. Pick a random unsatisfied clause
        # 2. Pick a variable from that clause
        # 3. Flip it
        
        # But we can do better: pick the variable that appears in the most unsatisfied clauses
        # or use a score-based approach.
        
        # Let's compute a score for each variable
        scores = np.zeros(n, dtype=np.int32)
        
        # For unsatisfied clauses, for each literal, if the variable is currently set such that
        # the literal is false, flipping would make it true -> +1
        # If the variable is currently set such that the literal is true, flipping would make it false -> -1
        # But wait, in an unsatisfied clause, ALL literals are false. So flipping any variable in the clause
        # will make that literal true, thus satisfying the clause. So each variable in an unsat clause gets +1.
        
        unsat_vars = clause_vars[unsat_indices]  # (k, 3)
        # Count occurrences
        for v in unsat_vars.flatten():
            scores[v] += 1
        
        # For satisfied clauses, we need to be careful: only if the clause has exactly one true literal,
        # flipping that variable would unsatisfy the clause.
        # This is more complex. Let's first try just the unsat-based score.
        
        # Add a small random tie-breaker
        rng_seed = random.Random()
        for _ in range(10):
            v = rng_seed.randint(0, n - 1)
            scores[v] += 1
        
        best_var = np.argmax(scores)
        return best_var, scores[best_var]
    
    def walk_sat(restart_assignment, max_flips, seed):
        """Run WalkSAT with the given initial assignment and max flips."""
        rng = random.Random(seed)
        assignment = restart_assignment.copy()
        
        for _ in range(max_flips):
            if time.time() - start_time > timeout:
                return None
            
            sat = is_satisfied(assignment)
            if np.all(sat):
                return assignment.tolist()
            
            unsat_indices = np.where(~sat)[0]
            if len(unsat_indices) == 0:
                return assignment.tolist()
            
            # Pick a random unsatisfied clause
            clause_idx = unsat_indices[rng.randint(0, len(unsat_indices) - 1)]
            
            # Pick a variable from that clause
            # 10% of the time, pick the variable with the best score
            # 90% of the time, pick randomly from the clause
            if rng.random() < 0.1:
                # Score-based selection
                scores = np.zeros(n, dtype=np.int32)
                unsat_vars = clause_vars[unsat_indices]
                for v in unsat_vars.flatten():
                    scores[v] += 1
                
                # Only consider variables in the chosen clause
                clause_vars_list = clause_vars[clause_idx]
                best_var = clause_vars_list[0]
                best_score = -1
                for v in clause_vars_list:
                    if scores[v] > best_score:
                        best_score = scores[v]
                        best_var = v
                var_to_flip = best_var
            else:
                # Random selection from the clause
                var_to_flip = clause_vars[clause_idx, rng.randint(0, 2)]
            
            # Flip the variable
            assignment[var_to_flip] = not assignment[var_to_flip]
        
        # Check if solved
        sat = is_satisfied(assignment)
        if np.all(sat):
            return assignment.tolist()
        
        return None
    
    # Main search loop with restarts
    best_result = None
    rng = random.Random(0)
    restart_count = 0
    
    while time.time() - start_time < timeout - 5:
        # Create a random initial assignment
        initial = np.array([rng.random() < 0.5 for _ in range(n)])
        
        # Run WalkSAT with increasing flip budgets
        max_flips = 5000 + restart_count * 1000
        result = walk_sat(initial, max_flips, seed=restart_count * 12345 + 67890)
        
        if result is not None:
            best_result = result
            break
        
        restart_count += 1
    
    if best_result is not None:
        return best_result
    
    # If we didn't find a solution, return a random assignment
    # (This will score 0, but at least it's a valid format)
    return [rng.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
