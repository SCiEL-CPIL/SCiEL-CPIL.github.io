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
    timeout = 315.0  # Leave margin for overhead
    
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
    
    def is_satisfied(assignment):
        """Return boolean array of satisfied clauses."""
        # assignment is a numpy array of bools
        vals = assignment[clause_vars]  # (m, 3) bools
        sign_bool = (clause_data == 1)  # (m, 3)
        literal_true = (vals == sign_bool)  # (m, 3)
        return np.any(literal_true, axis=1)  # (m,) bools
    
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
