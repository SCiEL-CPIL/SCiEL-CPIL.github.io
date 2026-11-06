"""Improved WalkSAT solver with vectorized operations and deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def solve(instance):
    """
    Solve a 3-SAT instance using an optimized WalkSAT strategy.
    
    Args:
        instance: dict with 'n' (number of variables) and 'clauses' (list of lists of signed literals).
        
    Returns:
        list of booleans representing the assignment.
    """
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Deadline: 320 seconds to be safe within 330s limit
    start_time = time.time()
    deadline = start_time + 320.0
    
    # Preprocess clauses for faster access
    # Convert to 0-indexed and separate signs
    num_clauses = len(clauses)
    if num_clauses == 0:
        return [True] * n

    # Create arrays for vectorized operations
    # clause_vars[i, j] is the 0-indexed variable of the j-th literal in clause i
    # clause_signs[i, j] is 1 if positive, -1 if negative
    max_clause_len = 3  # 3-SAT
    clause_vars = np.zeros((num_clauses, max_clause_len), dtype=np.int32)
    clause_signs = np.zeros((num_clauses, max_clause_len), dtype=np.int8)
    clause_lens = np.zeros(num_clauses, dtype=np.int8)
    
    for i, c in enumerate(clauses):
        for j, lit in enumerate(c):
            clause_vars[i, j] = abs(lit) - 1
            clause_signs[i, j] = 1 if lit > 0 else -1
        clause_lens[i] = len(c)
        
    # Mask for valid literals in each clause
    clause_mask = np.zeros((num_clauses, max_clause_len), dtype=bool)
    for i in range(num_clauses):
        clause_mask[i, :clause_lens[i]] = True

    # Variable occurrence counts for heuristic weighting
    # count_pos[v] = number of clauses where v appears positively
    # count_neg[v] = number of clauses where v appears negatively
    count_pos = np.zeros(n, dtype=np.int32)
    count_neg = np.zeros(n, dtype=np.int32)
    
    for i in range(num_clauses):
        for j in range(clause_lens[i]):
            v = clause_vars[i, j]
            if clause_signs[i, j] > 0:
                count_pos[v] += 1
            else:
                count_neg[v] += 1

    def check_satisfied(assignment):
        """
        Check which clauses are satisfied.
        Returns a boolean array of shape (num_clauses,) where True means satisfied.
        """
        # assignment is a boolean array of shape (n,)
        # We need to check: for each clause, does any literal evaluate to True?
        # Literal value = assignment[var] == (sign > 0)
        
        # Expand assignment to match clause_vars shape
        # var_vals[i, j] = assignment[clause_vars[i, j]]
        var_vals = assignment[clause_vars]
        
        # lit_vals[i, j] = 1 if literal is true, 0 otherwise
        # If sign is 1, lit is true if var_val is True (1)
        # If sign is -1, lit is true if var_val is False (0)
        # lit_val = (var_val == 1) if sign==1 else (var_val == 0)
        # This can be written as: (var_val * sign) > 0 ? No, that's for integers.
        # Let's use: lit_true = (var_vals == 1) if sign==1 else (var_vals == 0)
        # Vectorized: lit_true = (var_vals * clause_signs > 0) ? 
        # If sign=1, var=1 -> 1>0 True. var=0 -> 0>0 False.
        # If sign=-1, var=1 -> -1>0 False. var=0 -> 0>0 False. Wait.
        # If sign=-1 and var=0 (False), literal is True.
        # So: lit_true = (var_vals == 1) * (clause_signs == 1) + (var_vals == 0) * (clause_signs == -1)
        
        lit_true = ((var_vals == 1) & (clause_signs == 1)) | ((var_vals == 0) & (clause_signs == -1))
        
        # Mask out invalid positions
        lit_true = lit_true & clause_mask
        
        # A clause is satisfied if any literal in it is true
        # sum along axis 1, if > 0 then satisfied
        clause_satisfied = np.any(lit_true, axis=1)
        
        return clause_satisfied

    def get_unsat_clauses(assignment):
        sat = check_satisfied(assignment)
        return np.where(~sat)[0]

    # Main Search Loop
    best_assignment = None
    best_unsat_count = num_clauses + 1
    rng = random.Random(42)
    
    # Initialize assignment randomly
    assignment = np.array([rng.random() < 0.5 for _ in range(n)], dtype=bool)
    
    # WalkSAT parameters
    p_flip_good = 0.5  # Probability of flipping a literal that satisfies the clause
    max_steps_per_restart = 5000
    max_restarts = 500
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Reset assignment for restart (or keep if we want to continue from last best? 
        # Standard WalkSAT restarts randomly, but for planted instances, 
        # restarting from a good partial assignment can be better. 
        # Let's restart randomly to avoid local minima, but keep track of best.)
        if best_assignment is not None and rng.random() < 0.2:
            # Sometimes resume from best known
            assignment = best_assignment.copy()
        else:
            assignment = np.array([rng.random() < 0.5 for _ in range(n)], dtype=bool)
        
        unsat_clauses = get_unsat_clauses(assignment)
        current_unsat = len(unsat_clauses)
        
        if current_unsat == 0:
            return assignment.tolist()
            
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment.copy()
            
        steps_in_restart = 0
        while steps_in_restart < max_steps_per_restart:
            if time.time() > deadline:
                break
                
            if len(unsat_clauses) == 0:
                return assignment.tolist()
                
            # Pick a random unsatisfied clause
            clause_idx = unsat_clauses[rng.randrange(len(unsat_clauses))]
            clause = clauses[clause_idx]
            
            # Determine which literals in this clause are currently false
            # We want to flip one of them.
            # A literal is false if: (sign=1 and var=False) or (sign=-1 and var=True)
            # i.e., assignment[var] != (sign > 0)
            
            candidates = []
            for lit in clause:
                var = abs(lit) - 1
                sign_positive = lit > 0
                if assignment[var] != sign_positive:
                    candidates.append(var)
            
            if not candidates:
                # This shouldn't happen if clause is unsatisfied, but safety check
                break
                
            # Choose a literal to flip
            # Heuristic: Prefer variables that are involved in many other unsatisfied clauses?
            # Or just random (classic WalkSAT)?
            # Let's use a weighted random choice based on how "bad" the variable is.
            # Actually, simple random is often robust. Let's stick to simple random 
            # but with a slight bias towards variables with higher degree.
            
            # For speed, just pick random
            var_to_flip = candidates[rng.randrange(len(candidates))]
            
            # Flip the variable
            assignment[var_to_flip] = not assignment[var_to_flip]
            
            # Update unsat clauses efficiently?
            # Recalculating all is O(C*3). With N=50, C~500, it's 1500 ops.
            # 5000 steps * 1500 ops = 7.5M ops per restart.
            # NumPy overhead might make this slow.
            # Let's check time.
            
            unsat_clauses = get_unsat_clauses(assignment)
            current_unsat = len(unsat_clauses)
            
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment.copy()
                
            steps_in_restart += 1
            
            # If we get stuck in a loop, force a random flip
            if steps_in_restart % 100 == 0 and current_unsat == best_unsat_count:
                # Random flip to escape local minimum
                v = rng.randrange(n)
                assignment[v] = not assignment[v]
                unsat_clauses = get_unsat_clauses(assignment)
                current_unsat = len(unsat_clauses)
                if current_unsat < best_unsat_count:
                    best_unsat_count = current_unsat
                    best_assignment = assignment.copy()

    # Return the best assignment found
    if best_assignment is not None:
        return best_assignment.tolist()
    else:
        return assignment.tolist()

# EVOLVE-BLOCK-END
