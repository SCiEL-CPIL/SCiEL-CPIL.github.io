"""Rewritten program for planted 3-SAT solver."""
import numpy as np
import time
import random

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    if not clauses:
        return [False] * n

    # Convert to numpy arrays
    clause_arr = np.array(clauses, dtype=np.int32)
    m = clause_arr.shape[0]
    
    # 0-indexed variables
    var_indices = np.abs(clause_arr) - 1
    # Signs: 1 for positive, -1 for negative
    signs = np.sign(clause_arr)
    
    # Use a fixed seed for reproducibility if needed, but random restarts are better
    # The task says "locally generated", implying the instance is fixed per call.
    # We should use a random seed for the solver's internal randomness to avoid bias.
    rng = np.random.default_rng()
    
    best_assignment = None
    best_unsat_count = m + 1
    
    start_time = time.time()
    # Safety margin: 290s out of 300s limit
    deadline = start_time + 290.0
    
    # Restart strategy: Geometric distribution with mean 5000 steps
    # This allows for long runs on easy instances and quick restarts on hard ones.
    
    while time.time() < deadline:
        # Initialize with random assignment
        assignment = rng.integers(0, 2, size=n).astype(np.int8)
        
        # Determine steps for this restart
        # Using exponential distribution for restart length
        steps_limit = int(rng.exponential(5000) + 1000)
        
        for _ in range(steps_limit):
            if time.time() > deadline:
                break
            
            # Vectorized evaluation
            # sat_vals[i, j] is 1 if literal j in clause i is satisfied by assignment
            # If sign is positive, we want assignment[var] == 1
            # If sign is negative, we want assignment[var] == 0
            sat_vals = np.where(signs > 0, assignment[var_indices], 1 - assignment[var_indices])
            clause_sat_counts = sat_vals.sum(axis=1)
            
            unsat_mask = clause_sat_counts == 0
            unsat_indices = np.where(unsat_mask)[0]
            current_unsat = len(unsat_indices)
            
            # Track best
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment.copy()
                
            if current_unsat == 0:
                return assignment.tolist()
            
            # Select clause
            # Pick a random unsatisfied clause
            target_clause_idx = rng.choice(unsat_indices)
            target_clause_vars = var_indices[target_clause_idx]
            
            # Select variable
            # Heuristic: Pick the variable in the target clause that appears
            # most frequently in the current set of unsatisfied clauses.
            # This helps resolve conflicts globally.
            
            # Get all variables in unsatisfied clauses
            unsat_vars = var_indices[unsat_indices].flatten()
            
            # Count occurrences of each variable in unsatisfied clauses
            # minlength=n ensures the array size is n
            counts = np.bincount(unsat_vars, minlength=n)
            
            # Get weights for the 3 variables in the target clause
            weights = counts[target_clause_vars]
            
            # Find the max weight
            max_weight = np.max(weights)
            
            # If there are multiple variables with the max weight, pick one randomly
            candidates = target_clause_vars[weights == max_weight]
            var_to_flip = rng.choice(candidates)
            
            # Flip the variable
            assignment[var_to_flip] = 1 - assignment[var_to_flip]
            
        # End of restart steps
    
    # If time runs out, return the best assignment found
    if best_assignment is not None:
        return best_assignment.tolist()
    else:
        return [False] * n
