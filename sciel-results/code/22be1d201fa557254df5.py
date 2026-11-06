"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Deadline guard
    start_time = time.time()
    timeout = 320.0  # Leave 10s margin before 330s kill
    
    m = len(clauses)
    
    # Precompute clause structure as flat arrays for speed
    clause_vars = np.zeros((m, 3), dtype=np.int32)
    clause_signs = np.zeros((m, 3), dtype=np.int8)
    
    for i, clause in enumerate(clauses):
        for j, lit in enumerate(clause):
            clause_vars[i, j] = abs(lit) - 1
            clause_signs[i, j] = 1 if lit > 0 else 0
    
    # Build variable-to-clauses mapping
    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = int(clause_vars[i, j])
            var_to_clauses[v].append((i, j))
    
    # State tracking using numpy for vectorized operations
    def init_state(assignment):
        """Initialize clause satisfaction count from boolean assignment."""
        # assignment is list of bools, convert to numpy
        assign_arr = np.array(assignment, dtype=np.int8)
        # For each clause, count how many literals are satisfied
        # clause_signs[i][j] is 1 if positive, 0 if negative
        # A literal is satisfied if (assign[var] == 1 and sign == 1) or (assign[var] == 0 and sign == 0)
        # i.e., assign[var] == sign
        # Get all variable values
        var_vals = assign_arr[clause_vars]  # shape (m, 3)
        sat = (var_vals == clause_signs).astype(np.int32)
        return sat.sum(axis=1)
    
    def flip_var(v, assignment, clause_sat_count):
        """Flip variable v and update clause_sat_count in place."""
        new_val = not assignment[v]
        assignment[v] = new_val
        new_val_int = 1 if new_val else 0
        for ci, j in var_to_clauses[v]:
            if new_val_int == int(clause_signs[ci, j]):
                clause_sat_count[ci] += 1
            else:
                clause_sat_count[ci] -= 1
    
    # WalkSAT with adaptive parameters and restarts
    best_assignment = None
    best_unsat_count = m + 1
    
    rng = random.Random(42)
    
    # Adaptive parameters
    base_noise = 0.25
    base_max_flips = 10000
    
    while time.time() - start_time < timeout:
        # Random restart
        assignment = [rng.random() < 0.5 for _ in range(n)]
        clause_sat_count = init_state(assignment)
        
        current_unsat = int(np.sum(clause_sat_count == 0))
        if current_unsat == 0:
            return assignment
        
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment.copy()
        
        # Adaptive: increase flips if we're making progress
        max_flips = base_max_flips
        noise = base_noise
        
        for flip_num in range(max_flips):
            if time.time() - start_time > timeout - 5.0:
                break
            
            # Find unsatisfied clauses
            unsat_mask = clause_sat_count == 0
            unsat_count = int(np.sum(unsat_mask))
            if unsat_count == 0:
                return assignment
            
            # Pick a random unsatisfied clause
            unsat_indices = np.where(unsat_mask)[0]
            ci = int(rng.choice(unsat_indices))
            
            # WalkSAT heuristic with adaptive noise
            if rng.random() < noise:
                # Random pick among the 3 vars
                j = rng.randint(0, 2)
                v = int(clause_vars[ci, j])
            else:
                # Greedy: pick var that satisfies most other unsat clauses
                best_var = -1
                best_score = -1
                for j in range(3):
                    v = int(clause_vars[ci, j])
                    score = 0
                    for other_ci, _ in var_to_clauses[v]:
                        if other_ci != ci and clause_sat_count[other_ci] == 0:
                            score += 1
                    if score > best_score:
                        best_score = score
                        best_var = v
                v = best_var
            
            if v >= 0:
                flip_var(v, assignment, clause_sat_count)
            
            current_unsat = int(np.sum(clause_sat_count == 0))
            if current_unsat == 0:
                return assignment
            
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment.copy()
        
        # If we didn't solve, try a refinement pass from best assignment
        if best_assignment is not None and time.time() - start_time < timeout - 10.0:
            # Do a focused local search from the best assignment
            assignment = best_assignment.copy()
            clause_sat_count = init_state(assignment)
            current_unsat = int(np.sum(clause_sat_count == 0))
            
            # Use a tighter noise for refinement
            refine_noise = 0.1
            for _ in range(5000):
                if time.time() - start_time > timeout - 5.0:
                    break
                
                unsat_mask = clause_sat_count == 0
                unsat_count = int(np.sum(unsat_mask))
                if unsat_count == 0:
                    return assignment
                
                unsat_indices = np.where(unsat_mask)[0]
                ci = int(rng.choice(unsat_indices))
                
                if rng.random() < refine_noise:
                    j = rng.randint(0, 2)
                    v = int(clause_vars[ci, j])
                else:
                    best_var = -1
                    best_score = -1
                    for j in range(3):
                        v = int(clause_vars[ci, j])
                        score = 0
                        for other_ci, _ in var_to_clauses[v]:
                            if other_ci != ci and clause_sat_count[other_ci] == 0:
                                score += 1
                        if score > best_score:
                            best_score = score
                            best_var = v
                    v = best_var
                
                if v >= 0:
                    flip_var(v, assignment, clause_sat_count)
                
                current_unsat = int(np.sum(clause_sat_count == 0))
                if current_unsat == 0:
                    return assignment
                
                if current_unsat < best_unsat_count:
                    best_unsat_count = current_unsat
                    best_assignment = assignment.copy()
    
    # Return the best assignment found
    if best_assignment is not None:
        return best_assignment
    else:
        return [rng.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
