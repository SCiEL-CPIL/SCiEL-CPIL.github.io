"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Deadline guard
    start_time = time.time()
    timeout = 325.0  # Leave 5s margin before 330s kill
    
    # Precompute clause structure
    clause_vars = []
    clause_signs = []
    
    for clause in clauses:
        vars_list = []
        signs_list = []
        for lit in clause:
            var_idx = abs(lit) - 1
            vars_list.append(var_idx)
            signs_list.append(lit > 0)
        clause_vars.append(vars_list)
        clause_signs.append(signs_list)
    
    m = len(clauses)
    
    # Build variable-to-clauses mapping with position info
    var_to_clauses = [[] for _ in range(n)]
    for i, vars_list in enumerate(clause_vars):
        for j, v in enumerate(vars_list):
            var_to_clauses[v].append((i, j))
    
    # State tracking
    def init_state(assignment):
        """Initialize clause satisfaction state from assignment."""
        clause_sat_count = np.zeros(m, dtype=np.int32)
        for i in range(m):
            count = 0
            for j, v in enumerate(clause_vars[i]):
                if assignment[v] == clause_signs[i][j]:
                    count += 1
            clause_sat_count[i] = count
        return clause_sat_count
    
    def flip_var(v, assignment, clause_sat_count):
        """Flip variable v and update clause_sat_count in place."""
        assignment[v] = not assignment[v]
        for ci, j in var_to_clauses[v]:
            if assignment[v] == clause_signs[ci][j]:
                clause_sat_count[ci] += 1
            else:
                clause_sat_count[ci] -= 1
    
    # WalkSAT with restarts and time budget
    best_assignment = None
    best_unsat_count = m + 1
    
    rng = random.Random(42)
    max_flips_per_restart = 8000
    noise_param = 0.2  # Probability of random flip among tied candidates
    
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
        
        for flip_num in range(max_flips_per_restart):
            if time.time() - start_time > timeout - 5.0:
                break
            
            # Find unsatisfied clauses
            unsat_indices = np.where(clause_sat_count == 0)[0]
            if len(unsat_indices) == 0:
                return assignment
            
            # Pick a random unsatisfied clause
            ci = int(rng.choice(unsat_indices))
            
            # WalkSAT heuristic: for each var in clause, count how many
            # OTHER unsatisfied clauses it appears in
            # With noise: with prob noise_param, pick uniformly at random
            if rng.random() < noise_param:
                # Random pick
                j = rng.randint(0, 2)
                v = clause_vars[ci][j]
            else:
                # Greedy: pick var that satisfies most other unsat clauses
                best_var = -1
                best_score = -1
                for j, v in enumerate(clause_vars[ci]):
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
