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
    
    # Build variable-to-clauses mapping
    var_to_clauses = [[] for _ in range(n)]
    for i, vars_list in enumerate(clause_vars):
        for v in vars_list:
            var_to_clauses[v].append(i)
    
    # WalkSAT with optimized state tracking
    rng = random.Random(42)
    
    def init_state(assignment):
        """Initialize clause satisfaction state and auxiliary structures."""
        clause_sat_count = [0] * m
        unsat_clauses = set()
        
        for i in range(m):
            count = 0
            for j, v in enumerate(clause_vars[i]):
                if assignment[v] == clause_signs[i][j]:
                    count += 1
            clause_sat_count[i] = count
            if count == 0:
                unsat_clauses.add(i)
        
        # Initialize var_unsat_count
        var_unsat_count = [0] * n
        for c in unsat_clauses:
            for v in clause_vars[c]:
                var_unsat_count[v] += 1
                
        return clause_sat_count, unsat_clauses, var_unsat_count

    def flip_var(v, assignment, clause_sat_count, unsat_clauses, var_unsat_count):
        """Flip variable v and update all state structures."""
        old_val = assignment[v]
        new_val = not old_val
        assignment[v] = new_val
        
        for ci in var_to_clauses[v]:
            # Determine the position of v in clause ci to check sign
            # We need to know if the literal is satisfied by new_val
            # Find j such that clause_vars[ci][j] == v
            # Since clause size is 3, this is fast, but we can precompute index
            # For simplicity and speed, we iterate the 3 vars
            j = -1
            for k in range(3):
                if clause_vars[ci][k] == v:
                    j = k
                    break
            
            # old_sat: was the literal satisfied before?
            # new_sat: is the literal satisfied now?
            # Since we flipped v, the status of the literal for v in this clause changes.
            # If old_val == sign, it was satisfied. Now new_val != sign, so it's unsatisfied.
            # If old_val != sign, it was unsatisfied. Now new_val == sign, so it's satisfied.
            
            # Actually, we just need to update the count.
            # If the literal was satisfied before (old_val == sign), count decreases by 1.
            # If the literal was unsatisfied before (old_val != sign), count increases by 1.
            
            if old_val == clause_signs[ci][j]:
                # Was satisfied, now unsatisfied
                clause_sat_count[ci] -= 1
                if clause_sat_count[ci] == 0:
                    # Clause became unsatisfied
                    unsat_clauses.add(ci)
                    # Update var_unsat_count for all vars in this clause
                    for vv in clause_vars[ci]:
                        var_unsat_count[vv] += 1
            else:
                # Was unsatisfied, now satisfied
                clause_sat_count[ci] += 1
                if clause_sat_count[ci] == 1:
                    # Clause became satisfied (was 0, now 1)
                    unsat_clauses.remove(ci)
                    # Update var_unsat_count for all vars in this clause
                    for vv in clause_vars[ci]:
                        var_unsat_count[vv] -= 1

    best_assignment = None
    best_unsat_count = m + 1
    
    # Heuristic parameter: probability of picking a random variable from the clause
    # vs the variable that appears in the most other unsatisfied clauses.
    # Typical values for p are 0.2 to 0.5.
    p = 0.3 
    
    max_flips_per_restart = 20000
    num_restarts = 0
    
    while time.time() - start_time < timeout:
        # Random restart
        assignment = [rng.random() < 0.5 for _ in range(n)]
        clause_sat_count, unsat_clauses, var_unsat_count = init_state(assignment)
        
        if not unsat_clauses:
            return assignment
        
        current_unsat = len(unsat_clauses)
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment.copy()
        
        for flip_num in range(max_flips_per_restart):
            if time.time() - start_time > timeout - 5.0:
                break
            
            if not unsat_clauses:
                return assignment
            
            # Pick a random unsatisfied clause
            ci = rng.choice(list(unsat_clauses))
            
            # Select variable to flip
            # Heuristic: with probability p, pick a random variable from the clause.
            # Otherwise, pick the variable that appears in the most other unsatisfied clauses.
            if rng.random() < p:
                v = rng.choice(clause_vars[ci])
            else:
                # Find variable in clause with max var_unsat_count
                best_v = -1
                best_score = -1
                for vv in clause_vars[ci]:
                    if var_unsat_count[vv] > best_score:
                        best_score = var_unsat_count[vv]
                        best_v = vv
                v = best_v
            
            flip_var(v, assignment, clause_sat_count, unsat_clauses, var_unsat_count)
            
            current_unsat = len(unsat_clauses)
            if current_unsat == 0:
                return assignment
            
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment.copy()
        
        num_restarts += 1
        # Optionally adjust p or other parameters based on progress
        # For now, keep it simple.
        
    if best_assignment is not None:
        return best_assignment
    else:
        return [rng.random() < 0.5 for _ in range(n)]

# EVOLVE-BLOCK-END
