"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math
import numpy as np

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s kill
    
    # Preprocess: convert clauses to 0-based indices and store signs
    # clause_lit[i] = (var_idx, sign) where sign is +1 or -1
    clause_lits = []
    for c in clauses:
        lits = [(abs(l) - 1, 1 if l > 0 else -1) for l in c]
        clause_lits.append(lits)
    
    m = len(clause_lits)
    
    # Build variable to clauses mapping
    var_to_clauses = [[] for _ in range(n)]
    for ci, lits in enumerate(clause_lits):
        for var, sign in lits:
            var_to_clauses[var].append((ci, sign))
    
    def is_satisfied(ci, assignment):
        lits = clause_lits[ci]
        for var, sign in lits:
            if assignment[var] == (sign == 1):
                return True
        return False
    
    def count_unsat(assignment):
        count = 0
        for ci in range(m):
            if not is_satisfied(ci, assignment):
                count += 1
        return count
    
    # Better WalkSAT with gain heuristic and restarts
    def walksat_one_run(rng, max_steps):
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Compute initial unsat count
        unsat_count = count_unsat(assignment)
        if unsat_count == 0:
            return assignment, True
        
        # Track unsat clauses
        unsat_clauses = set()
        for ci in range(m):
            if not is_satisfied(ci, assignment):
                unsat_clauses.add(ci)
        
        step = 0
        while step < max_steps:
            if time.time() > deadline:
                break
            
            if not unsat_clauses:
                return assignment, True
            
            # Pick a random unsat clause
            ci = rng.choice(list(unsat_clauses))
            lits = clause_lits[ci]
            
            # For each literal in the clause, compute gain:
            # gain = (number of unsat clauses that would become satisfied) - 
            #        (number of satisfied clauses that would become unsatisfied)
            best_var = -1
            best_gain = -1000000
            
            for var, sign in lits:
                # If we flip var, what happens?
                # Current state of var
                current_val = assignment[var]
                new_val = not current_val
                
                # Clauses containing var
                gain = 0
                for (ci2, sign2) in var_to_clauses[var]:
                    # Check if clause ci2 is currently satisfied
                    ci2_satisfied = ci2 in unsat_clauses
                    # After flip, is ci2 satisfied?
                    # var's new value is new_val, sign2 indicates what value satisfies this literal
                    var_satisfies = (new_val == (sign2 == 1))
                    
                    if not ci2_satisfied:
                        # Clause is unsat. If flipping var makes it satisfied, +1
                        # Check if any other literal in ci2 is already satisfied
                        other_satisfied = False
                        for (v2, s2) in clause_lits[ci2]:
                            if v2 != var:
                                if assignment[v2] == (s2 == 1):
                                    other_satisfied = True
                                    break
                        if not other_satisfied and var_satisfies:
                            gain += 1
                    else:
                        # Clause is sat. If flipping var makes it unsat, -1
                        # Check if any other literal in ci2 is satisfied
                        other_satisfied = False
                        for (v2, s2) in clause_lits[ci2]:
                            if v2 != var:
                                if assignment[v2] == (s2 == 1):
                                    other_satisfied = True
                                    break
                        if not other_satisfied and not var_satisfies:
                            gain -= 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_var = var
            
            if best_var == -1:
                # Fallback: pick random literal
                var = rng.choice(lits)[0]
            else:
                var = best_var
            
            # Flip
            assignment[var] = not assignment[var]
            
            # Update unsat_clauses
            for (ci2, sign2) in var_to_clauses[var]:
                if ci2 in unsat_clauses:
                    # Was unsat, check if now sat
                    if is_satisfied(ci2, assignment):
                        unsat_clauses.discard(ci2)
                else:
                    # Was sat, check if now unsat
                    if not is_satisfied(ci2, assignment):
                        unsat_clauses.add(ci2)
            
            step += 1
        
        return assignment, False
    
    # Try multiple runs with different random seeds and parameters
    rng = random.Random(0)
    
    # First, try a simple random assignment to see if it works
    assignment = [rng.random() < 0.5 for _ in range(n)]
    if count_unsat(assignment) == 0:
        return assignment
    
    # WalkSAT with gain heuristic
    best_assignment = None
    best_unsat_count = float('inf')
    
    for run in range(50):
        if time.time() > deadline:
            break
        
        # Use different random seed for each run
        rng_run = random.Random(run * 12345 + 67890)
        
        max_steps = 50000 if run < 10 else 20000
        assignment_run, success = walksat_one_run(rng_run, max_steps)
        
        if success:
            return assignment_run
        
        # Track best
        unsat = count_unsat(assignment_run)
        if unsat < best_unsat_count:
            best_unsat_count = unsat
            best_assignment = assignment_run[:]
        
        # If we're close, try a final polish
        if unsat <= 5:
            # Try to fix the remaining unsat clauses
            for _ in range(10000):
                if time.time() > deadline:
                    break
                unsat_clauses = [ci for ci in range(m) if not is_satisfied(ci, assignment_run)]
                if not unsat_clauses:
                    return assignment_run
                
                # Try flipping variables in unsat clauses
                ci = rng_run.choice(unsat_clauses)
                lits = clause_lits[ci]
                var = rng_run.choice(lits)[0]
                assignment_run[var] = not assignment_run[var]
            
            unsat = count_unsat(assignment_run)
            if unsat < best_unsat_count:
                best_unsat_count = unsat
                best_assignment = assignment_run[:]
    
    return best_assignment if best_assignment is not None else [False] * n
# EVOLVE-BLOCK-END
