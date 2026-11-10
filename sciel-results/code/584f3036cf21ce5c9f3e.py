"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 280  # safety margin before 330s kill
    
    # Precompute clause literal indices for speed
    clause_indices = []
    for c in clauses:
        idxs = [abs(l) - 1 for l in c]
        signs = [l > 0 for l in c]
        clause_indices.append((idxs, signs))
    
    def count_unsat(assignment):
        count = 0
        for idxs, signs in clause_indices:
            sat = False
            for i, s in zip(idxs, signs):
                if assignment[i] == s:
                    sat = True
                    break
            if not sat:
                count += 1
        return count
    
    def is_satisfied(assignment):
        for idxs, signs in clause_indices:
            sat = False
            for i, s in zip(idxs, signs):
                if assignment[i] == s:
                    sat = True
                    break
            if not sat:
                return False
        return True
    
    # Try multiple strategies with restarts
    max_restarts = 50
    flips_per_restart = 5000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        rng = random.Random(restart * 1000 + 42)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Min-conflicts heuristic: flip variable in most unsat clauses
        for _ in range(flips_per_restart):
            if time.time() > deadline:
                break
            
            # Find unsatisfied clauses
            unsat_clauses = []
            for idxs, signs in clause_indices:
                sat = False
                for i, s in zip(idxs, signs):
                    if assignment[i] == s:
                        sat = True
                        break
                if not sat:
                    unsat_clauses.append((idxs, signs))
            
            if not unsat_clauses:
                return assignment
            
            # Count how many unsat clauses each variable appears in
            var_count = [0] * n
            for idxs, signs in unsat_clauses:
                for i in idxs:
                    var_count[i] += 1
            
            # Find variable with max count
            max_count = max(var_count)
            if max_count == 0:
                # Shouldn't happen, but break
                break
            
            # Choose among variables with max count
            candidates = [i for i in range(n) if var_count[i] == max_count]
            var_to_flip = rng.choice(candidates)
            
            # Flip it
            assignment[var_to_flip] = not assignment[var_to_flip]
        
        # After min-conflicts, try a short WalkSAT polish
        if time.time() > deadline:
            break
        
        for _ in range(2000):
            if time.time() > deadline:
                break
            
            unsat_clauses = []
            for idxs, signs in clause_indices:
                sat = False
                for i, s in zip(idxs, signs):
                    if assignment[i] == s:
                        sat = True
                        break
                if not sat:
                    unsat_clauses.append((idxs, signs))
            
            if not unsat_clauses:
                return assignment
            
            # Pick random unsat clause, flip random literal in it
            idxs, signs = rng.choice(unsat_clauses)
            var_to_flip = rng.choice(idxs)
            assignment[var_to_flip] = not assignment[var_to_flip]
    
    # Final check
    if is_satisfied(assignment):
        return assignment
    
    return assignment
# EVOLVE-BLOCK-END
