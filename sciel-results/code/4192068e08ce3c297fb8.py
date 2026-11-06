"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Preprocess
    clause_vars = []
    for c in clauses:
        cv = []
        for l in c:
            var_idx = abs(l) - 1
            is_pos = l > 0
            cv.append((var_idx, is_pos))
        clause_vars.append(cv)
        
    m = len(clause_vars)
    
    # Variable to clauses mapping
    var_to_clauses = [[] for _ in range(n)]
    for i, cv in enumerate(clause_vars):
        for var_idx, is_pos in cv:
            var_to_clauses[var_idx].append(i)
            
    start_time = time.time()
    deadline = start_time + 320
    
    rng = random.Random(0)
    
    best_unsat_count = float('inf')
    best_assignment = None
    
    restart_threshold = 10000
    max_restarts = 1000000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Compute initial clause satisfaction
        clause_sat = [False] * m
        unsat_count = 0
        for i, cv in enumerate(clause_vars):
            sat = False
            for var_idx, is_pos in cv:
                if assignment[var_idx] == is_pos:
                    sat = True
                    break
            clause_sat[i] = sat
            if not sat:
                unsat_count += 1
                
        if unsat_count == 0:
            return assignment
            
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
            
        steps = 0
        while steps < restart_threshold:
            if time.time() > deadline:
                break
                
            # Pick a random unsatisfied clause
            # To avoid O(m) scan every time, we can pick a random clause and check.
            # If it's satisfied, try again.
            # If m is large and unsat_count is small, this is inefficient.
            # Let's maintain a list of unsat clauses.
            
            # For simplicity in this iteration, let's just pick a random clause index.
            # If it's unsat, proceed.
            # This works well when unsat_count is significant.
            
            c_idx = rng.randrange(m)
            if clause_sat[c_idx]:
                continue
                
            cv = clause_vars[c_idx]
            
            # Damage minimizing heuristic
            # Pick variable with fewest appearances in satisfied clauses
            best_var = -1
            best_score = float('inf')
            
            for var_idx, is_pos in cv:
                # Count satisfied clauses containing this variable
                score = 0
                for c_prime_idx in var_to_clauses[var_idx]:
                    if clause_sat[c_prime_idx]:
                        score += 1
                        
                if score < best_score:
                    best_score = score
                    best_var = var_idx
                    
            if best_var == -1:
                var_idx, is_pos = rng.choice(cv)
            else:
                # Get polarity
                for var_idx, is_pos in cv:
                    if var_idx == best_var:
                        break
                        
            # Flip
            assignment[best_var] = not assignment[best_var]
            
            # Update clause_sat for clauses containing best_var
            for c_prime_idx in var_to_clauses[best_var]:
                old_sat = clause_sat[c_prime_idx]
                new_sat = False
                for v2, p2 in clause_vars[c_prime_idx]:
                    if assignment[v2] == p2:
                        new_sat = True
                        break
                if old_sat != new_sat:
                    clause_sat[c_prime_idx] = new_sat
                    if old_sat:
                        unsat_count += 1
                    else:
                        unsat_count -= 1
                        
            if unsat_count == 0:
                return assignment
                
            if unsat_count < best_unsat_count:
                best_unsat_count = unsat_count
                best_assignment = assignment[:]
                
            steps += 1
            
    return best_assignment if best_assignment else assignment

# EVOLVE-BLOCK-END
