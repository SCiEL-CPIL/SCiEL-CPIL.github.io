"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math


def solve(instance):
    """
    Solve a planted 3-SAT instance with 50 variables using an improved
    WalkSAT with multiple restarts, noise tuning, and a time budget.
    """
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 280  # safety margin before 300s kill
    
    # Pre-compute clause structure for speed
    # Represent each clause as a tuple of (var_index, sign) where sign is +1 or -1
    # var_index is 0-based
    clause_data = []
    for clause in clauses:
        cd = []
        for lit in clause:
            var_idx = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            cd.append((var_idx, sign))
        clause_data.append(tuple(cd))
    
    m = len(clause_data)
    
    def is_satisfied(clause_idx, assignment):
        for var_idx, sign in clause_data[clause_idx]:
            if assignment[var_idx] == (sign == 1):
                return True
        return False
    
    def count_satisfied(assignment):
        count = 0
        for i in range(m):
            for var_idx, sign in clause_data[i]:
                if assignment[var_idx] == (sign == 1):
                    count += 1
                    break
        return count
    
    best_assignment = None
    best_score = -1
    
    rng = random.Random(42)
    
    # Try multiple strategies with restarts
    # Strategy 1: WalkSAT with varying noise parameters
    noise_params = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    max_restarts = 500
    max_steps_per_restart = 5000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Choose noise parameter
        if restart < len(noise_params):
            noise = noise_params[restart]
        else:
            noise = rng.choice(noise_params)
        
        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Track best assignment in this restart
        local_best = None
        local_best_score = -1
        
        for step in range(max_steps_per_restart):
            if time.time() > deadline:
                break
            
            # Check if all clauses are satisfied
            unsat_clauses = []
            sat_count = 0
            for i in range(m):
                sat = False
                for var_idx, sign in clause_data[i]:
                    if assignment[var_idx] == (sign == 1):
                        sat = True
                        break
                if sat:
                    sat_count += 1
                else:
                    unsat_clauses.append(i)
            
            if sat_count > local_best_score:
                local_best_score = sat_count
                local_best = assignment[:]
            
            if not unsat_clauses:
                # Found a solution!
                return assignment
            
            if step % 500 == 0 and time.time() > deadline:
                break
            
            # Pick a random unsatisfied clause
            clause_idx = rng.choice(unsat_clauses)
            clause = clause_data[clause_idx]
            
            # Count how many literals in the clause would become true if we flip each variable
            # WalkSAT: flip the variable that maximizes the number of newly satisfied clauses
            # with some noise
            
            best_gain = -1
            best_var = None
            gains = []
            
            for var_idx, sign in clause:
                # Simulate flipping this variable
                old_val = assignment[var_idx]
                assignment[var_idx] = not old_val
                
                # Count satisfied clauses
                sat = 0
                for i in range(m):
                    for vi, s in clause_data[i]:
                        if assignment[vi] == (s == 1):
                            sat += 1
                            break
                
                assignment[var_idx] = old_val
                gains.append(sat)
            
            # Choose with noise
            max_gain = max(gains)
            candidates = [i for i, g in enumerate(gains) if g == max_gain]
            
            if rng.random() < noise:
                # Pick randomly from all variables in the clause
                choice_idx = rng.randrange(len(clause))
            else:
                # Pick from best candidates
                choice_idx = rng.choice(candidates)
            
            var_idx = clause[choice_idx][0]
            assignment[var_idx] = not assignment[var_idx]
        
        # After restart, keep the best assignment found
        if local_best_score > best_score:
            best_score = local_best_score
            best_assignment = local_best
    
    # If no solution found within time, return the best assignment we found
    if best_assignment is not None:
        return best_assignment
    
    # Fallback: return a random assignment
    return [random.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
