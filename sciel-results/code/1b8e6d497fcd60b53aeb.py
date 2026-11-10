"""Simulated Annealing based 3-SAT solver."""
# EVOLVE-BLOCK-START
import random
import math


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(0)
    
    # Pre-compute variable to clause indices
    # var_to_clauses[i] will be a list of tuples (clause_idx, literal_sign)
    # literal_sign is 1 if clause contains +i, -1 if -i
    var_to_clauses = [[] for _ in range(n)]
    for c_idx, clause in enumerate(clauses):
        for lit in clause:
            var_idx = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            var_to_clauses[var_idx].append((c_idx, sign))

    # Random initial assignment
    # Use 0/1 integers for speed
    assignment = [rng.randint(0, 1) for _ in range(n)]
    
    # Helper to check if a clause is satisfied
    def is_satisfied(clause, assignment):
        for lit in clause:
            if assignment[abs(lit) - 1] == (1 if lit > 0 else 0):
                return True
        return False
        
    # Count unsatisfied clauses
    def count_unsat(assignment, clauses):
        count = 0
        for clause in clauses:
            if not is_satisfied(clause, assignment):
                count += 1
        return count

    # Initial energy
    current_energy = count_unsat(assignment, clauses)
    
    if current_energy == 0:
        return [bool(x) for x in assignment]

    # Simulated Annealing parameters
    # With efficient delta calculation, we can afford many more iterations
    T_start = 10.0
    T_end = 0.01
    max_iterations = 3000000
    
    # Calculate cooling rate to reach T_end from T_start in max_iterations
    if max_iterations > 0:
        cooling_rate = (T_end / T_start) ** (1.0 / max_iterations)
    else:
        cooling_rate = 0.99
        
    T = T_start
    
    for iteration in range(max_iterations):
        if current_energy == 0:
            break
        
        # Choose a random variable to flip
        var = rng.randint(0, n - 1)
        
        # Calculate delta energy
        old_val = assignment[var]
        new_val = 1 - old_val
        
        delta = 0
        
        for c_idx, sign in var_to_clauses[var]:
            clause = clauses[c_idx]
            
            # Check if any other literal in the clause is satisfied
            others_satisfied = False
            for lit in clause:
                if abs(lit) - 1 == var:
                    continue
                if assignment[abs(lit) - 1] == (1 if lit > 0 else 0):
                    others_satisfied = True
                    break
            
            if others_satisfied:
                # Clause is satisfied regardless of var's value
                continue
            
            # If no other literal is satisfied, status depends solely on var
            # sign=1 means +var (satisfied if var=1)
            # sign=-1 means -var (satisfied if var=0)
            
            current_satisfied = (old_val == 1) if sign == 1 else (old_val == 0)
            new_satisfied = (new_val == 1) if sign == 1 else (new_val == 0)
            
            if current_satisfied and not new_satisfied:
                delta += 1
            elif not current_satisfied and new_satisfied:
                delta -= 1
        
        # Accept or reject
        if delta <= 0:
            assignment[var] = new_val
            current_energy += delta
        else:
            if T > 0:
                # Use log to avoid underflow for large delta/T, though exp is fine here
                if rng.random() < math.exp(-delta / T):
                    assignment[var] = new_val
                    current_energy += delta
        
        # Cool down
        T *= cooling_rate
    
    return [bool(x) for x in assignment]
# EVOLVE-BLOCK-END
