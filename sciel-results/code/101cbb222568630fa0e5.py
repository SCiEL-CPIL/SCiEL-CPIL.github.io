"""Simulated Annealing based 3-SAT solver."""
# EVOLVE-BLOCK-START
import random
import math


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(0)
    m = len(clauses)
    
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
    assignment = [rng.randint(0, 1) for _ in range(n)]
    
    # Initialize clause satisfaction counts
    # clause_sat_count[c] is the number of satisfied literals in clause c
    clause_sat_count = [0] * m
    for c_idx, clause in enumerate(clauses):
        count = 0
        for lit in clause:
            var_idx = abs(lit) - 1
            val = assignment[var_idx]
            # If lit is positive, satisfied if val==1. If negative, satisfied if val==0.
            if (lit > 0 and val == 1) or (lit < 0 and val == 0):
                count += 1
        clause_sat_count[c_idx] = count

    # List of unsatisfied clause indices
    unsat_clauses = [i for i in range(m) if clause_sat_count[i] == 0]
    
    if not unsat_clauses:
        return [bool(x) for x in assignment]

    # Simulated Annealing parameters
    T_start = 5.0
    T_end = 0.01
    max_iterations = 500000
    
    cooling_rate = (T_end / T_start) ** (1.0 / max_iterations)
    T = T_start
    
    for iteration in range(max_iterations):
        if not unsat_clauses:
            break
        
        # Pick a random unsatisfied clause
        c_idx = unsat_clauses[rng.randint(0, len(unsat_clauses) - 1)]
        clause = clauses[c_idx]
        
        # Pick a random variable from this clause
        lit = clause[rng.randint(0, len(clause) - 1)]
        var = abs(lit) - 1
        
        old_val = assignment[var]
        new_val = 1 - old_val
        
        delta = 0
        # We need to update clause_sat_count and track which clauses change status
        # We'll store the updates to apply later if accepted
        
        updates = [] # List of (c_idx, old_count, new_count)
        
        for c, sign in var_to_clauses[var]:
            old_count = clause_sat_count[c]
            
            # Determine if the literal for 'var' in clause 'c' is satisfied by old_val and new_val
            # sign is 1 for +var, -1 for -var
            # Satisfied if (sign==1 and val==1) or (sign==-1 and val==0)
            
            old_sat = 1 if (sign == 1 and old_val == 1) or (sign == -1 and old_val == 0) else 0
            new_sat = 1 if (sign == 1 and new_val == 1) or (sign == -1 and new_val == 0) else 0
            
            new_count = old_count + new_sat - old_sat
            
            # Determine change in satisfaction status (0 or 1)
            old_satisfied = 1 if old_count > 0 else 0
            new_satisfied = 1 if new_count > 0 else 0
            
            if old_satisfied == 1 and new_satisfied == 0:
                delta += 1
            elif old_satisfied == 0 and new_satisfied == 1:
                delta -= 1
            
            # Store update
            updates.append((c, old_count, new_count))
        
        # Accept or reject
        accepted = False
        if delta <= 0:
            accepted = True
        else:
            if T > 0:
                if rng.random() < math.exp(-delta / T):
                    accepted = True
        
        if accepted:
            assignment[var] = new_val
            
            # Apply updates to clause_sat_count and unsat_clauses
            for c, old_count, new_count in updates:
                # Only update if count changed
                if old_count != new_count:
                    # Update unsat_clauses list
                    if old_count == 0 and new_count > 0:
                        # Clause became satisfied, remove from unsat_clauses
                        try:
                            unsat_clauses.remove(c)
                        except ValueError:
                            pass
                    elif old_count > 0 and new_count == 0:
                        # Clause became unsatisfied, add to unsat_clauses
                        unsat_clauses.append(c)
                    
                    clause_sat_count[c] = new_count
        
        # Cool down
        T *= cooling_rate
    
    return [bool(x) for x in assignment]
# EVOLVE-BLOCK-END
