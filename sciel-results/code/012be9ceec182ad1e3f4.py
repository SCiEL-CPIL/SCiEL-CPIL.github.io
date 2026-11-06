"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 290.0
    best_assignment = None
    best_unsat_count = float('inf')
    
    # Precompute clause structure for speed
    m = len(clauses)
    # For each variable, list of (clause_idx, sign) where sign=+1 means lit is var, -1 means lit is -var
    var_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        for lit in clause:
            vi = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            var_clauses[vi].append((ci, sign))
    
    def is_satisfied(assignment):
        """Check if assignment satisfies all clauses."""
        for clause in clauses:
            sat = False
            for lit in clause:
                idx = abs(lit) - 1
                if assignment[idx] == (lit > 0):
                    sat = True
                    break
            if not sat:
                return False
        return True
    
    def count_unsat(assignment):
        """Count number of unsatisfied clauses."""
        count = 0
        for clause in clauses:
            sat = False
            for lit in clause:
                idx = abs(lit) - 1
                if assignment[idx] == (lit > 0):
                    sat = True
                    break
            if not sat:
                count += 1
        return count
    
    # WalkSAT with efficient delta computation
    # noise parameter controls probability of random flip
    for attempt in range(200):
        if time.time() >= deadline:
            break
            
        rng = random.Random(attempt * 7919 + 42)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        current_unsat = count_unsat(assignment)
        if current_unsat == 0:
            return assignment
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]
        
        max_steps = 3000
        for step in range(max_steps):
            if time.time() >= deadline:
                break
            
            # Find unsatisfied clauses
            unsat_clauses = []
            for ci, clause in enumerate(clauses):
                sat = False
                for lit in clause:
                    idx = abs(lit) - 1
                    if assignment[idx] == (lit > 0):
                        sat = True
                        break
                if not sat:
                    unsat_clauses.append(ci)
            
            if not unsat_clauses:
                return assignment
            
            # Pick a random unsatisfied clause
            ci = unsat_clauses[rng.randrange(len(unsat_clauses))]
            clause = clauses[ci]
            
            # For each literal in the clause, compute delta (change in unsat count if we flip that variable)
            # delta = (# clauses that become unsatisfied) - (# clauses that become satisfied)
            # A clause becomes satisfied if it was unsatisfied and this flip satisfies it
            # A clause becomes unsatisfied if it was satisfied and this flip breaks it (and no other lit satisfies it)
            
            best_var = -1
            best_delta = float('inf')
            
            for lit in clause:
                var_idx = abs(lit) - 1
                current_val = assignment[var_idx]
                
                # Compute delta: how does unsat count change if we flip var_idx?
                delta = 0
                for (c_idx, s) in var_clauses[var_idx]:
                    c = clauses[c_idx]
                    # Determine if this clause is currently satisfied
                    c_sat = False
                    for l in c:
                        idx = abs(l) - 1
                        if idx == var_idx:
                            # This is the literal we're considering flipping
                            if current_val == (l > 0):
                                c_sat = True
                                break
                        else:
                            if assignment[idx] == (l > 0):
                                c_sat = True
                                break
                    
                    # Determine if this clause would be satisfied after flip
                    new_val = not current_val
                    c_sat_after = False
                    for l in c:
                        idx = abs(l) - 1
                        if idx == var_idx:
                            if new_val == (l > 0):
                                c_sat_after = True
                                break
                        else:
                            if assignment[idx] == (l > 0):
                                c_sat_after = True
                                break
                    
                    if c_sat and not c_sat_after:
                        delta += 1  # clause becomes unsatisfied
                    elif not c_sat and c_sat_after:
                        delta -= 1  # clause becomes satisfied
                
                # Add small random perturbation for exploration
                delta += rng.random() * 0.1
                
                if delta < best_delta:
                    best_delta = delta
                    best_var = var_idx
            
            if best_var >= 0:
                assignment[best_var] = not assignment[best_var]
                
                new_unsat = count_unsat(assignment)
                if new_unsat == 0:
                    return assignment
                if new_unsat < best_unsat_count:
                    best_unsat_count = new_unsat
                    best_assignment = assignment[:]
        
        if time.time() >= deadline:
            break
    
    # Return the best assignment found
    if best_assignment is not None:
        return best_assignment
    
    # Fallback: random assignment
    rng = random.Random(0)
    return [rng.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
