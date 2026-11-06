"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 290.0
    best_assignment = None
    best_unsat_count = float('inf')
    
    def evaluate(assignment):
        """Count unsatisfied clauses and return (count, is_satisfied)."""
        unsat_count = 0
        for clause in clauses:
            sat = False
            for lit in clause:
                idx = abs(lit) - 1
                if assignment[idx] == (lit > 0):
                    sat = True
                    break
            if not sat:
                unsat_count += 1
                if unsat_count >= best_unsat_count:
                    return unsat_count, False
        return unsat_count, (unsat_count == 0)

    # Try multiple random restarts with WalkSAT
    for attempt in range(100):
        if time.time() >= deadline:
            break
            
        rng = random.Random(attempt * 1000 + 42)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        current_unsat, is_sat = evaluate(assignment)
        if is_sat:
            return assignment
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]
        
        # WalkSAT with adaptive noise
        max_steps = 5000
        for step in range(max_steps):
            if time.time() >= deadline:
                break
                
            # Find unsatisfied clauses
            unsat_clauses = []
            for clause in clauses:
                sat = False
                for lit in clause:
                    idx = abs(lit) - 1
                    if assignment[idx] == (lit > 0):
                        sat = True
                        break
                if not sat:
                    unsat_clauses.append(clause)
            
            if not unsat_clauses:
                return assignment
            
            # Pick a random unsatisfied clause
            clause = rng.choice(unsat_clauses)
            
            # Count how many literals in the clause would be satisfied if flipped
            # Use min-conflicts heuristic: flip the variable that minimizes increase in unsat
            best_var = -1
            best_delta = float('inf')
            best_count = float('inf')
            
            for lit in clause:
                var_idx = abs(lit) - 1
                # Count how many clauses this variable appears in
                # and how many would become unsatisfied if we flip
                flip_count = 0
                for c in clauses:
                    # Check if this clause is currently satisfied
                    c_sat = False
                    for l in c:
                        if assignment[abs(l) - 1] == (l > 0):
                            c_sat = True
                            break
                    
                    # Check if this clause would be satisfied after flip
                    c_sat_after = False
                    for l in c:
                        if abs(l) - 1 == var_idx:
                            # This literal's value changes
                            if not (assignment[abs(l) - 1] == (l > 0)):
                                # Currently false, would be true after flip
                                c_sat_after = True
                                break
                        else:
                            if assignment[abs(l) - 1] == (l > 0):
                                c_sat_after = True
                                break
                    
                    if c_sat and not c_sat_after:
                        flip_count += 1
                
                if flip_count < best_delta or (flip_count == best_delta and rng.random() < 0.5):
                    best_delta = flip_count
                    best_var = var_idx
            
            if best_var >= 0:
                assignment[best_var] = not assignment[best_var]
                
                # Evaluate new state
                new_unsat, new_sat = evaluate(assignment)
                if new_sat:
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
