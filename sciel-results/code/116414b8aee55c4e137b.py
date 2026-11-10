import random

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Use a fixed seed for reproducibility, but WalkSAT is stochastic.
    # Different seeds might help if one gets stuck, but fixed is fine for scoring consistency.
    rng = random.Random(42)
    
    # WalkSAT parameters
    p = 0.5 # Probability to choose the best flip (maximizing gain)
    max_restarts = 50
    max_steps = 5000
    
    for restart in range(max_restarts):
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        for step in range(max_steps):
            # Find an unsatisfied clause
            unsat_clause = None
            for c in clauses:
                sat = False
                for lit in c:
                    var = abs(lit) - 1
                    if assignment[var] == (lit > 0):
                        sat = True
                        break
                if not sat:
                    unsat_clause = c
                    break
            
            if unsat_clause is None:
                return assignment
            
            # Calculate gain for each literal in the unsatisfied clause
            best_gain = -100
            best_lit = None
            gains = []
            
            for lit in unsat_clause:
                gain = 0
                # Check all other clauses to determine if they are unsatisfied
                # and how the flip would affect them
                for c2 in clauses:
                    if c2 is unsat_clause:
                        continue
                    
                    # Check if c2 is currently unsatisfied
                    c2_sat = False
                    for l2 in c2:
                        v2 = abs(l2) - 1
                        if assignment[v2] == (l2 > 0):
                            c2_sat = True
                            break
                            
                    if not c2_sat:
                        if lit in c2:
                            gain += 1
                        elif -lit in c2:
                            gain -= 1
                            
                gains.append((gain, lit))
                if gain > best_gain:
                    best_gain = gain
                    best_lit = lit
            
            # Choose which literal to flip based on heuristic
            if rng.random() < p:
                lit_to_flip = best_lit
            else:
                lit_to_flip = rng.choice(unsat_clause)
                
            var = abs(lit_to_flip) - 1
            assignment[var] = not assignment[var]
            
    # If no solution found after all restarts, return the last assignment
    # This is a best-effort fallback, though it should rarely be reached for satisfiable instances
    return assignment
